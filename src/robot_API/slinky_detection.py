
"""
slinky_cv.py
============
Slinky Vision API — Intel RealSense + ArUco marker tracking.

Changes from the original
--------------------------
**Stability gate (new)**
    The original ``get_positions()`` returned a single-frame reading with no
    convergence check.  After a robot move the camera vibrates for 1–3 s,
    producing noisy readings even if a ``time.sleep()`` is used.

    New method ``get_stable_positions()`` replaces the ad-hoc sleep:
      - It polls the camera in a tight loop.
      - It maintains a sliding window of the last N readings per marker.
      - It returns only when per-axis std-dev across the window drops below a
        configurable threshold, OR a timeout is reached (in which case the
        running mean is returned with ``stability_reached=False`` in the dict).
      - This makes every recorded pose genuinely stable rather than "waited
        long enough".

**3-D pose (preserved)**
    ``get_positions()`` still estimates full 6-DoF pose via
    ``cv2.aruco.estimatePoseSingleMarkers`` and returns (x, y, z) in the
    camera frame (metres).  ``get_stable_positions()`` returns the same
    schema, augmented with per-marker ``std_x/y/z``, ``n_frames``, and a
    top-level ``stability_reached`` bool.

**calibrate() (preserved)**
    The existing ``calibrate(num_frames)`` method is unchanged and continues
    to work as a multi-frame averaging / warm-up tool.

**All existing public methods preserved**
    ``start_camera``, ``stop_camera``, ``get_frame``, ``detect_markers``,
    ``get_marker_center``, ``create_annotated_frame``, ``get_positions``,
    ``get_positions_json``, ``calibrate``, ``get_camera_info``.

**Module-level ``get_positions()`` helper (preserved)**
    The module-level convenience function is kept for compatibility with
    callers that do not want to manage the ``SlinkyVisionAPI`` lifetime.
"""

from __future__ import annotations

import base64
import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path as _Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image


# ---------------------------------------------------------------------------
# Module-level data path
# ---------------------------------------------------------------------------

DATA_PATH: str = os.environ.get(
    "DATA_PATH",
    str(_Path(__file__).resolve().parent / "data"),
)

def _marker_key(marker_id: int) -> str:
    """Return the canonical key for a given ArUco marker ID, e.g. ``marker_0``."""
    return f"marker_{marker_id}"


# ---------------------------------------------------------------------------
# SlinkyVisionAPI
# ---------------------------------------------------------------------------

class SlinkyVisionAPI:
    """
    Computer-vision API for tracking slinky positions using ArUco markers
    and an Intel RealSense D405 camera.

    Marker IDs
    ----------
    Each detected ArUco marker is identified by its own ID.  The result dict
    key for marker ID *N* is ``marker_N`` (e.g. ``marker_0``, ``marker_1``).

    Typical lifecycle
    -----------------
    ::

        api = SlinkyVisionAPI(headless=True)
        api.start_camera()

        # During data collection – blocks until markers are stable:
        result = api.get_stable_positions()

        # Raw single-frame read (no stability guarantee):
        result = api.get_positions()

        api.stop_camera()

    Stability parameters
    --------------------
    Pass ``stability_window``, ``stability_threshold_m``, and
    ``stability_timeout_sec`` to the constructor to tune the gate:

    - ``stability_window`` (int, default 8):
          Number of consecutive frames over which std-dev is measured.
    - ``stability_threshold_m`` (float, default 0.002):
          Per-axis std-dev threshold in **metres**.  0.002 m = 2 mm is a
          reasonable value for a 20 mm marker at ~0.5 m range.
    - ``stability_timeout_sec`` (float, default 5.0):
          Wall-clock seconds before giving up and returning the running mean.
    - ``stability_poll_sec`` (float, default 0.05):
          Sleep between camera frames inside the polling loop (~20 Hz).
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        aruco_dict_type: int = cv2.aruco.DICT_6X6_50,
        headless: bool = False,
        # ── Stability gate parameters ──────────────────────────────────
        stability_window: int = 8,
        stability_threshold_m: float = 0.002,
        stability_timeout_sec: float = 5.0,
        stability_poll_sec: float = 0.05,
        live_preview: bool = True,
        live_preview_interval_sec: float = 0.5,
    ) -> None:
        # Camera parameters
        self.width = width
        self.height = height
        self.fps = fps
        self.headless = headless

        # RealSense handles (filled by start_camera)
        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

        # ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        # self.aruco_params = cv2.aruco.DetectorParameters()
        # self.aruco_params.adaptiveThreshConstant = 7        # default ~7, lower for dark images
        self.aruco_params.polygonalApproxAccuracyRate = 0.07 # default 0.05, increase for distorted markers
        self.aruco_params.errorCorrectionRate = 0.7          # default 0.6, increase to tolerate bit errors

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Stability gate parameters
        self._stab_window = stability_window
        self._stab_threshold = stability_threshold_m
        self._stab_timeout = stability_timeout_sec
        self._stab_poll = stability_poll_sec
        self._live_preview = live_preview
        self._live_preview_interval = live_preview_interval_sec
        self._last_preview_time: float = 0.0

        # Populated at runtime: set of integer marker IDs seen so far
        self._known_marker_ids: set = set()

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------

    def start_camera(self) -> bool:
        """
        Start the RealSense pipeline and extract camera intrinsics.

        Returns ``True`` on success.  Raises ``RuntimeError`` if the camera
        cannot be opened.
        """
        if self.pipeline is not None:
            print("Camera already started")
            return True

        os.makedirs(DATA_PATH, exist_ok=True)

        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
            )
            profile = self.pipeline.start(self.config)

            color_stream = (
                profile.get_stream(rs.stream.color).as_video_stream_profile()
            )
            intr = color_stream.get_intrinsics()
            self.camera_matrix = np.array(
                [[intr.fx, 0.0, intr.ppx],
                 [0.0, intr.fy, intr.ppy],
                 [0.0, 0.0,    1.0      ]],
                dtype=np.float32,
            )
            self.dist_coeffs = np.array(intr.coeffs, dtype=np.float32)

            print(
                f"✓ RealSense camera started "
                f"({self.width}×{self.height} @ {self.fps} fps)"
            )
            return True
        except Exception as exc:
            raise RuntimeError(f"Failed to start RealSense camera: {exc}") from exc

    def stop_camera(self) -> None:
        """Stop the RealSense pipeline and release resources."""
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
            self.config = None
            print("✓ RealSense camera stopped")

    def __del__(self) -> None:
        self.stop_camera()
        if not self.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Low-level frame / detection helpers
    # ------------------------------------------------------------------

    def get_frame(self) -> np.ndarray:
        """
        Capture and return one BGR frame from the camera.

        Raises ``RuntimeError`` if the camera is not started or if the frame
        cannot be captured.
        """
        if self.pipeline is None:
            raise RuntimeError("Camera not started. Call start_camera() first.")
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to capture colour frame from RealSense.")
        return np.asanyarray(color_frame.get_data())

    def _ensure_intrinsics(self) -> None:
        """Raise if camera intrinsics are not yet available."""
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise RuntimeError(
                "Camera intrinsics not initialised. "
                "Make sure start_camera() completed successfully."
            )

    def detect_markers(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run ArUco detection on *frame*.

        Returns ``(corners, ids, rejected)`` exactly as
        ``ArucoDetector.detectMarkers`` does.
        """
        return self.detector.detectMarkers(frame)

    def get_marker_center(self, corners: np.ndarray) -> Tuple[float, float]:
        """
        Return the pixel centroid ``(u, v)`` of a single detected marker.

        Parameters
        ----------
        corners:
            The corners array for **one** marker (shape ``(1, 4, 2)``).
        """
        return (
            float(np.mean(corners[0][:, 0])),
            float(np.mean(corners[0][:, 1])),
        )

    def create_annotated_frame(
        self,
        frame: np.ndarray,
        corners,
        ids,
        show_timestamp: bool = True,
    ) -> np.ndarray:
        """
        Draw detected markers, labels, and an optional timestamp onto *frame*.

        Returns a copy of the frame with overlays.
        """
        vis = frame.copy()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            for i, marker_id in enumerate(ids.flatten()):
                u, v = self.get_marker_center(corners[i])
                cv2.circle(vis, (int(u), int(v)), 5, (0, 255, 0), -1)
                label = _marker_key(marker_id)
                cv2.putText(
                        vis,
                        f"{label} ({int(u)}, {int(v)})",
                        (int(u) + 10, int(v) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                    )

        if show_timestamp:
            cv2.putText(
                vis, time.strftime("%Y-%m-%d %H:%M:%S"),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

        n_detected = len(ids) if ids is not None else 0
        cv2.putText(
            vis, f"Detected: {n_detected}/5 markers",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        return vis

    # ------------------------------------------------------------------
    # Single-frame position read
    # ------------------------------------------------------------------

    def get_positions(
        self,
        visualize: bool = False,
        save_image: Optional[str] = None,
        return_annotated_b64: bool = False,
    ) -> Dict:
        """
        Capture **one** frame and return marker 3-D positions.

        .. warning::
            This method provides **no stability guarantee**.  After a robot
            move, camera vibration can make single-frame readings noisy.
            Use ``get_stable_positions()`` during data collection.

        Returns
        -------
        Dict with keys:

        - ``marker_N`` for each detected ArUco ID *N*:
              ``{"x": float, "y": float, "z": float}`` in metres (camera
              frame), or ``None`` if the marker was not detected.
        - ``frame_shape``: ``{"width": int, "height": int}``
        - ``detected_markers``: list of detected integer IDs
        - ``saved_path``: path where ``latest_frame.png`` was written
        - (optional) ``annotated_frame_b64``: base-64 PNG string
        """
        if self.pipeline is None:
            self.start_camera()
        self._ensure_intrinsics()

        marker_length_m = 0.020  # physical marker side length

        frame = self.get_frame()
        corners, ids, _ = self.detect_markers(frame)

        result: Dict = {
            "frame_shape":      {"width": frame.shape[1], "height": frame.shape[0]},
            "detected_markers": [],
        }

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                label = _marker_key(marker_id)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    np.array([corners[i]]),
                    marker_length_m,
                    self.camera_matrix,
                    self.dist_coeffs,
                )
                tx, ty, tz = tvecs[0][0].astype(float)
                result[label] = {"x": float(tx), "y": float(ty), "z": float(tz)}
                result["detected_markers"].append(int(marker_id))

        vis_frame = self.create_annotated_frame(frame, corners, ids)

        if self._live_preview and self._should_write_preview():
            try:
                latest_path = os.path.join(DATA_PATH, "latest_frame.png")
                Image.fromarray(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)).save(latest_path)
                result["saved_path"] = latest_path
            except Exception as exc:
                result["save_error"] = f"{type(exc).__name__}: {exc}"

        if save_image:
            try:
                Image.fromarray(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)).save(save_image)
                result["saved_image_path"] = save_image
            except Exception as exc:
                result["save_image_error"] = f"{type(exc).__name__}: {exc}"

        if return_annotated_b64:
            try:
                _, buf = cv2.imencode(".png", vis_frame)
                result["annotated_frame_b64"] = base64.b64encode(
                    buf.tobytes()
                ).decode("ascii")
            except Exception as exc:
                result["annotated_frame_error"] = str(exc)

        if visualize and not self.headless:
            try:
                cv2.imshow("Slinky Vision - RealSense", vis_frame)
                cv2.waitKey(1)
            except cv2.error as exc:
                print(f"Warning: cannot display window: {exc}")

        return result

    # ------------------------------------------------------------------
    # Stable position read  ← NEW
    # ------------------------------------------------------------------

    def get_stable_positions(
        self,
        window: Optional[int] = None,
        threshold_m: Optional[float] = None,
        timeout_sec: Optional[float] = None,
        save_image: Optional[str] = None,
    ) -> Dict:
        """
        Poll the camera until all **currently visible** markers are stable,
        then return their windowed-mean positions.

        This method replaces the ``time.sleep(STABILIZE_DELAY_SEC)`` pattern
        in the original collection script.  It adapts automatically: if the
        scene settles quickly it returns early; if it takes longer it waits up
        to *timeout_sec* before falling back to the running mean.

        Algorithm
        ---------
        1. Capture a frame and append each detected marker's (x, y, z) to a
           per-marker sliding window (``deque`` of length *window*).
        2. A marker is **stable** when its window is full AND the per-axis
           std-dev is below *threshold_m*.
        3. Once every marker that has been seen **at least once** is stable,
           return the windowed means.
        4. On timeout, return the running means with
           ``stability_reached = False``.

        Parameters
        ----------
        window:
            Override the instance's ``stability_window``.
        threshold_m:
            Override the instance's ``stability_threshold_m`` (metres).
        timeout_sec:
            Override the instance's ``stability_timeout_sec`` (seconds).
        save_image:
            If given, save the annotated frame from the **last** polled
            frame to this path.

        Returns
        -------
        Dict with the same marker keys as ``get_positions()``, plus:

        - Per-marker dict (when detected):
              ``{"x", "y", "z", "std_x", "std_y", "std_z",
              "n_frames", "stable"}``
        - ``stability_reached`` (bool): ``True`` if all markers converged.
        - ``elapsed_sec`` (float): wall-clock seconds spent polling.
        - ``frame_shape``, ``detected_markers`` (same as ``get_positions()``).
        """
        if self.pipeline is None:
            self.start_camera()
        self._ensure_intrinsics()

        win     = window      if window      is not None else self._stab_window
        thresh  = threshold_m if threshold_m is not None else self._stab_threshold
        timeout = timeout_sec if timeout_sec is not None else self._stab_timeout

        # Per-marker sliding windows keyed by marker_N label
        from collections import defaultdict
        windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=win))
        seen: set = set()
        frame_shape: Dict = {"width": self.width, "height": self.height}
        last_vis_frame = None

        deadline = time.monotonic() + timeout
        marker_length_m = 0.020

        while time.monotonic() < deadline:
            frame = self.get_frame()
            frame_shape = {"width": frame.shape[1], "height": frame.shape[0]}
            corners, ids, _ = self.detect_markers(frame)
            last_vis_frame = self.create_annotated_frame(frame, corners, ids)

            if ids is not None and len(ids) == 5:
                for i, marker_id in enumerate(ids.flatten()):
                    label = _marker_key(marker_id)
                    seen.add(label)
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        np.array([corners[i]]),
                        marker_length_m,
                        self.camera_matrix,
                        self.dist_coeffs,
                    )
                    tx, ty, tz = tvecs[0][0].astype(float)
                    windows[label].append(
                        {"x": float(tx), "y": float(ty), "z": float(tz)}
                    )

            if self._live_preview and self._should_write_preview():
                try:
                    Image.fromarray(
                        cv2.cvtColor(last_vis_frame, cv2.COLOR_BGR2RGB)
                    ).save(os.path.join(DATA_PATH, "latest_frame.png"))
                except Exception:
                    pass

            if seen and self._windows_stable(windows, seen, win, thresh):
                elapsed = timeout - max(0.0, deadline - time.monotonic())
                return self._build_stable_result(
                    windows, seen, frame_shape, True, elapsed, save_image, last_vis_frame
                )

            time.sleep(self._stab_poll)

        # Timeout path
        elapsed = timeout
        print(
            f"  ⚠ Stability timeout ({timeout:.1f}s). "
            f"Returning running mean for: {sorted(seen) or 'none'}."
        )
        return self._build_stable_result(
            windows, seen, frame_shape, False, elapsed, save_image, last_vis_frame
        )

    # ------------------------------------------------------------------
    # Stability helpers  (private)
    # ------------------------------------------------------------------

    def _should_write_preview(self) -> bool:
        """Return True and update timestamp if enough time has passed since the last preview write."""
        now = time.monotonic()
        if now - self._last_preview_time >= self._live_preview_interval:
            self._last_preview_time = now
            return True
        return False

    @staticmethod
    def _windows_stable(
        windows: Dict[str, deque],
        seen: set,
        win: int,
        thresh: float,
    ) -> bool:
        """Return ``True`` if every seen marker's window is full and within threshold."""
        for key in seen:
            dq = windows[key]
            if len(dq) < win:
                return False
            if (
                float(np.std([o["x"] for o in dq])) > thresh
                or float(np.std([o["y"] for o in dq])) > thresh
                or float(np.std([o["z"] for o in dq])) > thresh
            ):
                return False
        return True

    @staticmethod
    def _build_stable_result(
        windows: Dict[str, deque],
        seen: set,
        frame_shape: Dict,
        stability_reached: bool,
        elapsed_sec: float,
        save_image: Optional[str],
        vis_frame: Optional[np.ndarray],
    ) -> Dict:
        """Assemble the return dict from windowed observations."""
        result: Dict = {
            "frame_shape":        frame_shape,
            "detected_markers":   [],
            "stability_reached":  stability_reached,
            "elapsed_sec":        round(elapsed_sec, 3),
        }
        for key in seen:
            dq = windows[key]
            if len(dq) > 0:
                xs = [o["x"] for o in dq]
                ys = [o["y"] for o in dq]
                zs = [o["z"] for o in dq]
                result[key] = {
                    "x":     float(np.mean(xs)),
                    "y":     float(np.mean(ys)),
                    "z":     float(np.mean(zs)),
                    "std_x": float(np.std(xs)),
                    "std_y": float(np.std(ys)),
                    "std_z": float(np.std(zs)),
                    "stable": stability_reached,
                }
                # Recover the integer marker ID from the key name
                result["detected_markers"].append(int(key.split("_", 1)[1]))

        if save_image and vis_frame is not None:
            try:
                Image.fromarray(
                    cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                ).save(save_image)
                result["saved_image_path"] = save_image
            except Exception as exc:
                result["save_image_error"] = str(exc)

        return result

    # ------------------------------------------------------------------
    # Existing helpers (unchanged API)
    # ------------------------------------------------------------------

    def get_positions_json(
        self,
        visualize: bool = False,
        save_image: Optional[str] = None,
    ) -> str:
        """Return ``get_positions()`` result as a JSON string."""
        return json.dumps(
            self.get_positions(visualize=visualize, save_image=save_image), indent=2
        )

    def calibrate(
        self, num_frames: int = 30, save_final_image: Optional[str] = None
    ) -> Dict:
        """
        Average marker positions over *num_frames* frames.

        Returns a dict with per-marker ``{"x", "y", "std_x", "std_y"}`` (or
        ``None``) plus ``"frame_shape"``.

        .. note::
            This method is kept for backward compatibility and warm-up use.
            For per-pose stability during collection, use
            ``get_stable_positions()`` instead.
        """
        print(f"Calibrating over {num_frames} frames …")
        positions_list: List[Dict] = []

        for i in range(num_frames):
            save_img = save_final_image if i == num_frames - 1 else None
            positions_list.append(self.get_positions(save_image=save_img))
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{num_frames}")

        # Collect all marker keys seen across all frames
        skip_keys = {"frame_shape", "detected_markers", "saved_path", "save_error"}
        all_keys = {k for p in positions_list for k in p if k not in skip_keys}

        averaged: Dict = {"frame_shape": positions_list[0]["frame_shape"]}
        for key in sorted(all_keys):
            valid = [p[key] for p in positions_list if p.get(key) is not None]
            if valid:
                averaged[key] = {
                    "x":     float(np.mean([p["x"] for p in valid])),
                    "y":     float(np.mean([p["y"] for p in valid])),
                    "std_x": float(np.std( [p["x"] for p in valid])),
                    "std_y": float(np.std( [p["y"] for p in valid])),
                }
            else:
                averaged[key] = None
        print("✓ Calibration complete")
        return averaged

    def get_camera_info(self) -> Dict:
        """Return a dict of camera hardware and configuration information."""
        if self.pipeline is None:
            return {"error": "Camera not started"}
        try:
            profile = self.pipeline.get_active_profile()
            device  = profile.get_device()
            return {
                "name":             device.get_info(rs.camera_info.name),
                "serial_number":    device.get_info(rs.camera_info.serial_number),
                "firmware_version": device.get_info(rs.camera_info.firmware_version),
                "resolution":       f"{self.width}×{self.height}",
                "fps":              self.fps,
                "headless_mode":    self.headless,
                "data_path":        DATA_PATH,
                "stability_window":     self._stab_window,
                "stability_threshold_m": self._stab_threshold,
                "stability_timeout_sec": self._stab_timeout,
            }
        except Exception as exc:
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Module-level convenience function (backward compatible)
# ---------------------------------------------------------------------------

def get_positions(
    width: int = 640,
    height: int = 480,
    headless: bool = True,
    return_annotated_b64: bool = False,
) -> Dict:
    """
    One-shot helper: start camera → capture one frame → stop camera.

    .. warning::
        This creates and destroys the camera pipeline on every call.  For
        tight collection loops, instantiate ``SlinkyVisionAPI`` once and call
        ``api.get_stable_positions()`` directly.

    Returns
    -------
    Same dict schema as ``SlinkyVisionAPI.get_positions()``.
    """
    api = SlinkyVisionAPI(width=width, height=height, headless=headless)
    try:
        api.start_camera()
        return api.get_positions(return_annotated_b64=return_annotated_b64)
    except Exception as exc:
        import traceback
        err = str(exc)
        print(f"ERROR in get_positions: {err}")
        print(traceback.format_exc())
        return {
            "frame_shape":      {},
            "detected_markers": [],
            "error":            err,
        }
    finally:
        api.stop_camera()


# ---------------------------------------------------------------------------
# CLI test harness (unchanged behaviour, minor clean-up)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import select
    import sys

    parser = argparse.ArgumentParser(description="Slinky Vision API – test mode")
    parser.add_argument("--headless",          action="store_true")
    parser.add_argument("--save-dir",          default="slinky_snapshots")
    parser.add_argument("--live-snapshot",     action="store_true")
    parser.add_argument("--snapshot-interval", type=int, default=1)
    parser.add_argument("--stable",            action="store_true",
                        help="Use get_stable_positions() instead of get_positions()")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("Slinky Vision API – RealSense – Test Mode")
    print("=" * 50)
    print(f"Frame output : {DATA_PATH}/latest_frame.png")
    print(f"Stable mode  : {args.stable}")
    if args.headless:
        print("HEADLESS mode")
    print()

    api = SlinkyVisionAPI(width=640, height=480, fps=30, headless=args.headless)


    try:
        api.start_camera()
        print("Camera Info:")
        for k, v in api.get_camera_info().items():
            print(f"  {k}: {v}")
        print("\nCommands: q=quit  s=snapshot  c=calibrate\n")

        frame_count = 0
        snapshots_saved = 0


        while True:
            save_path = None
            if args.live_snapshot and frame_count % args.snapshot_interval == 0:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = os.path.join(args.save_dir, f"frame_{ts}.jpg")

            if args.stable:
                positions = api.get_stable_positions(save_image=save_path)
            else:
                positions = api.get_positions(save_image=save_path)

            frame_count += 1
            if save_path:
                snapshots_saved += 1

            if frame_count % 10 == 0:
                print("\n" + "=" * 50)
                print(f"Frame {frame_count} – detected: {positions['detected_markers']}", end="")
                if args.stable:
                    print(f"  stable={positions.get('stability_reached')}  "
                          f"elapsed={positions.get('elapsed_sec')}s", end="")
                if args.live_snapshot:
                    print(f"  snapshots={snapshots_saved}", end="")
                print()
                skip = {"frame_shape", "detected_markers", "saved_path",
                        "save_error", "stability_reached", "elapsed_sec",
                        "annotated_frame_b64", "annotated_frame_error",
                        "saved_image_path", "save_image_error"}
                for key in sorted(k for k in positions if k not in skip):
                    pos = positions[key]
                    if pos:
                        print(f"  {key:15s}: x={pos['x']:7.4f}  y={pos['y']:7.4f}  z={pos['z']:7.4f}")
                    else:
                        print(f"  {key:15s}: NOT DETECTED")

            if select.select([sys.stdin], [], [], 0)[0]:
                cmd = sys.stdin.readline().strip()
                if cmd == "q":
                    print("Quitting …")
                    break
                elif cmd == "s":
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    snap = os.path.join(args.save_dir, f"manual_{ts}.jpg")
                    api.get_positions(save_image=snap)
                    print(f"✓ Snapshot: {snap}")
                    with open(os.path.join(args.save_dir, f"manual_{ts}.json"), "w") as fh:
                        json.dump(positions, fh, indent=2)
                elif cmd == "c":
                    print("Calibrating …")
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cal = api.calibrate(
                        num_frames=30,
                        save_final_image=os.path.join(args.save_dir, f"calibrated_{ts}.jpg"),
                    )
                    skip_cal = {"frame_shape"}
                    for key in sorted(k for k in cal if k not in skip_cal):
                        pos = cal[key]
                        if pos:
                            print(f"  {key:15s}: x={pos['x']:.4f}±{pos['std_x']:.4f}  "
                                  f"y={pos['y']:.4f}±{pos['std_y']:.4f}")
                    with open(os.path.join(args.save_dir, f"calibrated_{ts}.json"), "w") as fh:
                        json.dump(cal, fh, indent=2)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as exc:
        import traceback
        print(f"✗ Error: {exc}")
        traceback.print_exc()
    finally:
        api.stop_camera()
        if args.live_snapshot:
            print(f"✓ Total snapshots: {snapshots_saved}")
        print("Camera closed.")
