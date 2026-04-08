from __future__ import annotations

import argparse
import os
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project path bootstrap
# ---------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASE / "robot_API"))  # for sawyer_controller.py
# sys.path.insert(0, str(_BASE / "piper_sdk/piper_sdk/custom_controller"))


# ═══════════════════════════════════════════════════════════════════════════
# 1 ▸ ExperimentConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    """
    Single source of truth for all experiment parameters.
    Modify this class (or pass CLI flags) to change any experiment setting.
    """

    # ── Naming / output ────────────────────────────────────────────────
    experiment_name: str = "slinky"
    output_dir: Path = field(default_factory=lambda: Path("."))

    # ── Robot workspace limits ─────────────────────────────────────────
    workspace_x_min_mm: float =  400.0
    workspace_x_max_mm: float =  600.0
    workspace_z_min_mm: float =  100.0
    workspace_z_max_mm: float =  800.0
    workspace_y_min_mm: float =  100.0
    workspace_y_max_mm: float =  700.0   

    # ── Home pose ──────────────────────────────────────────────────────
    # home_x_mm: float = 0.0
    # home_z_mm: float = 400.0
    home_timeout_sec: float = 20.0

    # ── Motion ────────────────────────────────────────────────────────
    default_step_mm: float = 50.0
    move_timeout_sec: float = 15.0
    joint_speed: float = 0.01  # (SawyerArmController joint speed, 0.0-1.0.  Lower is slower and more stable.)

    # ── Stability gate ─────────────────────────────────────────────────
    stability_window: int = 8
    stability_threshold_m: float = 0.003
    stability_timeout_sec: float = 5.0
    stability_poll_sec: float = 0.05

    # ── Camera ────────────────────────────────────────────────────────
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # ── Google Drive ──────────────────────────────────────────────────
    drive_library_root: str = "DLO_Library"
    drive_credentials_file: str = "credentials.json"
    drive_token_file: str = "token.json"

    # drive_target_folder_id: str = "1sba59QFwaT6yMKejjck0-49RKjyyokpj"
    drive_target_folder_id: str = "1sba59QFwaT6yMKejjck0-49RKjyyokpj"
    # This is Jiawen Wang's personal google drive "DLO_Library" folder.  Change if you want it to go somewhere else.
 
    # ── Helpers ───────────────────────────────────────────────────────
    def make_output_path(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / "dataset" / f"{self.experiment_name}_data_{ts}.json"

    def make_config_dir_path(self, json_path: Path) -> Path:
        """
        Return a config folder path that sits beside the dataset/ folder,
        named after the JSON file (without extension).

        Example
        -------
        json_path  : …/output_dir/dataset/slinky_data_20240101_120000.json
        config_dir : …/output_dir/configs/slinky_data_20240101_120000/
        """
        configs_root = json_path.parent.parent / "configs"
        return configs_root / json_path.stem

    def inside_workspace(self, x_mm: float, y_mm: float, z_mm: float) -> bool:
        return (
            self.workspace_x_min_mm <= x_mm <= self.workspace_x_max_mm
            and self.workspace_y_min_mm <= y_mm <= self.workspace_y_max_mm
            and self.workspace_z_min_mm <= z_mm <= self.workspace_z_max_mm
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2 ▸ RobotController
# ═══════════════════════════════════════════════════════════════════════════

_DIRECTION_MAP: Dict[str, Tuple[int, int, int]] = {
    # (dx, dy, dz) – Y is the lateral axis, X/Z are the original plane axes
    "z+":    ( 0,  0, +1),
    "z-":  ( 0,  0, -1),
    "x+":  (+1,  0,  0),
    "x-": (-1,  0,  0),
    "y+":  ( 0, +1,  0),
    "y-":  ( 0, -1,  0),
}


class RobotController:
    """
    Wraps SawyerArmController and adds:
    - Workspace boundary enforcement.
    - reset_to_home().
    - Trajectory simulation mode (dry-run without IK).    ← NEW
    - Debug mode that skips real motion.
    """

    def __init__(self, config: ExperimentConfig, debug: bool = False) -> None:
        self.config = config
        self.debug  = debug
        self.arm    = None

        # ── Simulation state ────────────────────────────────────────── ← NEW
        self._sim_mode:       bool       = False
        self._sim_x:          float      = 0.0
        self._sim_y:          float      = 0.0
        self._sim_z:          float      = 0.0
        self._sim_trajectory: List[Dict] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def enable(self) -> None:
        if self.debug:
            print("  [DEBUG] Robot enable skipped.")
            return
        from sawyer_controller import SawyerArmController
        self.arm = SawyerArmController(
            limb="right", init_node=False, joint_speed=self.config.joint_speed
        )
        self.arm.enable()
        print("✓ Robot enabled")

    def disable(self) -> None:
        if self.arm is not None:
            self.arm.disable()
            print("✓ Robot disabled")

    # ------------------------------------------------------------------
    # Position read-back
    # ------------------------------------------------------------------

    def get_position(self) -> Tuple[float, float, float]:
        """Return (x_mm, y_mm, z_mm).  In sim mode, returns the simulated cursor."""
        if self._sim_mode:                             # ← NEW
            return self._sim_x, self._sim_y, self._sim_z
        if self.debug:
            return 0.0, 0.0, 400.0
        return self.arm.get_position()

    # ------------------------------------------------------------------
    # Motion  (shared by real run AND simulation)
    # ------------------------------------------------------------------

    def move_relative(self, dx_mm: float, dy_mm: float, dz_mm: float) -> bool:
        """
        Move by (dx_mm, dy_mm, dz_mm).

        In simulation mode: records the target point, advances the internal
        cursor, and returns immediately – no IK, no real motion.
        In normal mode: checks workspace limits, then calls IK.
        """
        current_x, current_y, current_z = self.get_position()
        target_x = current_x + dx_mm
        target_y = current_y + dy_mm
        target_z = current_z + dz_mm
        valid = self.config.inside_workspace(target_x, target_y, target_z)

        # ── Simulation branch ────────────────────────────────────────── ← NEW
        if self._sim_mode:
            self._sim_x = target_x
            self._sim_y = target_y
            self._sim_z = target_z
            self._sim_trajectory.append({
                "x_mm":  round(target_x, 2),
                "y_mm":  round(target_y, 2),
                "z_mm":  round(target_z, 2),
                "valid": valid,
            })
            if not valid:
                print(
                    f"  ⚠ [PREVIEW] Step would exceed workspace: "
                    f"({target_x:.1f}, {target_y:.1f}, {target_z:.1f}) mm"
                )
            return valid

        # ── Normal / debug branch ────────────────────────────────────────
        if not valid:
            print(
                f"  ✗ Move vetoed – target ({target_x:.1f}, {target_y:.1f}, {target_z:.1f}) mm "
                f"is outside workspace bounds."
            )
            return False

        if self.debug:
            print(
                f"  [DEBUG] Would move dx={dx_mm:+.1f} dy={dy_mm:+.1f} dz={dz_mm:+.1f} "
                f"→ ({target_x:.1f}, {target_y:.1f}, {target_z:.1f}) mm"
            )
            return True

        ok = self.arm.move_relative(
            dx_mm=dx_mm, dy_mm=dy_mm, dz_mm=dz_mm,
            timeout=self.config.move_timeout_sec
        )
        if not ok:
            print("  ✗ IK failed.")
        return ok
    
    def record_start_position(self) -> None:
        x, y, z = self.get_position()
        print(f"  Start position: ({x:.1f}, {y:.1f}, {z:.1f}) mm")
    
    def move_to_start_position(self) -> bool:
        if self.debug:
            print("  [DEBUG] Would move to home position")
            return True
        ok = self.arm.move_to_home(timeout=self.config.home_timeout_sec)
        print("  ✓ Home position reached" if ok is not False else "  ✗ Failed to reach home position – IK error")
        return ok is not False

    def move_home(self) -> bool:
        return self.arm.move_to_home(timeout=self.config.home_timeout_sec) is not False

    # ------------------------------------------------------------------
    # Trajectory simulation helpers                                 ← NEW
    # ------------------------------------------------------------------

    def start_trajectory_simulation(
        self, start_x: float, start_y: float, start_z: float
    ) -> None:
        """
        Enter simulation mode.

        Subsequent calls to move_relative() will accumulate waypoints
        instead of moving the real arm.

        Parameters
        ----------
        start_x, start_y, start_z:
            Starting end-effector position in mm.  Pass the arm's current
            real position so the preview reflects actual starting conditions.
        """
        self._sim_mode = True
        self._sim_x    = start_x
        self._sim_y    = start_y
        self._sim_z    = start_z
        self._sim_trajectory = [
            {"x_mm": round(start_x, 2), "y_mm": round(start_y, 2), "z_mm": round(start_z, 2), "valid": True}
        ]

    def stop_trajectory_simulation(self) -> List[Dict]:
        """
        Exit simulation mode and return the waypoint list.

        Returns
        -------
        List of ``{"x_mm", "y_mm", "z_mm", "valid"}`` dicts.
        Element 0 is always the start point (``valid=True``).
        """
        self._sim_mode = False
        return list(self._sim_trajectory)


# ═══════════════════════════════════════════════════════════════════════════
# 3 ▸ TrajectoryPreview                                              ← NEW
# ═══════════════════════════════════════════════════════════════════════════

class TrajectoryPreview:
    """
    Renders the planned end-effector trajectory as a matplotlib figure,
    prints a text table, and waits for user confirmation.

    The plot contains
    -----------------
    - Grey dashed rectangle  : workspace boundary
    - Blue filled circles    : valid waypoints
    - Red filled circles     : waypoints outside the workspace
    - Numbered labels        : step index next to each waypoint
    - Direction arrows       : between consecutive waypoints
    - Green star             : start position
    - Monospaced table       : coordinates + validity of every step
    """

    def __init__(
        self,
        config: ExperimentConfig,
        output_path: Optional[Path] = None,
        show_plot: bool = False,
    ) -> None:
        self.config      = config
        self.output_path = output_path or Path("trajectory_preview.png")
        self.show_plot   = show_plot

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, trajectory: List[Dict]) -> Path:
        """
        Draw the trajectory and save the PNG.

        Parameters
        ----------
        trajectory:
            List of ``{"x_mm", "z_mm", "valid"}`` dicts from
            ``RobotController.stop_trajectory_simulation()``.

        Returns
        -------
        Path where the figure was saved.
        """
        import matplotlib
        if not self.show_plot:
            matplotlib.use("Agg")         # headless – no display required
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        cfg = self.config
        fig, ax = plt.subplots(figsize=(9, 7))

        # ── Workspace boundary ────────────────────────────────────────
        ws_x = cfg.workspace_x_min_mm
        ws_z = cfg.workspace_z_min_mm
        ws_y = cfg.workspace_y_min_mm
        ws_w = cfg.workspace_y_max_mm - cfg.workspace_y_min_mm
        ws_h = cfg.workspace_z_max_mm - cfg.workspace_z_min_mm
        ax.add_patch(mpatches.FancyBboxPatch(
            (ws_y, ws_z), ws_w, ws_h,
            boxstyle="square,pad=0",
            linewidth=1.5, edgecolor="#888888",
            facecolor="#f7f7f7", linestyle="--", zorder=0,
        ))
        # ax.text(
        #     ws_y + ws_w / 2, ws_z - 18,
        #     "workspace boundary",
        #     ha="center", va="top", fontsize=8, color="#888888",
        # )

        # ── Path line ─────────────────────────────────────────────────
        xs = [p["x_mm"] for p in trajectory]
        ys = [p["y_mm"] for p in trajectory]
        zs = [p["z_mm"] for p in trajectory]
        ax.plot(ys, zs, color="#aaaaaa", linewidth=1.0, zorder=1)

        # ── Direction arrows (midpoint of each segment) ───────────────
        for i in range(len(trajectory) - 1):
            y0, z0 = trajectory[i]["y_mm"],   trajectory[i]["z_mm"]
            y1, z1 = trajectory[i+1]["y_mm"], trajectory[i+1]["z_mm"]
            my, mz = (y0 + y1) / 2, (z0 + z1) / 2
            dy, dz = y1 - y0, z1 - z0
            if (dy**2 + dz**2) < 1e-6:
                continue
            ax.annotate(
                "",
                xy     =(my + dy * 0.01, mz + dz * 0.01),
                xytext =(my - dy * 0.01, mz - dz * 0.01),
                arrowprops=dict(
                    arrowstyle="-|>", color="#555555",
                    lw=1.2, mutation_scale=12,
                ),
                zorder=2,
            )

        # ── Waypoint markers + labels ─────────────────────────────────
        any_invalid = False
        for i, pt in enumerate(trajectory):
            y, z  = pt["y_mm"], pt["z_mm"]
            color = "#2196F3" if pt["valid"] else "#F44336"
            if not pt["valid"]:
                any_invalid = True
            ax.scatter(
                y, z,
                c=color, s=(250 if i == 0 else 110),
                marker=("*" if i == 0 else "o"),
                zorder=3, edgecolors="white", linewidths=0.8,
            )
            ax.annotate(
                "start" if i == 0 else str(i),
                xy=(y, z), xytext=(6, 6),
                textcoords="offset points",
                fontsize=7.5, color="#333333", zorder=4,
            )

        # ── Warning banner ────────────────────────────────────────────
        if any_invalid:
            ax.text(
                0.5, 0.97,
                "⚠  One or more steps exceed the workspace – confirm is blocked",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="#F44336", alpha=0.85
                ),
            )

        # ── Legend ────────────────────────────────────────────────────
        ax.legend(
            handles=[
                mpatches.Patch(facecolor="#2196F3", label="Valid waypoint"),
                mpatches.Patch(facecolor="#F44336", label="OUT OF WORKSPACE"),
                plt.Line2D(
                    [0], [0], marker="*", color="w",
                    markerfacecolor="#2196F3", markersize=12,
                    label="Start position",
                ),
            ],
            loc="upper right", fontsize=8,
        )

        # ── Axis formatting ───────────────────────────────────────────
        margin = 60
        ax.set_xlim(
            min(ys + [cfg.workspace_y_min_mm]) - margin,
            max(ys + [cfg.workspace_y_max_mm]) + margin,
        )
        ax.set_ylim(
            min(zs + [cfg.workspace_z_min_mm]) - margin,
            max(zs + [cfg.workspace_z_max_mm]) + margin,
        )
        ax.set_xlabel("Y  (mm)   [ +left  /  −right ]", fontsize=9)
        ax.set_ylabel("Z  (mm)   [ +up  /  −down ]",    fontsize=9)
        ax.set_title(
            f"Planned trajectory – {len(trajectory) - 1} step(s)",
            fontsize=11, fontweight="bold",
        )
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.4)

        # ── Coordinate table (right margin) ───────────────────────────
        lines = ["Step   X(mm)   Y(mm)   Z(mm)   OK"]
        lines.append("─" * 36)
        for i, pt in enumerate(trajectory):
            lbl  = "start" if i == 0 else f"{i:>4d}"
            flag = "✓" if pt["valid"] else "✗OOB"
            lines.append(
                f"{lbl:>5}  {pt['x_mm']:>7.1f} {pt['y_mm']:>7.1f} {pt['z_mm']:>7.1f}  {flag}"
            )
        # ax.text(
        #     1.02, 1.0, "\n".join(lines),
        #     transform=ax.transAxes, va="top", ha="left",
        #     fontsize=7, family="monospace",
        #     bbox=dict(
        #         boxstyle="round,pad=0.4", facecolor="#f0f0f0", alpha=0.85
        #     ),
        # )

        plt.tight_layout()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_path, dpi=130, bbox_inches="tight")
        print(f"  ✓ Preview plot saved: {self.output_path.resolve()}")

        if self.show_plot:
            plt.show()

        plt.close(fig)
        return self.output_path

    # ------------------------------------------------------------------
    # Confirmation prompt
    # ------------------------------------------------------------------

    def confirm(self, trajectory: List[Dict]) -> bool:
        """
        Render the plot, print a text table, and prompt for y/n.

        Returns
        -------
        ``True`` if the user types ``y`` or ``yes``.
        ``False`` if the user cancels OR if any step is outside the workspace.
        """
        print("\n" + "═" * 60)
        print("  TRAJECTORY PREVIEW")
        print("═" * 60)

        any_invalid = False
        for i, pt in enumerate(trajectory):
            label = "start " if i == 0 else f"step {i:>3}"
            flag  = "✓" if pt["valid"] else "✗  OUT OF WORKSPACE"
            if not pt["valid"]:
                any_invalid = True
            print(
                f"  {label}  →  X = {pt['x_mm']:>8.1f} mm  "
                f"Y = {pt['y_mm']:>8.1f} mm  "
                f"Z = {pt['z_mm']:>8.1f} mm   {flag}"
            )

        print("─" * 60)
        print(f"  Total: {len(trajectory) - 1} step(s) + start")

        self.render(trajectory)

        if any_invalid:
            print(
                "\n  ✗ Cannot proceed: one or more steps are outside the "
                "configured workspace.\n"
                "    Adjust run_sequence() or workspace limits in "
                "ExperimentConfig and try again.\n"
            )
            return False

        print()
        try:
            answer = input("  Proceed with data collection? [y/N]  ").strip().lower()
        except EOFError:
            answer = "n"

        if answer in ("y", "yes"):
            print("  ✓ Confirmed – starting data collection.\n")
            return True
        print("  Cancelled by user.\n")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 4 ▸ VisionController
# ═══════════════════════════════════════════════════════════════════════════

class VisionController:
    """Wraps SlinkyVisionAPI – only exposes what the orchestrator needs."""

    def __init__(self, config: ExperimentConfig, debug: bool = False) -> None:
        self.config = config
        self.debug  = debug
        self._api   = None

    def start(self) -> None:
        if self.debug:
            print("  [DEBUG] Camera start skipped.")
            return
        from slinky_detection import SlinkyVisionAPI
        self._api = SlinkyVisionAPI(
            width=self.config.camera_width,
            height=self.config.camera_height,
            fps=self.config.camera_fps,
            headless=True,
            stability_window=self.config.stability_window,
            stability_threshold_m=self.config.stability_threshold_m,
            stability_timeout_sec=self.config.stability_timeout_sec,
            stability_poll_sec=self.config.stability_poll_sec,
        )
        self._api.start_camera()
        print("✓ Camera started")

    def stop(self) -> None:
        if self._api is not None:
            self._api.stop_camera()
            self._api = None
            print("✓ Camera stopped")

    def capture_first_frame(self) -> Optional["np.ndarray"]:
        """
        Capture one frame with ArUco detection overlays drawn on top.
        Returns a BGR numpy array, or None in debug mode / on error.
        """
        if self.debug:
            print("  [DEBUG] capture_first_frame skipped.")
            return None
        if self._api is None:
            return None
        try:
            frame = self._api.get_frame()
            corners, ids, _ = self._api.detect_markers(frame)
            annotated = self._api.create_annotated_frame(frame, corners, ids)
            return annotated
        except Exception as exc:
            print(f"  ⚠ Could not capture first frame: {exc}")
            return None

    def get_stable_positions(self) -> Dict:
        if self.debug:
            return self._synthetic_positions()
        return self._api.get_stable_positions()

    @staticmethod
    def _synthetic_positions() -> Dict:
        print("  [DEBUG] Returning synthetic marker positions.")
        obs = {"x": 0.0, "y": -0.05, "z": 0.50,
               "std_x": 0.0, "std_y": 0.0, "std_z": 0.0,
               "n_frames": 8, "stable": True}
        return {
            "marker_0":          {**obs, "x":  0.10},
            "marker_1":          {**obs, "x":  0.00},
            "marker_2":          {**obs, "x": -0.10},
            "frame_shape":       {"width": 640, "height": 480},
            "detected_markers":  [0, 1, 2],
            "stability_reached": True,
            "elapsed_sec":       0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 5 ▸ DataManager
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PoseRecord:
    timestamp:             str
    label:                 str
    ee_pose:               Dict
    markers:               Dict
    stability_reached:     bool
    stability_elapsed_sec: float


class DataManager:
    """Incremental JSON flush, session index, optional Drive upload."""

    def __init__(
        self,
        config: ExperimentConfig,
        output_file: Optional[Path] = None,
        config_dir: Optional[Path] = None,          # ← NEW
    ) -> None:
        self.config = config
        self.output_file: Path = output_file or config.make_output_path()
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Config folder sits at  <output_dir>/configs/<json_stem>/   ← NEW
        self.config_dir: Path = (
            config_dir or config.make_config_dir_path(self.output_file)
        )
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self._records: List[Dict] = []
        self._session_start: str  = datetime.utcnow().isoformat()

        if self.output_file.exists():
            with self.output_file.open("r") as fh:
                self._records = json.load(fh)
            print(f"  Resuming – {len(self._records)} existing record(s) loaded.")

    # ------------------------------------------------------------------
    # Config snapshot                                               ← NEW
    # ------------------------------------------------------------------

    def save_experiment_config(
        self,
        args_dict: Dict,
        trajectory_preview_png: Optional[Path] = None,
        first_frame: Optional["np.ndarray"] = None,
    ) -> None:
        """
        Persist the experiment configuration to self.config_dir:

        configs/<json_stem>/
            experiment_params.json   – all CLI args + ExperimentConfig fields
            trajectory_preview.png   – copy of the preview plot (if available)
            first_frame.png          – first camera frame (if available)

        Parameters
        ----------
        args_dict:
            ``vars(args)`` from argparse – the raw CLI arguments.
        trajectory_preview_png:
            Path to the preview PNG produced by TrajectoryPreview.render().
            If provided, the file is *copied* into the config folder.
        first_frame:
            BGR numpy array from VisionController.capture_first_frame().
            Saved as ``first_frame.png`` via OpenCV or PIL.
        """
        import shutil

        # ── 1. experiment_params.json ─────────────────────────────────
        params: Dict = {
            "json_data_file": str(self.output_file.resolve()),
            "session_start":  self._session_start,
            "cli_args":       {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in args_dict.items()
            },
            "experiment_config": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in asdict(self.config).items()
            },
        }
        params_path = self.config_dir / "experiment_params.json"
        with params_path.open("w") as fh:
            json.dump(params, fh, indent=2)
        print(f"  ✓ Config params saved : {params_path.resolve()}")

        # ── 2. trajectory_preview.png ─────────────────────────────────
        if trajectory_preview_png is not None and trajectory_preview_png.exists():
            dest = self.config_dir / "trajectory_preview.png"
            shutil.copy2(trajectory_preview_png, dest)
            print(f"  ✓ Trajectory preview  : {dest.resolve()}")

        # ── 3. first_frame.png ────────────────────────────────────────
        if first_frame is not None:
            frame_path = self.config_dir / "first_frame.png"
            try:
                import cv2
                cv2.imwrite(str(frame_path), first_frame)
                print(f"  ✓ First frame saved   : {frame_path.resolve()}")
            except ImportError:
                try:
                    from PIL import Image
                    import numpy as np
                    img = Image.fromarray(
                        first_frame[:, :, ::-1]   # BGR → RGB
                        if first_frame.ndim == 3 else first_frame
                    )
                    img.save(str(frame_path))
                    print(f"  ✓ First frame saved   : {frame_path.resolve()}")
                except Exception as exc:
                    print(f"  ⚠ Could not save first frame: {exc}")


    # ------------------------------------------------------------------
    # Dataset status plot                                           ← NEW
    # ------------------------------------------------------------------

    def save_dataset_status(
        self,
        trajectory: List[Dict],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Render a scatter plot showing the status of every planned waypoint
        after the real collection run and save it to the config folder.

        Colour coding
        -------------
        Grey  (#9E9E9E) : Robot could not reach the point (IK failed /
                          workspace violation) – no PoseRecord was written.
        Red   (#F44336) : Point was reached but ArUco stability was NOT
                          achieved (stability_reached=False).
        Green (#4CAF50) : Point was reached AND ArUco was stable
                          (stability_reached=True).

        Parameters
        ----------
        trajectory:
            List of ``{"x_mm", "z_mm", "valid"}`` dicts produced by the
            trajectory simulation (same list used for the preview plot).
            Element 0 is the start position (always valid=True).
        output_path:
            Where to write the PNG.  Defaults to
            ``self.config_dir / "dataset_status.png"``.

        Returns
        -------
        Path where the figure was saved.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # ── Build a lookup: (x_mm, y_mm, z_mm) → PoseRecord ─────────────
        # Round to 1 decimal place to make matching robust against float noise.
        def _key(x, y, z):
            return (round(float(x), 1), round(float(y), 1), round(float(z), 1))

        record_map: Dict = {}
        for rec in self._records:
            ee = rec.get("ee_pose", {})
            k  = _key(ee.get("x_mm", 0), ee.get("y_mm", 0), ee.get("z_mm", 0))
            record_map[k] = rec

        # ── Classify each waypoint ────────────────────────────────────
        STATUS_UNREACHABLE = "unreachable"
        STATUS_UNSTABLE    = "unstable"
        STATUS_OK          = "ok"

        COLOR_MAP = {
            STATUS_UNREACHABLE: "#9E9E9E",
            STATUS_UNSTABLE:    "#F44336",
            STATUS_OK:          "#4CAF50",
        }
        LABEL_MAP = {
            STATUS_UNREACHABLE: "Unreachable (IK failed / out of workspace)",
            STATUS_UNSTABLE:    "ArUco not stable",
            STATUS_OK:          "OK (stable)",
        }

        statuses = []
        for pt in trajectory:
            if not pt.get("valid", True):
                statuses.append(STATUS_UNREACHABLE)
                continue
            k = _key(pt["x_mm"], pt.get("y_mm", 0), pt["z_mm"])
            rec = record_map.get(k)
            if rec is None:
                # Planned point that was never recorded → IK failed at runtime
                statuses.append(STATUS_UNREACHABLE)
            elif not rec.get("stability_reached", False):
                statuses.append(STATUS_UNSTABLE)
            else:
                statuses.append(STATUS_OK)

        # ── Plot ──────────────────────────────────────────────────────
        cfg = self.config
        fig, ax = plt.subplots(figsize=(9, 7))

        # Workspace boundary
        ws_x = cfg.workspace_x_min_mm
        ws_z = cfg.workspace_z_min_mm
        ws_w = cfg.workspace_x_max_mm - cfg.workspace_x_min_mm
        ws_h = cfg.workspace_z_max_mm - cfg.workspace_z_min_mm
        ax.add_patch(mpatches.FancyBboxPatch(
            (ws_x, ws_z), ws_w, ws_h,
            boxstyle="square,pad=0",
            linewidth=1.5, edgecolor="#888888",
            facecolor="#f7f7f7", linestyle="--", zorder=0,
        ))
        # ax.text(
        #     ws_x + ws_w / 2, ws_z - 18,
        #     "workspace boundary",
        #     ha="center", va="top", fontsize=8, color="#888888",
        # )

        # Path line (light grey)
        xs = [p["x_mm"] for p in trajectory]
        zs = [p["z_mm"] for p in trajectory]
        ax.plot(xs, zs, color="#cccccc", linewidth=0.8, zorder=1)

        # Scatter each waypoint with its status colour
        for i, (pt, status) in enumerate(zip(trajectory, statuses)):
            x, z  = pt["x_mm"], pt["z_mm"]
            color = COLOR_MAP[status]
            marker = "*" if i == 0 else "o"
            size   = 300 if i == 0 else 130
            ax.scatter(
                x, z,
                c=color, s=size, marker=marker,
                zorder=3, edgecolors="white", linewidths=0.8,
            )
            ax.annotate(
                "start" if i == 0 else str(i),
                xy=(x, z), xytext=(6, 6),
                textcoords="offset points",
                fontsize=7.5, color="#333333", zorder=4,
            )

        # Legend
        legend_handles = [
            mpatches.Patch(color=COLOR_MAP[s], label=LABEL_MAP[s])
            for s in (STATUS_OK, STATUS_UNSTABLE, STATUS_UNREACHABLE)
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.85,
        )

        # Summary counts in title
        n_ok          = statuses.count(STATUS_OK)
        n_unstable    = statuses.count(STATUS_UNSTABLE)
        n_unreachable = statuses.count(STATUS_UNREACHABLE)
        total         = len(statuses)
        ax.set_title(
            f"Dataset Status  –  {self.output_file.stem}\n"
            f"Total: {total}  |  ✓ OK: {n_ok}  |  "
            f"⚠ Unstable: {n_unstable}  |  ✗ Unreachable: {n_unreachable}",
            fontsize=10,
        )
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z (mm)")
        ax.set_aspect("equal")
        margin = 80
        ax.set_xlim(ws_x - margin, ws_x + ws_w + margin)
        ax.set_ylim(ws_z - margin, ws_z + ws_h + margin)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

        # Side-panel text table
        lines = ["  #   X(mm)   Y(mm)   Z(mm)  Status"]
        lines.append("  " + "─" * 38)
        for i, (pt, status) in enumerate(zip(trajectory, statuses)):
            tag = {"ok": "OK", "unstable": "UNSTABLE", "unreachable": "UNREACHABLE"}[status]
            lines.append(
                f"  {'S' if i == 0 else str(i):>2}  "
                f"{pt['x_mm']:>7.1f}  {pt.get('y_mm', 0.0):>7.1f}  {pt['z_mm']:>7.1f}  {tag}"
            )
        # ax.text(
        #     1.02, 1.0, "\n".join(lines),
        #     transform=ax.transAxes, va="top", ha="left",
        #     fontsize=7, family="monospace",
        #     bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", alpha=0.85),
        # )

        plt.tight_layout()

        dest = output_path or (self.config_dir / "dataset_status.png")
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=130, bbox_inches="tight")
        print(f"  ✓ Dataset status plot saved: {dest.resolve()}")
        plt.close(fig)
        return dest

    def add(self, record: PoseRecord) -> None:
        self._records.append(asdict(record))
        self._flush()

    def count(self) -> int:
        return len(self._records)

    def _flush(self) -> None:
        with self.output_file.open("w") as fh:
            json.dump(self._records, fh, indent=2)

    def write_session_index(self) -> None:
        index_path = self.output_file.parent / "session_index.json"
        index: List[Dict] = []
        if index_path.exists():
            with index_path.open("r") as fh:
                index = json.load(fh)
        index.append({
            "file":          self.output_file.name,
            "session_start": self._session_start,
            "session_end":   datetime.utcnow().isoformat(),
            "n_records":     len(self._records),
        })
        with index_path.open("w") as fh:
            json.dump(index, fh, indent=2)
        print(f"  Session index updated: {index_path}")

    def check_drive_connection(self) -> bool:
        """
        Verify that Google Drive auth works and the target folder is reachable.
        Returns True if everything is OK, else False.
        """
        try:
            creds_file = Path(self.config.drive_credentials_file)
            token_file = Path(self.config.drive_token_file)

            if not creds_file.exists():
                print(f"  ✗ Drive credentials not found at '{creds_file}'.")
                return False

            creds = None

            # Reuse token if present
            if token_file.exists():
                from google.oauth2.credentials import Credentials
                creds = Credentials.from_authorized_user_file(
                    str(token_file),
                    scopes=["https://www.googleapis.com/auth/drive"]
                )

            # Refresh or create token
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    from google.auth.transport.requests import Request
                    creds.refresh(Request())
                else:
                    from google_auth_oauthlib.flow import InstalledAppFlow
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(creds_file),
                        scopes=["https://www.googleapis.com/auth/drive"]
                    )
                    creds = flow.run_local_server(port=0)

                token_file.write_text(creds.to_json())

            # Build Drive service
            from googleapiclient.discovery import build
            service = build("drive", "v3", credentials=creds)

            # Check target folder exists and is accessible
            folder = service.files().get(
                fileId=self.config.drive_target_folder_id,
                fields="id, name, mimeType"
            ).execute()

            if folder.get("mimeType") != "application/vnd.google-apps.folder":
                print("  ✗ Target ID exists, but it is not a Google Drive folder.")
                return False

            print(f"  ✓ Drive connected.")
            print(f"  ✓ Target folder: {folder.get('name')} ({folder.get('id')})")
            return True

        except Exception as e:
            print(f"  ✗ Drive connection failed: {e}")
            return False

    def upload_to_drive(self) -> bool:
        creds_path = Path(self.config.drive_credentials_file)
        if not creds_path.exists():
            print(f"  ✗ Drive credentials not found at '{creds_path}'. Skipping.")
            return False
        try:
            svc  = self._get_drive_service()
            root = self._get_or_create_folder(svc, self.config.drive_library_root)
            exp  = self._get_or_create_folder(svc, self.config.experiment_name, root)

            # ── upload dataset ─────────────────────────────
            ds  = self._get_or_create_folder(svc, "dataset", exp)
            fid = self._upload_file(svc, self.output_file, ds)
            print(f"  ✓ Uploaded '{self.output_file.name}' → Drive (id={fid})")

            session_index = self.output_file.parent / "session_index.json"
            if session_index.exists():
                sid = self._upload_file(svc, session_index, ds)
                print(f"  ✓ Uploaded 'session_index.json' → Drive (id={sid})")

            # ── upload configs ─────────────────────────────────────
            if self.config_dir.exists():
                configs_root = self._get_or_create_folder(svc, "configs", exp)
                cfg_folder   = self._get_or_create_folder(svc, self.config_dir.name, configs_root)
                for f in sorted(self.config_dir.iterdir()):
                    if f.is_file():
                        cid = self._upload_file(svc, f, cfg_folder)
                        print(f"  ✓ Uploaded '{f.name}' → Drive/configs/{self.config_dir.name}/ (id={cid})")

            return True
        except Exception as exc:
            print(f"  ✗ Drive upload failed: {exc}")
            return False

    def _get_drive_service(self):
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        SCOPES = ["https://www.googleapis.com/auth/drive"]
        token  = Path(self.config.drive_token_file)
        creds  = None
        if token.exists():
            creds = Credentials.from_authorized_user_file(str(token), SCOPES)
        if not creds or not creds.valid:
            flow  = InstalledAppFlow.from_client_secrets_file(
                self.config.drive_credentials_file, SCOPES
            )
            creds = flow.run_local_server(port=0)
            token.write_text(creds.to_json())
        return build("drive", "v3", credentials=creds)

    @staticmethod
    def _get_or_create_folder(svc, name: str, parent: Optional[str] = None) -> str:
        q = (f"name='{name}' and mimeType='application/vnd.google-apps.folder' "
             f"and trashed=false")
        if parent:
            q += f" and '{parent}' in parents"
        files = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
        if files:
            return files[0]["id"]
        meta = {"name": name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent] if parent else []}
        return svc.files().create(body=meta, fields="id").execute()["id"]

    @staticmethod
    def _upload_file(svc, path: Path, parent: str) -> str:
        from googleapiclient.http import MediaFileUpload
        media = MediaFileUpload(str(path), resumable=True)
        meta  = {"name": path.name, "parents": [parent]}
        return svc.files().create(
            body=meta, media_body=media, fields="id"
        ).execute()["id"]


# ═══════════════════════════════════════════════════════════════════════════
# 6 ▸ DataCollector  (orchestrator)
# ═══════════════════════════════════════════════════════════════════════════

class DataCollector:
    """
    Top-level orchestrator.

    New parameters vs. previous version
    ------------------------------------
    preview : bool (default True)
        Simulate the sequence, render the trajectory plot, and wait for
        y/n confirmation before touching the robot.
    preview_output : Path | None
        Where to save the preview PNG.
    preview_show : bool (default False)
        Open the matplotlib GUI window (requires a display).
    """

    def __init__(
        self,
        config: ExperimentConfig,
        debug: bool = False,
        upload: bool = False,
        preview: bool = True,                           # ← NEW
        preview_output: Optional[Path] = None,          # ← NEW
        preview_show: bool = False,                     # ← NEW
        output_file: Optional[Path] = None,
        args_dict: Optional[Dict] = None,               # ← NEW
    ) -> None:
        self.config   = config
        self.debug    = debug
        self.upload   = upload
        self.preview  = preview                         # ← NEW
        self._args_dict: Dict = args_dict or {}         # ← NEW  (for config snapshot)

        self.robot   = RobotController(config, debug=debug)
        self.vision  = VisionController(config, debug=debug)
        self.data    = DataManager(config, output_file=output_file)
        self._last_trajectory: List[Dict] = []        # ← NEW: cached for status plot

        # Preview renderer                              ← NEW
        _preview_png = (
            preview_output
            or config.output_dir / f"{config.experiment_name}_trajectory_preview.png"
        )
        self._previewer = TrajectoryPreview(
            config, output_path=_preview_png, show_plot=preview_show
        )

    # ------------------------------------------------------------------
    # Core move primitive
    # ------------------------------------------------------------------

    def move(
        self,
        direction: str,
        steps: int = 1,
        mm: Optional[float] = None,
    ) -> None:
        """
        Move in a cardinal direction, recording a pose at each step.

        This method is called TWICE per run: once during trajectory
        simulation (robot._sim_mode = True) and once for real collection.
        The sim branch is handled inside RobotController.move_relative(),
        so this method needs no changes to support preview.
        """
        if direction not in _DIRECTION_MAP:
            raise ValueError(
                f"Unknown direction '{direction}'. Valid: {list(_DIRECTION_MAP)}"
            )
        mm = mm if mm is not None else self.config.default_step_mm
        ux, uy, uz = _DIRECTION_MAP[direction]
        dx, dy, dz = ux * mm, uy * mm, uz * mm

        # ── Simulation mode: silent waypoint accumulation ────────────── ← NEW
        if self.robot._sim_mode:
            for _ in range(steps):
                self.robot.move_relative(dx_mm=dx, dy_mm=dy, dz_mm=dz)
            return

        # ── Real collection mode ─────────────────────────────────────────
        print(f"\n{'─' * 60}")
        print(f"  {direction.upper()}  ×{steps}  ({mm:.0f} mm/step)")

        for step in range(1, steps + 1):
            label = f"{direction} step {step}/{steps}"
            print(f"\n  → [{label}]  dx={dx:+.0f} mm  dy={dy:+.0f} mm  dz={dz:+.0f} mm")

            ok = self.robot.move_relative(dx_mm=dx, dy_mm=dy, dz_mm=dz)
            if not ok:
                print("    Skipping record for this step.")
                continue

            if not self.debug:
                time.sleep(0.3)

            vision_result             = self.vision.get_stable_positions()
            actual_x, actual_y, actual_z = self.robot.get_position()

            _skip = {"frame_shape", "detected_markers", "stability_reached",
                     "elapsed_sec", "saved_path", "save_error",
                     "saved_image_path", "save_image_error",
                     "annotated_frame_b64", "annotated_frame_error"}
            markers = {k: v for k, v in vision_result.items() if k not in _skip}

            record = PoseRecord(
                timestamp=datetime.utcnow().isoformat(),
                label=label,
                ee_pose={
                    "x_mm": round(actual_x, 3),
                    "y_mm": round(actual_y, 3),
                    "z_mm": round(actual_z, 3),
                },
                markers=markers,
                stability_reached=vision_result.get("stability_reached", False),
                stability_elapsed_sec=vision_result.get("elapsed_sec", 0.0),
            )
            self.data.add(record)

            detected = list(markers.keys())
            print(
                f"  ✓ Recorded  ee=({actual_x:.1f}, {actual_y:.1f}, {actual_z:.1f}) mm  "
                f"detected={detected or 'none'}  "
                f"stable={record.stability_reached}  "
                f"t={record.stability_elapsed_sec:.2f}s"
            )

    # ------------------------------------------------------------------
    # Movement sequence  ← EDIT THIS
    # ------------------------------------------------------------------

    def run_sequence(self) -> None:
        """
        Define the traversal sequence here.

        This method is called TWICE: once for preview simulation and once
        for real collection.  Keep it pure – only self.move() calls here.

        Example snake grid::

            for col in range(4):
                self.move("up",   steps=7, mm=50)
                self.move("left", steps=1, mm=50)
                self.move("down", steps=7, mm=50)
                if col < 3:
                    self.move("left", steps=1, mm=50)
        """
        self.move("z+", steps=1, mm=150)
        for _ in range(25):
            self.move("y+", steps=40, mm=5)
            self.move("z-", steps=1, mm=5)
            self.move("y-", steps=40, mm=5)
            self.move('z-', steps=1, mm=5)

    # ------------------------------------------------------------------
    # Trajectory preview helpers                                    ← NEW
    # ------------------------------------------------------------------

    def _build_trajectory(self) -> List[Dict]:
        """
        Dry-run run_sequence() in simulation mode.

        Uses the arm's current real position as the start point (or home
        position in debug mode) so the preview is accurate.
        """
        start_x, start_y, start_z = self.robot.get_position()
        self.robot.start_trajectory_simulation(start_x, start_y, start_z)
        self.run_sequence()                            # ← same sequence, no side effects
        traj = self.robot.stop_trajectory_simulation()
        self._last_trajectory = traj                  # ← cache for dataset status plot
        return traj

    def _preview_and_confirm(self) -> bool:
        """
        Simulate the trajectory, render the plot, and ask for confirmation.

        Returns True only if the user types 'y' and all steps are valid.
        """
        print("\nSimulating trajectory for preview …")
        trajectory = self._build_trajectory()
        return self._previewer.confirm(trajectory)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _record_start_pose(self) -> None:
        print("\nRecording starting position …")
        vision_result                = self.vision.get_stable_positions()
        actual_x, actual_y, actual_z = self.robot.get_position()
        _skip = {"frame_shape", "detected_markers", "stability_reached",
                 "elapsed_sec", "saved_path", "save_error",
                 "saved_image_path", "save_image_error",
                 "annotated_frame_b64", "annotated_frame_error"}
        markers = {k: v for k, v in vision_result.items() if k not in _skip}
        self.data.add(PoseRecord(
            timestamp=datetime.utcnow().isoformat(),
            label="start",
            ee_pose={
                "x_mm": round(actual_x, 3),
                "y_mm": round(actual_y, 3),
                "z_mm": round(actual_z, 3),
            },
            markers=markers,
            stability_reached=vision_result.get("stability_reached", False),
            stability_elapsed_sec=vision_result.get("elapsed_sec", 0.0),
        ))
        print(
            f"  ✓ Start pose recorded  "
            f"ee=({actual_x:.1f}, {actual_y:.1f}, {actual_z:.1f}) mm  "
            f"stable={vision_result.get('stability_reached')}"
        )

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Full experiment lifecycle:

        1. Init ROS node + enable robot.
        2. NEW: Simulate trajectory → plot preview → wait for y/n.
           Exit immediately if the user cancels or any step is out of bounds.
        3. Start camera.
        4. Record start pose.
        5. Execute run_sequence() for real.
        6. Teardown + optional Drive upload.
        """
        import rospy
        rospy.init_node("slinky_data_collection", anonymous=True)

        print("=" * 60)
        print("Slinky ArUco Data Collection")
        print(f"  Experiment   : {self.config.experiment_name}")
        print(f"  Debug mode   : {self.debug}")
        print(f"  Upload       : {self.upload}")
        print(f"  Preview      : {self.preview}")
        print(f"  Output       : {self.data.output_file.resolve()}")
        print(f"  Workspace X  : [{self.config.workspace_x_min_mm}, "
              f"{self.config.workspace_x_max_mm}] mm")
        print(f"  Workspace Z  : [{self.config.workspace_z_min_mm}, "
              f"{self.config.workspace_z_max_mm}] mm")
        print(f"  Stability    : window={self.config.stability_window}  "
              f"threshold={self.config.stability_threshold_m * 1000:.1f} mm  "
              f"timeout={self.config.stability_timeout_sec}s")
        print("=" * 60)

        # Drive preflight first
        if self.upload:
            print("Checking Google Drive connection …")
            if not self.data.check_drive_connection():
                print("Drive check failed. Exiting before data collection.")
                return

        self.robot.enable()

        # self.robot.move_to_home()  # best effort to start from a consistent pose
        # print("Moved to home position.")
        # time.sleep(10.0)

        # ── NEW: preview ───────────────────────────────────────────────
        if self.preview:
            if not self._preview_and_confirm():
                print("Exiting without data collection.")
                return
        # ──────────────────────────────────────────────────────────────

        self.vision.start()

        # ── NEW: save config snapshot ──────────────────────────────────
        print("\nSaving experiment config snapshot …")
        first_frame = self.vision.capture_first_frame()
        preview_png = self._previewer.output_path if self.preview else None
        self.data.save_experiment_config(
            args_dict=self._args_dict,
            trajectory_preview_png=preview_png,
            first_frame=first_frame,
        )
        # ──────────────────────────────────────────────────────────────

        try:
            self._record_start_pose()
            self.run_sequence()
        except KeyboardInterrupt:
            print("\n  ⚠ Interrupted – partial data saved.")
            self.robot.move_to_home()  # best effort to get arm out of the way for inspection
        finally:
            self.vision.stop()
            self.data.write_session_index()

            # ── NEW: dataset status plot ───────────────────────────────
            if self._last_trajectory:
                print("\nSaving dataset status plot …")
                self.data.save_dataset_status(self._last_trajectory)
            else:
                # No preview was run (--no-preview): build trajectory now
                # purely for the status plot (no side effects on the arm).
                print("\nBuilding trajectory for dataset status plot …")
                traj = self._build_trajectory()
                self.data.save_dataset_status(traj)
            # ──────────────────────────────────────────────────────────

            print(f"\n{'=' * 60}")
            print(f"Done.  {self.data.count()} pose(s) recorded.")
            print(f"Data: {self.data.output_file.resolve()}")
            if self.upload:
                print("Uploading to Google Drive …")
                self.data.upload_to_drive()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Slinky ArUco data collection (Sawyer + RealSense).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--debug",            action="store_true")
    p.add_argument("--upload",           action="store_true")
    p.add_argument("--output",           type=Path, default=None, metavar="FILE")
    p.add_argument("--experiment-name",  type=str,  default="DLO_slinky_test", metavar="NAME")

    # Preview                                                       ← NEW
    g = p.add_argument_group("Trajectory preview")
    g.add_argument("--no-preview",      action="store_true",
                   help="Skip trajectory preview and go straight to collection.")
    g.add_argument("--preview-output",  type=Path, default=None, metavar="FILE",
                   help="Path for the preview PNG.")
    g.add_argument("--preview-show",    action="store_true",
                   help="Open the matplotlib GUI window (requires a display).")

    # Motion
    p.add_argument("--step-mm",            type=float, default=50.0,  metavar="MM")
    p.add_argument("--move-timeout",       type=float, default=15.0,  metavar="SEC")
    p.add_argument("--joint-speed",        type=float, default=0.01,   metavar="FRAC")

    # Stability
    p.add_argument("--stability-window",    type=int,   default=10,     metavar="N")
    p.add_argument("--stability-threshold", type=float, default=0.005, metavar="M")
    p.add_argument("--stability-timeout",   type=float, default=120.0,   metavar="SEC")

    # # Home
    # p.add_argument("--home-x", type=float, default=0.0,   metavar="MM")
    # p.add_argument("--home-z", type=float, default=400.0, metavar="MM")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        default_step_mm=args.step_mm,
        move_timeout_sec=args.move_timeout,
        joint_speed=args.joint_speed,
        stability_window=args.stability_window,
        stability_threshold_m=args.stability_threshold,
        stability_timeout_sec=args.stability_timeout,
        # home_x_mm=args.home_x,
        # home_z_mm=args.home_z,
    )
    collector = DataCollector(
        config=config,
        debug=args.debug,
        upload=args.upload,
        preview=not args.no_preview,                   # ← NEW
        preview_output=args.preview_output,            # ← NEW
        preview_show=args.preview_show,                # ← NEW
        output_file=args.output,
        args_dict=vars(args),                          # ← NEW
    )
    collector.run()


if __name__ == "__main__":
    main()