from __future__ import annotations

import argparse
import math
import os
import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project path bootstrap
# ---------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASE/ "../robot_API"))  # for sawyer_controller.py
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

    # ── Grid / domain traversal ────────────────────────────────────────
    grid_step_mm: float = 50.0  # Spacing between grid sample points in the domain (mm)

    # ── Stability gate ─────────────────────────────────────────────────
    stability_window: int = 8
    stability_threshold_m: float = 0.007
    stability_timeout_sec: float = 5.0
    stability_poll_sec: float = 0.05

    # ── Camera ────────────────────────────────────────────────────────
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    # live_preview: save latest_frame.png at a throttled rate (for Streamlit).
    # live_preview_interval_sec: minimum seconds between live preview writes.
    # save_frames: save one annotated JPEG per waypoint for timelapse/gif.
    # These are independent — both, neither, or either can be enabled.
    live_preview: bool = True
    live_preview_interval_sec: float = 0.5
    save_frames: bool = False

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
# 2 ▸ Domain classes
# ═══════════════════════════════════════════════════════════════════════════

class Domain(ABC):
    """
    Abstract 2D boundary domain in the Y-Z plane of the robot workspace.

    Subclasses define the shape of the region where data should be collected.
    The grid generator samples this region and produces a snake-pattern path.
    """

    @abstractmethod
    def contains(self, y_mm: float, z_mm: float) -> bool:
        """Return True if the point (y_mm, z_mm) lies inside the domain."""
        ...

    @abstractmethod
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (y_min, y_max, z_min, z_max) bounding box of the domain."""
        ...


@dataclass
class RectangularDomain(Domain):
    """
    Axis-aligned rectangular domain in the Y-Z plane.

    Suitable for objects like a slinky that occupy a rectangular region.

    Parameters
    ----------
    y_min_mm, y_max_mm : lateral extents
    z_min_mm, z_max_mm : vertical extents
    """
    y_min_mm: float
    y_max_mm: float
    z_min_mm: float
    z_max_mm: float

    def contains(self, y_mm: float, z_mm: float) -> bool:
        return (
            self.y_min_mm <= y_mm <= self.y_max_mm
            and self.z_min_mm <= z_mm <= self.z_max_mm
        )

    def bounds(self) -> Tuple[float, float, float, float]:
        return self.y_min_mm, self.y_max_mm, self.z_min_mm, self.z_max_mm


@dataclass
class SemicircleDomain(Domain):
    """
    Half-disk domain in the Y-Z plane.

    Suitable for objects like a taut elastic strip whose endpoints are fixed
    and whose maximum reach traces a semicircular arc.

    Parameters
    ----------
    center_y_mm, center_z_mm : centre of the full circle
    radius_mm                : radius (= max reach of the elastic strip)
    orientation              : which half to keep –
                               "up"    → z >= center_z  (arc opens upward)
                               "down"  → z <= center_z  (arc opens downward)
                               "right" → y >= center_y
                               "left"  → y <= center_y
    """
    center_y_mm: float
    center_z_mm: float
    radius_mm: float
    orientation: str = "up"   # "up" | "down" | "left" | "right"

    def contains(self, y_mm: float, z_mm: float) -> bool:
        dy = y_mm - self.center_y_mm
        dz = z_mm - self.center_z_mm
        if dy ** 2 + dz ** 2 > self.radius_mm ** 2:
            return False
        if self.orientation == "up":    return dz >= 0
        if self.orientation == "down":  return dz <= 0
        if self.orientation == "right": return dy >= 0
        if self.orientation == "left":  return dy <= 0
        return True

    def bounds(self) -> Tuple[float, float, float, float]:
        r  = self.radius_mm
        cy = self.center_y_mm
        cz = self.center_z_mm
        if self.orientation == "up":
            return cy - r, cy + r, cz,     cz + r
        if self.orientation == "down":
            return cy - r, cy + r, cz - r, cz
        if self.orientation == "right":
            return cy,     cy + r, cz - r, cz + r
        if self.orientation == "left":
            return cy - r, cy,     cz - r, cz + r
        return cy - r, cy + r, cz - r, cz + r


def generate_snake_path(
    domain: Domain,
    step_mm: float,
) -> List[Tuple[float, float]]:
    """
    Generate a boustrophedon (snake) grid path within the domain.

    The Y-Z plane is sampled on a regular grid with spacing ``step_mm``.
    Only points that fall inside ``domain`` are kept.  Columns (constant Z)
    are traversed in alternating Y directions to minimise travel distance.

    Parameters
    ----------
    domain   : any Domain subclass
    step_mm  : grid spacing in mm

    Returns
    -------
    List of ``(y_mm, z_mm)`` absolute waypoints in traversal order.
    """
    y_min, y_max, z_min, z_max = domain.bounds()

    def _frange(start: float, stop: float, step: float) -> List[float]:
        result: List[float] = []
        v = start
        while v <= stop + step * 0.5:
            result.append(round(v, 4))
            v += step
        return result

    ys = _frange(y_min, y_max, step_mm)
    zs = _frange(z_min, z_max, step_mm)

    waypoints: List[Tuple[float, float]] = []
    for col_idx, z in enumerate(zs):
        col_ys = [y for y in ys if domain.contains(y, z)]
        if col_idx % 2 == 1:
            col_ys = col_ys[::-1]   # reverse every other column → snake
        for y in col_ys:
            waypoints.append((y, z))

    return waypoints


# ═══════════════════════════════════════════════════════════════════════════
# 3 ▸ RobotController
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
    
    def get_home_position(self) -> Tuple[float, float, float]:
        """Return the home position as (x_mm, y_mm, z_mm)."""
        if self.arm is None:
            # debug / pre-enable fallback — matches get_position() debug return
            return 0.0, 0.0, 400.0
        return (
            self.arm._home_pos_x * 1000.0,
            self.arm._home_pos_y * 1000.0,
            self.arm._home_pos_z * 1000.0,
        )

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

        ok = self.arm.move_cartesian_relative(
            dx_mm=dx_mm, dy_mm=dy_mm, dz_mm=dz_mm,
            timeout=self.config.move_timeout_sec,
            joint_speed=self.config.joint_speed,
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
        return self.arm.move_home() is not False

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
# 4 ▸ TrajectoryPreview                                              ← NEW
# ═══════════════════════════════════════════════════════════════════════════

class TrajectoryPreview:
    """
    Renders the planned end-effector trajectory as a matplotlib figure,
    prints a text table, and waits for user confirmation.

    The plot contains
    -----------------
    - Grey dashed rectangle  : workspace boundary
    - Coloured outline       : domain boundary (if provided)
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
        domain: Optional[Domain] = None,
    ) -> None:
        self.config      = config
        self.output_path = output_path or Path("trajectory_preview.png")
        self.show_plot   = show_plot
        self.domain      = domain

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

        # ── Domain boundary overlay ───────────────────────────────────
        if self.domain is not None:
            self._draw_domain_boundary(ax, self.domain)

        # ── Density-adaptive display settings ────────────────────────
        n        = len(trajectory)
        dense    = n > 50
        import numpy as np
        cmap     = plt.get_cmap("Blues")
        colors   = [cmap(0.3 + 0.7 * i / max(n - 1, 1)) for i in range(n)]

        xs = [p["x_mm"] for p in trajectory]
        ys = [p["y_mm"] for p in trajectory]
        zs = [p["z_mm"] for p in trajectory]

        # ── Path line ─────────────────────────────────────────────────
        # Dense: colour-encoded segments to show traversal order.
        # Sparse: plain grey line.
        if dense:
            for i in range(n - 1):
                ax.plot(
                    [trajectory[i]["y_mm"], trajectory[i+1]["y_mm"]],
                    [trajectory[i]["z_mm"], trajectory[i+1]["z_mm"]],
                    color=colors[i], linewidth=1.2, zorder=1,
                )
        else:
            ax.plot(ys, zs, color="#aaaaaa", linewidth=1.0, zorder=1)

        # ── Direction arrows ──────────────────────────────────────────
        # Dense: ~25 evenly-spaced arrows.  Sparse: every segment.
        n_arrows   = 25 if dense else (n - 1)
        arrow_step = max(1, (n - 1) // n_arrows)
        for i in range(0, n - 1, arrow_step):
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
                    arrowstyle="-|>",
                    color=colors[i] if dense else "#555555",
                    lw=1.2, mutation_scale=12,
                ),
                zorder=2,
            )

        # ── Waypoint markers + labels ─────────────────────────────────
        # Dense: small semi-transparent dots, colour = traversal order,
        #        red for out-of-workspace; label only start and end.
        # Sparse: original behaviour — labelled circles for every point.
        any_invalid = False
        for i, pt in enumerate(trajectory):
            y, z = pt["y_mm"], pt["z_mm"]
            if not pt["valid"]:
                any_invalid = True

            is_start = i == 0
            is_end   = i == n - 1

            if dense:
                color  = "#F44336" if not pt["valid"] else colors[i]
                marker = "*" if is_start else "o"
                size   = 250 if is_start else (120 if is_end else 18)
                alpha  = 1.0 if (is_start or is_end) else 0.55
                lw     = 0.8 if (is_start or is_end) else 0.0
                ax.scatter(y, z, c=[color], s=size, marker=marker,
                           zorder=3, edgecolors="white", linewidths=lw,
                           alpha=alpha)
                if is_start:
                    ax.annotate("start", xy=(y, z), xytext=(6, 6),
                                textcoords="offset points",
                                fontsize=7.5, color="#333333", zorder=4)
                elif is_end:
                    ax.annotate("end", xy=(y, z), xytext=(6, 6),
                                textcoords="offset points",
                                fontsize=7.5, color="#333333", zorder=4)
            else:
                color = "#2196F3" if pt["valid"] else "#F44336"
                ax.scatter(y, z, c=color, s=(250 if is_start else 110),
                           marker=("*" if is_start else "o"),
                           zorder=3, edgecolors="white", linewidths=0.8)
                ax.annotate(
                    "start" if is_start else str(i),
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
        if dense:
            import matplotlib.cm as cm
            sm = cm.ScalarMappable(
                cmap="Blues",
                norm=plt.Normalize(vmin=0, vmax=n - 1),
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Traversal order (first → last)", fontsize=8)
            cbar.set_ticks([0, n - 1])
            cbar.set_ticklabels(["start", "end"])
            legend_handles = [
                plt.Line2D([0], [0], marker="*", color="w",
                           markerfacecolor=cmap(0.3), markersize=12,
                           label="Start position"),
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=cmap(1.0), markersize=8,
                           label="End position"),
                mpatches.Patch(facecolor="#F44336", label="OUT OF WORKSPACE"),
            ]
        else:
            legend_handles = [
                mpatches.Patch(facecolor="#2196F3", label="Valid waypoint"),
                mpatches.Patch(facecolor="#F44336", label="OUT OF WORKSPACE"),
                plt.Line2D([0], [0], marker="*", color="w",
                           markerfacecolor="#2196F3", markersize=12,
                           label="Start position"),
            ]
        if self.domain is not None:
            legend_handles.append(
                mpatches.Patch(facecolor="none", edgecolor="#FF9800",
                               linewidth=2,
                               label=f"Domain boundary ({type(self.domain).__name__})")
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

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

        plt.tight_layout()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.output_path, dpi=130, bbox_inches="tight")
        print(f"  ✓ Preview plot saved: {self.output_path.resolve()}")

        if self.show_plot:
            plt.show()

        plt.close(fig)
        return self.output_path

    # ------------------------------------------------------------------
    # Domain boundary drawing
    # ------------------------------------------------------------------

    def _draw_domain_boundary(self, ax, domain: Domain) -> None:
        """Overlay the domain boundary on ax."""
        import matplotlib.patches as mpatches

        color = "#FF9800"
        lw    = 2.0

        if isinstance(domain, RectangularDomain):
            w = domain.y_max_mm - domain.y_min_mm
            h = domain.z_max_mm - domain.z_min_mm
            ax.add_patch(mpatches.FancyBboxPatch(
                (domain.y_min_mm, domain.z_min_mm), w, h,
                boxstyle="square,pad=0",
                linewidth=lw, edgecolor=color,
                facecolor="#FF980015", linestyle="-", zorder=1,
            ))

        elif isinstance(domain, SemicircleDomain):
            import numpy as _np
            theta_map = {
                "up":    (0,   180),
                "down":  (180, 360),
                "right": (-90,  90),
                "left":  (90,  270),
            }
            t0, t1 = theta_map.get(domain.orientation, (0, 180))
            thetas = _np.linspace(math.radians(t0), math.radians(t1), 200)
            ys = domain.center_y_mm + domain.radius_mm * _np.cos(thetas)
            zs = domain.center_z_mm + domain.radius_mm * _np.sin(thetas)
            # close the shape with a diameter line
            ax.plot(ys, zs, color=color, linewidth=lw, zorder=1)
            ax.plot(
                [ys[0], ys[-1]], [zs[0], zs[-1]],
                color=color, linewidth=lw, zorder=1,
            )

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
                "    Adjust the domain bounds or workspace limits in "
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
# 5 ▸ VisionController
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
            live_preview=self.config.live_preview,
            live_preview_interval_sec=self.config.live_preview_interval_sec,
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

    def get_stable_positions(self, save_image: Optional[str] = None) -> Dict:
        if self.debug:
            return self._synthetic_positions()
        return self._api.get_stable_positions(save_image=save_image)

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
# 6 ▸ DataManager
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

        # Frames folder for per-waypoint timelapse images
        self.frames_dir: Path = self.config_dir / "frames"
        if config.save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

        self._records: List[Dict] = []
        self._session_start: str  = datetime.utcnow().isoformat()

        if self.output_file.exists():
            with self.output_file.open("r") as fh:
                self._records = [json.loads(line) for line in fh if line.strip()]
            print(f"  Resuming – {len(self._records)} existing record(s) loaded.")
        else:
            self.output_file.touch()  # create immediately so data is never lost on early crash
            print(f"  Output file created: {self.output_file.resolve()}")

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

        plt.tight_layout()

        dest = output_path or (self.config_dir / "dataset_status.png")
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=130, bbox_inches="tight")
        print(f"  ✓ Dataset status plot saved: {dest.resolve()}")
        plt.close(fig)
        return dest

    def completed_grid_waypoints(self) -> List[Tuple[float, float]]:
        """
        Return the (y_mm, z_mm) coordinates of grid waypoints already recorded.

        Coordinates are read from ``ee_pose`` in each record rather than
        parsed from the label string, so resume matching is robust to changes
        in grid spacing, domain bounds, or label format.  Used by
        ``_run_domain_sequence`` together with a tolerance check.
        """
        done: List[Tuple[float, float]] = []
        for rec in self._records:
            if not rec.get("label", "").startswith("grid_"):
                continue
            pose = rec.get("ee_pose", {})
            try:
                done.append((float(pose["y_mm"]), float(pose["z_mm"])))
            except (KeyError, TypeError, ValueError):
                pass
        return done

    def frame_path(self, label: str) -> Optional[str]:
        """Return the JPEG path to save for *label*, or None if save_frames is off."""
        if not self.config.save_frames:
            return None
        safe = label.replace(" ", "_").replace("/", "-")
        return str(self.frames_dir / f"{safe}.jpg")

    def add(self, record: PoseRecord) -> None:
        d = asdict(record)
        self._records.append(d)
        with self.output_file.open("a") as fh:
            fh.write(json.dumps(d) + "\n")

    def replace_or_add(self, record: PoseRecord) -> None:
        """Replace the existing record with the same label, or append if not found."""
        d = asdict(record)
        for i, rec in enumerate(self._records):
            if rec.get("label") == d["label"]:
                self._records[i] = d
                self._flush()
                return
        self._records.append(d)
        self._flush()

    def save_snapshot(self, path: Path) -> None:
        """Write a copy of the current records to *path* without affecting the main file."""
        tmp = path.with_suffix(".tmp")
        with tmp.open("w") as fh:
            for rec in self._records:
                fh.write(json.dumps(rec) + "\n")
        tmp.replace(path)

    def count(self) -> int:
        return len(self._records)

    def _flush(self) -> None:
        # Rewrite the entire file as JSON Lines.
        # Used only by replace_or_add() and save_snapshot() — not on every add().
        # On Linux, os.replace() is a single rename() syscall — if the process
        # is killed mid-write you get the previous complete file, never corrupt data.
        tmp = self.output_file.with_suffix(".tmp")
        with tmp.open("w") as fh:
            for rec in self._records:
                fh.write(json.dumps(rec) + "\n")
        tmp.replace(self.output_file)

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
# 7 ▸ DataCollector  (orchestrator)
# ═══════════════════════════════════════════════════════════════════════════

class DataCollector:
    """
    Top-level orchestrator.

    New parameters vs. previous version
    ------------------------------------
    domain : Domain | None
        If provided, run_sequence() will auto-generate a snake-grid path
        through the domain instead of executing the hardcoded sequence.
        X stays at its starting value; the domain defines coverage in Y-Z.
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
        domain: Optional[Domain] = None,                # ← NEW
    ) -> None:
        self.config   = config
        self.debug    = debug
        self.upload   = upload
        self.preview  = preview                         # ← NEW
        self._args_dict: Dict = args_dict or {}         # ← NEW  (for config snapshot)
        self._domain  = domain                          # ← NEW

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
            config, output_path=_preview_png, show_plot=preview_show,
            domain=domain,                              # ← NEW: draws boundary on preview
        )

    # ------------------------------------------------------------------
    # Core move primitive  (relative, cardinal directions)
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

            vision_result             = self.vision.get_stable_positions(
                save_image=self.data.frame_path(label))
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
    # Domain waypoint primitive  (absolute Y-Z target)
    # ------------------------------------------------------------------

    def _move_to_waypoint(self, y_mm: float, z_mm: float, label: str) -> None:
        """
        Move to an absolute (y_mm, z_mm) position and record a pose there.

        X is not changed (the domain sweeps in the Y-Z plane only).
        Works transparently in both simulation mode and real mode.

        Simulation mode: delta is computed from the sim cursor so that each
        waypoint in the preview lands at the correct absolute position.

        Real mode: delta is computed from the home position so that
        move_cartesian_relative() targets ``home + (dy, dz)`` — completely
        drift-free regardless of how many waypoints have been visited.
        """
        # ── Simulation mode ───────────────────────────────────────────
        if self.robot._sim_mode:
            # Use sim cursor as the reference so the preview accumulates correctly.
            _, curr_y, curr_z = self.robot.get_position()
            self.robot.move_relative(dx_mm=0.0, dy_mm=y_mm - curr_y, dz_mm=z_mm - curr_z)
            return

        # ── Real collection mode ──────────────────────────────────────
        # Delta from home: move_cartesian_relative adds the offset to the
        # stored home position, so accumulated IK drift is impossible.
        homex, hompy, hompz = self.robot.get_home_position()
        dy = y_mm - hompy
        dz = z_mm - hompz
        print(f"\n  → [Waypoint: {label}]  Moving to y={y_mm:.1f} mm, z={z_mm:.1f} mm "
              f"(dy={dy:+.1f} mm from home, dz={dz:+.1f} mm from home)")

        ok = self.robot.move_relative(dx_mm=0.0, dy_mm=dy, dz_mm=dz)
        if not ok:
            print(f"    Skipping record for waypoint {label}.")
            return

        vision_result                = self.vision.get_stable_positions(
            save_image=self.data.frame_path(label))
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
    # Movement sequence
    # ------------------------------------------------------------------

    def run_sequence(self) -> None:
        """
        Traverse the collection domain and record a pose at every grid point.

        If a domain was provided at construction, a boustrophedon (snake)
        grid is generated automatically from the domain geometry and
        ``config.grid_step_mm``.

        If no domain was provided, falls back to the original hardcoded
        sequence (edit ``_run_hardcoded_sequence`` to change it).

        This method is called TWICE: once for preview simulation and once
        for real collection.
        """
        if self._domain is not None:
            self._run_domain_sequence()
        else:
            self._run_hardcoded_sequence()

    def _run_domain_sequence(self) -> None:
        """Snake-grid traversal driven by the boundary domain."""
        waypoints = generate_snake_path(self._domain, self.config.grid_step_mm)

        if not waypoints:
            print(
                "  ⚠ No waypoints generated – check domain bounds vs grid_step_mm."
            )
            return

        # Skip waypoints already present in the dataset (resume support).
        # In sim mode this still applies so the preview reflects what remains.
        # Matching uses actual ee_pose coordinates with a 2 mm tolerance so
        # that small floating-point differences or grid-parameter changes do
        # not cause previously visited points to be revisited.
        _RESUME_TOL_MM = 2.0
        done = self.data.completed_grid_waypoints()
        def _already_done(y_mm: float, z_mm: float) -> bool:
            return any(
                abs(y_mm - dy) <= _RESUME_TOL_MM and abs(z_mm - dz) <= _RESUME_TOL_MM
                for dy, dz in done
            )
        remaining = [
            (i, y_mm, z_mm) for i, (y_mm, z_mm) in enumerate(waypoints)
            if not _already_done(y_mm, z_mm)
        ]

        if not self.robot._sim_mode:
            if done:
                print(f"\n  ↺ Resume mode: {len(done)} waypoint(s) already "
                      f"collected, {len(remaining)} remaining.")
            print(f"\n  Domain grid: {len(waypoints)} total waypoints  "
                  f"(step={self.config.grid_step_mm:.0f} mm, "
                  f"domain={type(self._domain).__name__})")

        if not remaining:
            print("  ✓ All waypoints already collected – nothing to do.")
            return

        def _fmt_duration(seconds: float) -> str:
            s = int(seconds)
            return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"

        n_remaining = len(remaining)
        run_start   = time.time()

        for idx, (i, y_mm, z_mm) in enumerate(remaining):
            label = f"grid_{i+1:04d}_y{y_mm:.0f}_z{z_mm:.0f}"
            if not self.robot._sim_mode:
                print(f"\n{'─' * 60}")
                print(f"  Waypoint {idx+1}/{n_remaining}  "
                      f"(grid {i+1}/{len(waypoints)})  "
                      f"y={y_mm:.1f} mm  z={z_mm:.1f} mm")
            self._move_to_waypoint(y_mm, z_mm, label)

            if not self.robot._sim_mode:
                completed = idx + 1
                elapsed   = time.time() - run_start
                avg_sec   = elapsed / completed
                eta_sec   = avg_sec * (n_remaining - completed)
                pct       = completed / n_remaining
                bar_len   = 30
                filled    = int(bar_len * pct)
                bar       = "█" * filled + "░" * (bar_len - filled)
                eta_str   = _fmt_duration(eta_sec) if completed < n_remaining else "0:00:00"
                print(
                    f"\n  [{bar}] {completed}/{n_remaining} ({pct * 100:.1f}%)  "
                    f"| Elapsed: {_fmt_duration(elapsed)}  "
                    f"| ETA: {eta_str}"
                )

        if not self.robot._sim_mode:
            self._retry_underdetected_waypoints()

    def _retry_underdetected_waypoints(self) -> None:
        """
        Single retry pass for grid waypoints that detected fewer markers than
        the dataset average.

        Expected marker count = ceil(mean marker count across all grid records).
        Any record below that threshold is re-visited once.  Before retrying,
        the current dataset is snapshotted to <stem>_pre_retry.json so the
        original observations are never lost.
        """
        import math

        grid_records = [
            rec for rec in self.data._records
            if rec.get("label", "").startswith("grid_")
        ]
        if not grid_records:
            return

        marker_counts = [
            len([k for k in rec.get("markers", {}).keys()])
            for rec in grid_records
        ]
        threshold = math.ceil(sum(marker_counts) / len(marker_counts))

        to_retry = [
            rec for rec in grid_records
            if len(rec.get("markers", {})) < threshold
            or not rec.get("stability_reached", True)
        ]

        if not to_retry:
            return

        n_markers  = sum(1 for r in to_retry if len(r.get("markers", {})) < threshold)
        n_unstable = sum(1 for r in to_retry if not r.get("stability_reached", True))

        print(f"\n{'═' * 60}")
        print(f"  RETRY PASS  –  {len(to_retry)} waypoint(s) flagged")
        print(f"    {n_markers} below marker threshold ({threshold})")
        print(f"    {n_unstable} stability_reached=False")
        print(f"    (may overlap)")

        snapshot_path = self.data.output_file.with_name(
            self.data.output_file.stem + "_pre_retry.json"
        )
        self.data.save_snapshot(snapshot_path)
        print(f"  Pre-retry snapshot : {snapshot_path.resolve()}")
        print(f"{'═' * 60}")

        for rec in to_retry:
            ee        = rec.get("ee_pose", {})
            y_mm      = float(ee.get("y_mm", 0.0))
            z_mm      = float(ee.get("z_mm", 0.0))
            label     = rec["label"]
            n_before  = len(rec.get("markers", {}))
            unstable  = not rec.get("stability_reached", True)
            reasons   = []
            if n_before < threshold:
                reasons.append(f"{n_before}/{threshold} markers")
            if unstable:
                reasons.append("unstable")
            print(f"\n{'─' * 60}")
            print(f"  ↺ Retry  {label}  ({', '.join(reasons)})  "
                  f"y={y_mm:.1f} mm  z={z_mm:.1f} mm")
            self._move_to_waypoint_replace(y_mm, z_mm, label)

    def _move_to_waypoint_replace(self, y_mm: float, z_mm: float, label: str) -> None:
        """Like _move_to_waypoint but uses replace_or_add instead of add."""
        # Same home-relative approach as _move_to_waypoint so retries are
        # drift-free and independent of the current encoder reading.
        homex, hompy, hompz = self.robot.get_home_position()
        dy = y_mm - hompy
        dz = z_mm - hompz

        ok = self.robot.move_relative(dx_mm=0.0, dy_mm=dy, dz_mm=dz)
        if not ok:
            print(f"    Skipping retry for waypoint {label} – IK failed.")
            return

        vision_result                = self.vision.get_stable_positions(
            save_image=self.data.frame_path(label))
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
        self.data.replace_or_add(record)

        detected = list(markers.keys())
        print(
            f"  ✓ Retry recorded  ee=({actual_x:.1f}, {actual_y:.1f}, {actual_z:.1f}) mm  "
            f"detected={detected or 'none'}  "
            f"stable={record.stability_reached}  "
            f"t={record.stability_elapsed_sec:.2f}s"
        )

    def _run_hardcoded_sequence(self) -> None:
        """
        Original hardcoded traversal (fallback when no domain is given).

        Edit this method to change the manual trajectory.
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
        vision_result                = self.vision.get_stable_positions(
            save_image=self.data.frame_path("start"))
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
        2. Simulate trajectory → plot preview → wait for y/n.
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
        if self._domain is not None:
            print(f"  Domain       : {type(self._domain).__name__}  "
                  f"(step={self.config.grid_step_mm:.0f} mm)")
        else:
            print(f"  Domain       : hardcoded sequence")
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

        self.robot.move_home()  # Ensure we start from a known position (also for preview)

        # ── Preview ────────────────────────────────────────────────────
        if self.preview:
            if not self._preview_and_confirm():
                print("Exiting without data collection.")
                return
        # ──────────────────────────────────────────────────────────────

        try:
            self.vision.start()

            # ── Save config snapshot ───────────────────────────────────
            print("\nSaving experiment config snapshot …")
            first_frame = self.vision.capture_first_frame()
            preview_png = self._previewer.output_path if self.preview else None
            self.data.save_experiment_config(
                args_dict=self._args_dict,
                trajectory_preview_png=preview_png,
                first_frame=first_frame,
            )
            # ──────────────────────────────────────────────────────────

            self._record_start_pose()
            self.run_sequence()
        except KeyboardInterrupt:
            print("\n  ⚠ Interrupted – partial data saved.")
            try:
                self.robot.move_home()
            except Exception as e:
                print(f"  ⚠ Could not return to home after interrupt: {e}")
        finally:
            try:
                self.vision.stop()
            except Exception as e:
                print(f"  ⚠ Camera stop failed: {e}")
            self.data.write_session_index()

            # ── Dataset status plot ────────────────────────────────────
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

def _build_domain_from_args(args: argparse.Namespace) -> Optional[Domain]:
    """
    Construct a Domain from parsed CLI arguments.
    Returns None if --domain was not specified (falls back to hardcoded sequence).
    """
    if args.domain is None:
        return None

    if args.domain == "rect":
        missing = [f for f in ("domain_y_min", "domain_y_max",
                               "domain_z_min", "domain_z_max")
                   if getattr(args, f) is None]
        if missing:
            raise SystemExit(
                f"--domain rect requires: {', '.join('--' + m.replace('_', '-') for m in missing)}"
            )
        return RectangularDomain(
            y_min_mm=args.domain_y_min,
            y_max_mm=args.domain_y_max,
            z_min_mm=args.domain_z_min,
            z_max_mm=args.domain_z_max,
        )

    if args.domain == "semicircle":
        missing = [f for f in ("domain_center_y", "domain_center_z", "domain_radius")
                   if getattr(args, f) is None]
        if missing:
            raise SystemExit(
                f"--domain semicircle requires: {', '.join('--' + m.replace('_', '-') for m in missing)}"
            )
        return SemicircleDomain(
            center_y_mm=args.domain_center_y,
            center_z_mm=args.domain_center_z,
            radius_mm=args.domain_radius,
            orientation=args.domain_orientation,
        )

    raise SystemExit(f"Unknown domain type: {args.domain}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Slinky ArUco data collection – domain-based trajectory (Sawyer + RealSense).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--debug",            action="store_true")
    p.add_argument("--upload",           action="store_true")
    p.add_argument("--output",           type=Path, default=None, metavar="FILE")
    p.add_argument("--experiment-name",  type=str,  default="DLO_slinky_test", metavar="NAME")

    # Preview
    g = p.add_argument_group("Trajectory preview")
    g.add_argument("--no-preview",      action="store_true",
                   help="Skip trajectory preview and go straight to collection.")
    g.add_argument("--preview-output",  type=Path, default=None, metavar="FILE",
                   help="Path for the preview PNG.")
    g.add_argument("--preview-show",    action="store_true",
                   help="Open the matplotlib GUI window (requires a display).")

    # Boundary domain
    d = p.add_argument_group(
        "Boundary domain",
        "Define the region to collect data in.  If omitted, the hardcoded "
        "sequence in _run_hardcoded_sequence() is used instead.",
    )
    d.add_argument(
        "--domain", choices=["rect", "semicircle"], default=None,
        help=(
            "Shape of the collection domain in the Y-Z plane.  "
            "'rect' = rectangular region (e.g. slinky).  "
            "'semicircle' = half-disk (e.g. taut elastic strip)."
        ),
    )
    d.add_argument("--grid-step",         type=float, default=5.0, metavar="MM",
                   help="Grid spacing between sample points (mm).")
    # Rectangular domain
    d.add_argument("--domain-y-min",      type=float, default=None, metavar="MM",
                   help="[rect] Minimum Y coordinate of the rectangle (mm).")
    d.add_argument("--domain-y-max",      type=float, default=None, metavar="MM",
                   help="[rect] Maximum Y coordinate of the rectangle (mm).")
    d.add_argument("--domain-z-min",      type=float, default=None, metavar="MM",
                   help="[rect] Minimum Z coordinate of the rectangle (mm).")
    d.add_argument("--domain-z-max",      type=float, default=None, metavar="MM",
                   help="[rect] Maximum Z coordinate of the rectangle (mm).")
    # Semicircle domain
    d.add_argument("--domain-center-y",   type=float, default=None, metavar="MM",
                   help="[semicircle] Y coordinate of the circle centre (mm).")
    d.add_argument("--domain-center-z",   type=float, default=None, metavar="MM",
                   help="[semicircle] Z coordinate of the circle centre (mm).")
    d.add_argument("--domain-radius",     type=float, default=None, metavar="MM",
                   help="[semicircle] Radius = max reach of the elastic strip (mm).")
    d.add_argument(
        "--domain-orientation",
        choices=["up", "down", "left", "right"], default="up",
        help=(
            "[semicircle] Which half of the circle to keep.  "
            "'up' keeps z >= center (arc opens upward); "
            "'down' keeps z <= center."
        ),
    )

    # Motion
    p.add_argument("--step-mm",            type=float, default=2,  metavar="MM",
                   help="Default step size for cardinal-direction moves (mm).")
    p.add_argument("--move-timeout",       type=float, default=15.0,  metavar="SEC")
    p.add_argument("--joint-speed",        type=float, default=0.01,  metavar="FRAC")

    # Stability
    p.add_argument("--stability-window",    type=int,   default=8,     metavar="N")
    p.add_argument("--stability-threshold", type=float, default=0.005, metavar="M")
    p.add_argument("--stability-timeout",   type=float, default=120.0, metavar="SEC")

    # Frame saving
    p.add_argument("--no-live-preview",        action="store_true",
                   help="Disable latest_frame.png writes entirely (Streamlit feed).")
    p.add_argument("--live-preview-interval",  type=float, default=0.5, metavar="SEC",
                   help="Minimum seconds between latest_frame.png writes (default 0.5).")
    p.add_argument("--save-frames",            action="store_true",
                   help="Save one annotated JPEG per waypoint for timelapse/gif.")

    # Output
    p.add_argument("--output-dir", type=Path, default=Path("."), metavar="DIR",
                   help="Root directory for all outputs (dataset, configs, frames).")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    domain = _build_domain_from_args(args)

    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        default_step_mm=args.step_mm,
        grid_step_mm=args.grid_step,
        move_timeout_sec=args.move_timeout,
        joint_speed=args.joint_speed,
        stability_window=args.stability_window,
        stability_threshold_m=args.stability_threshold,
        stability_timeout_sec=args.stability_timeout,
        live_preview=not args.no_live_preview,
        live_preview_interval_sec=args.live_preview_interval,
        save_frames=args.save_frames,
    )
    collector = DataCollector(
        config=config,
        debug=args.debug,
        upload=args.upload,
        preview=not args.no_preview,
        preview_output=args.preview_output,
        preview_show=args.preview_show,
        output_file=args.output,
        args_dict=vars(args),
        domain=domain,
    )
    collector.run()


if __name__ == "__main__":
    main()
