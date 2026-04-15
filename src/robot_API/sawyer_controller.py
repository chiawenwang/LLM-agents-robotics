"""
sawyer_controller.py
====================
Sawyer arm controller wrapping intera_interface (Limb, Gripper, RobotEnable).

Changes from the original
--------------------------
- `get_endpoint_pose()` added – returns the full 6-DoF pose dict so callers
  can read orientation as well as position.
- Plane-restricted motion (move_relative_xz / move_to_xz / move_relative_yz /
  move_to_yz) replaced by a unified 3-D Cartesian API:
    - `get_position()` – returns (X, Y, Z) in mm.
    - `move_relative()` – relative move by (dx, dy, dz) in mm, orientation unchanged.
    - `move_to()` – absolute move to (x, y, z) in mm, orientation unchanged.
    - `move_relative_async()` – non-blocking version of `move_relative()`.
    - `is_moving` / `last_move_result()` – async move state.
- `reset_to_pose()` added – moves to the stored home joint configuration.
- All `Optional` / `Tuple` / `Dict` / `List` / `Any` imports kept.

Requires ROS/Intera environment (source sawyer_robot_ws via intera.sh).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Lazy ROS/Intera import
# ---------------------------------------------------------------------------

def _import_intera():
    """Import ROS and Intera SDK lazily so the module can be loaded in
    environments where neither is installed (e.g. offline unit tests)."""
    import rospy
    import intera_interface
    from intera_interface import CHECK_VERSION
    return rospy, intera_interface, CHECK_VERSION


# ---------------------------------------------------------------------------
# SawyerArmController
# ---------------------------------------------------------------------------

class SawyerArmController:
    """
    Thin wrapper around the Intera SDK: Limb + Gripper + RobotEnable.

    Workflow
    --------
    1. Instantiate with ``init_node=True`` when you own the ROS node, or
       ``init_node=False`` when the caller has already called
       ``rospy.init_node()``.
    2. Call ``enable()`` before any motion.
    3. Use ``move_relative()`` / ``move_to()`` for 3-D Cartesian moves, or
       ``set_joint_positions()`` for direct joint control.
    4. Inspect ``get_position()`` or ``get_endpoint_pose()`` for feedback.
    5. Call ``disable()`` when finished (optional; the hardware does this on
       E-stop anyway).
    """

    # ------------------------------------------------------------------
    # Construction / lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        limb: str = "right",
        init_node: bool = False,
        joint_speed: float = 0.01,
    ) -> None:
        """
        Parameters
        ----------
        limb:
            Which limb to control – ``"right"`` (default) or ``"left"``.
        init_node:
            If ``True`` and rospy is not yet initialised, call
            ``rospy.init_node()`` anonymously.  Set to ``False`` when the
            caller owns the node (e.g. inside ``data_collection.py``).
        joint_speed:
            Fractional joint speed [0.0, 1.0] passed to
            ``set_joint_position_speed()``.  The original file used 0.01
            (very slow / safe).  0.3 is a reasonable default for normal
            data-collection moves; lower it if you need extra caution.
        """
        rospy, intera_interface, CHECK_VERSION = _import_intera()

        if init_node and not rospy.core.is_initialized():
            rospy.init_node("sawyer_arm_controller", anonymous=True)

        self._limb_name: str = limb
        self._limb = intera_interface.Limb(limb)
        self._limb.set_joint_position_speed(joint_speed)

        self._rs = intera_interface.RobotEnable(CHECK_VERSION)

        # Home position
        # Home joint angles: [0.5733994140625, -1.3600634765625, 0.198025390625, 1.9905546875, 2.047806640625, 0.9621435546875, 2.4713955078125]
        # Home end effector pose: {'position': Point(x=0.472317521880045, y=0.4989624087475566, z=0.5607591405299359), 'orientation': Quaternion(x=0.4861507340849625, y=0.5167452194383757, z=0.5062532171850699, w=-0.49024434930553207)}
        
        # self._home_joint_angles: Dict[str, float] = {
        #     "right_j0": 0.5733994140625,
        #     "right_j1": -1.3600634765625,
        #     "right_j2": 0.198025390625,
        #     "right_j3": 1.9905546875,
        #     "right_j4": 2.047806640625,
        #     "right_j5": 0.9621435546875,
        #     "right_j6": 2.4713955078125,
        # }

        self._home_joint_angles: Dict[str, float] = {
            "right_j0": 0.939201171875,
            "right_j1": -0.85804296875,
            "right_j2": -0.2770380859375,
            "right_j3": 1.1909169921875,
            "right_j4": 1.9205302734375,
            "right_j5": 0.996357421875,
            "right_j6": 2.8419912109375,
        }
        # rosrun go_to_joint_positions.py -q 0.0573 -1.3600 0.1980 1.9905 2.0478 0.9621 2.4713 -s 0.01

        # Gripper is optional – not all Sawyer setups have one
        self._gripper: Optional[Any] = None
        try:
            self._gripper = intera_interface.Gripper(f"{limb}_gripper")
        except Exception:
            pass

        # Async move state
        self._move_thread: Optional[threading.Thread] = None
        self._move_result: Optional[bool] = None

        # ------------------------------------------------------------------
        # Home Cartesian pose – captured once at startup so that all
        # move_relative() calls target an absolute position anchored to home,
        # eliminating accumulated IK drift over thousands of moves.
        # ------------------------------------------------------------------
        _home_ep = self._limb.endpoint_pose()
        _hp = _home_ep["position"]
        _ho = _home_ep["orientation"]
        # Home position in metres (robot base frame)
        self._home_pos_x: float = float(_hp.x)
        self._home_pos_y: float = float(_hp.y)
        self._home_pos_z: float = float(_hp.z)
        # Home orientation quaternion – kept fixed for every move
        self._home_ori_x: float = float(_ho.x)
        self._home_ori_y: float = float(_ho.y)
        self._home_ori_z: float = float(_ho.z)
        self._home_ori_w: float = float(_ho.w)
        # Logical offset (mm) from home that the controller is currently
        # commanded to.  Updated on every successful move_relative / move_to
        # call so that repeated relative moves compose correctly without
        # reading the noisy actual position.
        self._cmd_dx_mm: float = 0.0
        self._cmd_dy_mm: float = 0.0
        self._cmd_dz_mm: float = 0.0

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def enable(self) -> None:
        """Enable the robot. Must be called before any motion command."""
        self._rs.enable()

    def disable(self) -> None:
        """Disable the robot (safe state, motors de-energised)."""
        self._rs.disable()

    # ------------------------------------------------------------------
    # Joint-level access
    # ------------------------------------------------------------------

    def joint_names(self) -> List[str]:
        """Return the ordered list of joint names for this limb."""
        return list(self._limb.joint_names())

    def joint_angles(self) -> Dict[str, float]:
        """Return the current joint angles as ``{joint_name: radians}``."""
        return dict(self._limb.joint_angles())

    def joint_angle(self, name: str) -> float:
        """Return the current angle of a single joint (radians)."""
        return float(self._limb.joint_angle(name))

    def set_joint_positions(
        self,
        positions: Dict[str, float],
        timeout: float = 10.0,
    ) -> None:
        """
        Command joint positions directly.

        Parameters
        ----------
        positions:
            ``{joint_name: angle_in_radians}``
        timeout:
            Forwarded to ``move_to_joint_positions``.
        """
        self._limb.move_to_joint_positions(positions, timeout=timeout)

    # ------------------------------------------------------------------
    # Pose / position read-back
    # ------------------------------------------------------------------

    def get_endpoint_pose(self) -> Dict[str, Any]:
        """
        Return the full end-effector pose from ``endpoint_pose()``.

        Returns
        -------
        Dict with keys ``"position"`` (x, y, z in metres) and
        ``"orientation"`` (x, y, z, w quaternion) as returned by Intera.
        """
        return self._limb.endpoint_pose()

    def get_position(self) -> Tuple[float, float, float]:
        """
        Return end-effector (X, Y, Z) in **millimetres** (robot base frame).

        Returns
        -------
        ``(x_mm, y_mm, z_mm)``
        """
        pos = self._limb.endpoint_pose()["position"]
        return float(pos.x * 1000.0), float(pos.y * 1000.0), float(pos.z * 1000.0)

    def get_commanded_position(self) -> Tuple[float, float, float]:
        """
        Return the **commanded** end-effector position in millimetres.

        Unlike ``get_position()``, which reads the actual encoder-based pose
        (and therefore accumulates sensor noise), this value is derived purely
        from the logical offsets applied since the last home reset.  It is the
        ground-truth target that ``move_relative`` / ``move_to`` aim for.

        Returns
        -------
        ``(x_mm, y_mm, z_mm)`` – absolute position in the robot base frame.
        """
        return (
            self._home_pos_x * 1000.0 + self._cmd_dx_mm,
            self._home_pos_y * 1000.0 + self._cmd_dy_mm,
            self._home_pos_z * 1000.0 + self._cmd_dz_mm,
        )

    def move_cartesian_relative(
        self,
        dx_mm: float = 0.0,
        dy_mm: float = 0.0,
        dz_mm: float = 0.0,
        timeout: float = 360.0,
        joint_speed: float = 0.15,
        end_point: str = "right_hand",
    ) -> bool:
        """
        Move the end-effector to ``home_position + (dx_mm, dy_mm, dz_mm)``,
        keeping orientation unchanged.

        Solves the target Cartesian pose with Intera's IK service and then
        executes a joint-space move, identical to ``move_relative()``.

        Parameters
        ----------
        dx_mm, dy_mm, dz_mm:
            Displacement from home position in mm (any axis defaults to 0).
        timeout:
            Motion timeout in seconds forwarded to ``move_to_joint_positions``.
        joint_speed:
            Fractional joint speed [0.0, 1.0] set before the move.
        end_point:
            Intera IK end-point name (default ``"right_hand"``).

        Returns
        -------
        ``True`` on success, ``False`` if IK failed or motion timed out.
        """
        # print(f"Moving Cartesian relative: dx={dx_mm} mm, dy={dy_mm} mm, dz={dz_mm} mm")
        import rospy
        from geometry_msgs.msg import PoseStamped
        from intera_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest

        # Build absolute target pose (home + given offset)
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = "base"
        pose_stamped.pose.position.x = self._home_pos_x + dx_mm / 1000.0
        pose_stamped.pose.position.y = self._home_pos_y + dy_mm / 1000.0
        pose_stamped.pose.position.z = self._home_pos_z + dz_mm / 1000.0
        pose_stamped.pose.orientation.x = self._home_ori_x
        pose_stamped.pose.orientation.y = self._home_ori_y
        pose_stamped.pose.orientation.z = self._home_ori_z
        pose_stamped.pose.orientation.w = self._home_ori_w

        # print(f"Target pose for IK:\n  Position (m): ({pose_stamped.pose.position.x:.3f}, "
        #       f"{pose_stamped.pose.position.y:.3f}, {pose_stamped.pose.position.z:.3f})\n"
        #       f"  Orientation (quat): ({pose_stamped.pose.orientation.x:.3f}, "
        #       f"{pose_stamped.pose.orientation.y:.3f}, {pose_stamped.pose.orientation.z:.3f}, "
        #       f"{pose_stamped.pose.orientation.w:.3f})")

        # Call the IK service
        ik_service = f"ExternalTools/{self._limb_name}/PositionKinematicsNode/IKService"
        rospy.wait_for_service(ik_service, timeout=5.0)
        iksvc = rospy.ServiceProxy(ik_service, SolvePositionIK)

        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(pose_stamped)
        ikreq.tip_names.append(end_point)

        resp = iksvc(ikreq)
        if not resp.result_type or resp.result_type[0] < 0:
            print("IK failed: no solution found for the target pose.")
            return False

        joint_solution = dict(zip(resp.joints[0].name, resp.joints[0].position))

        # Execute joint-space move
        self._limb.set_joint_position_speed(joint_speed)
        try:
            self._limb.move_to_joint_positions(joint_solution, timeout=timeout)
        except Exception as e:
            print(f"Joint move failed: {e}")
            return False

        # Update commanded offsets on success
        self._cmd_dx_mm = dx_mm
        self._cmd_dy_mm = dy_mm
        self._cmd_dz_mm = dz_mm
        return True

    # ------------------------------------------------------------------
    # Reset to home
    # ------------------------------------------------------------------

    def move_home(self, timeout: float = 360.0) -> bool:
        """
        Move the arm back to the stored home joint configuration.

        Parameters
        ----------
        timeout:
            Motion timeout in seconds.
        """
        try:
            print("Moving to home joint configuration...")
            print(f"  Home joint angles: {self._home_joint_angles}")
            print(f"current joint angles: {self.joint_angles()}")
            self._limb.move_to_joint_positions(self._home_joint_angles, timeout=timeout)
        except Exception as e:
            print(f"Error occurred while moving to home: {e}")
            return False
        # Reset the commanded offset so subsequent move_relative calls are
        # again expressed relative to the true home position.
        self._cmd_dx_mm = 0.0
        self._cmd_dy_mm = 0.0
        self._cmd_dz_mm = 0.0
        return True

    # ------------------------------------------------------------------
    # Async motion (non-blocking)
    # ------------------------------------------------------------------

    def move_relative_async(
        self,
        dx_mm: float = 0.0,
        dy_mm: float = 0.0,
        dz_mm: float = 0.0,
        timeout: float = 15.0,
        end_point: str = "right_hand",
    ) -> None:
        """
        Non-blocking 3-D move. Runs ``move_relative`` in a daemon thread.

        Poll ``is_moving`` to check completion; ``last_move_result()`` returns
        ``True`` / ``False`` once the thread finishes.

        Parameters
        ----------
        dx_mm, dy_mm, dz_mm:
            Displacements in mm (any axis defaults to 0.0).
        timeout:
            Forwarded to ``move_relative``.
        end_point:
            Intera IK end-point name.
        """
        def _run() -> None:
            self._move_result = self.move_relative(
                dx_mm=dx_mm, dy_mm=dy_mm, dz_mm=dz_mm,
                timeout=timeout, end_point=end_point,
            )

        self._move_result = None
        self._move_thread = threading.Thread(target=_run, daemon=True)
        self._move_thread.start()

    @property
    def is_moving(self) -> bool:
        """``True`` while a background ``move_relative_async`` is running."""
        return self._move_thread is not None and self._move_thread.is_alive()

    def last_move_result(self) -> Optional[bool]:
        """
        Return the result of the most recent async move.

        Returns ``None`` while the move is still in progress, ``True`` on
        success, and ``False`` on IK failure.
        """
        return None if self.is_moving else self._move_result

    # ------------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------------

    def gripper_open(self) -> None:
        """Open the gripper (no-op if no gripper is attached)."""
        if self._gripper is not None:
            self._gripper.open()

    def gripper_close(self) -> None:
        """Close the gripper (no-op if no gripper is attached)."""
        if self._gripper is not None:
            self._gripper.close()

    def gripper_calibrate(self) -> None:
        """Calibrate the gripper (no-op if no gripper is attached)."""
        if self._gripper is not None:
            self._gripper.calibrate()

    # ------------------------------------------------------------------
    # Raw SDK access (escape hatch)
    # ------------------------------------------------------------------

    @property
    def limb(self) -> Any:
        """Direct reference to ``intera_interface.Limb`` for advanced use."""
        return self._limb

    @property
    def gripper(self) -> Optional[Any]:
        """Direct reference to ``intera_interface.Gripper``, or ``None``."""
        return self._gripper

if __name__ == "__main__":
    # Quick test: move in a small square pattern around the current position
    controller = SawyerArmController(init_node=True, joint_speed=0.1)
    # controller.enable()
    print("Current position (mm):", controller.get_position())
    controller.move_home()
    print("Moved to home position.")

    # time.sleep(2.0)
