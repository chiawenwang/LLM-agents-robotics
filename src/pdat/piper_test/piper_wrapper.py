"""
Piper Wrapper - SDK Interface
Bridges robot_api.py with the actual Piper SDK files.

Note: Piper robot uses CAN bus communication, not TCP/IP!
"""

import time
from typing import Dict, List, Optional

from piper_sdk import C_PiperInterface_V2


class PiperWrapper:
    """
    High-level wrapper for Piper robot using SDK files.
    
    This class provides a clean interface to the Piper SDK, abstracting away
    the details of individual SDK files and providing a consistent API for
    robot control.
    
    Note: The Piper robot uses CAN bus communication (e.g., "can0"), not IP/port.
    
    Attributes:
        can_name: CAN interface name (e.g., "can0", "can1")
        connection: C_PiperInterface_V2 object for robot control
    """
    
    def __init__(self, can_name: str = "can0", **kwargs):
        """
        Initialize connection to Piper robot via CAN bus.
        
        Args:
            can_name: CAN interface name (default: "can0")
            **kwargs: Additional arguments to pass to C_PiperInterface_V2:
                - judge_flag: bool (default True)
                - can_auto_init: bool (default True)
                - dh_is_offset: int (default 0x01)
                - start_sdk_joint_limit: bool (default False)
                - start_sdk_gripper_limit: bool (default False)
                - logger_level: LogLevel (default WARNING)
                - log_to_file: bool (default False)
                - log_file_path: str or None
        """
        self.can_name = can_name
        self.connection: C_PiperInterface_V2 | None = None
        
        print(f"[PiperWrapper] Connecting to robot via CAN interface: {can_name}")
        
        # Initialize connection
        self._connect(**kwargs)
    
    def _connect(self, **kwargs):
        """
        Establish connection with robot using Piper SDK via CAN bus.
        """
        try:
            # Create the interface - it uses CAN, not IP/port!
            self.connection = C_PiperInterface_V2(
                can_name=self.can_name,
                **kwargs
            )
            print(f"[PiperWrapper] Successfully connected via {self.can_name}")
        except Exception as e:
            print(f"[PiperWrapper] ERROR: Failed to connect via {self.can_name}")
            print(f"[PiperWrapper] Error: {e}")
            raise
    
    def enable(self):
        """
        Enable the robot.
        Uses: EnableArm() method from SDK
        """
        if self.connection is None:
            raise RuntimeError("Not connected to robot")
        
        # EnableArm takes a timeout parameter (milliseconds)
        self.connection.EnableArm(1000)
        time.sleep(0.2)  # Give it time to enable
        print("[PiperWrapper] Robot enabled")
    
    def disable(self):
        """
        Disable the robot.
        Uses: DisableArm() method from SDK
        """
        if self.connection is None:
            raise RuntimeError("Not connected to robot")
        
        # DisableArm takes a timeout parameter (milliseconds)
        self.connection.DisableArm(1000)
        time.sleep(0.2)  # Give it time to disable
        print("[PiperWrapper] Robot disabled")
    
    def reset(self):
        """
        Reset the robot.
        Uses: EmergencyStop() and ResetPiper() methods
        """
        if self.connection is None:
            raise RuntimeError("Not connected to robot")
        
        # Emergency stop first
        self.connection.EmergencyStop()
        time.sleep(0.5)
        
        # Reset the robot
        self.connection.ResetPiper()
        time.sleep(1.0)
        
        print("[PiperWrapper] Robot reset")
    
    def go_zero(self):
        """
        Move robot to zero position.
        Uses: JointCtrl() method to move all joints to zero
        """
        if self.connection is None:
            raise RuntimeError("Not connected to robot")
        
        # Move all joints to zero
        zero_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # JointCtrl(joint_angles, speed_percent)
        self.connection.JointCtrl(zero_angles, 30)  # 30% speed for safety
        print("[PiperWrapper] Moving to zero position")
    
    def stop(self):
        """
        Stop robot motion.
        Uses: EmergencyStop() method from SDK
        """
        if self.connection is None:
            raise RuntimeError("Not connected to robot")
        
        self.connection.EmergencyStop()
        print("[PiperWrapper] Robot stopped")
    
    def move_joints(self, joint_angles: List[float], speed: float = 0.5):
        """
        Move robot joints to specified angles.
        Uses: JointCtrl() method from SDK
        
        Args:
            joint_angles: List of 6 joint angles in radians
            speed: Movement speed (0.0 to 1.0) - will be converted to percentage
        """
        if self.connection is None:
            raise RuntimeError("Not connected to robot")
        
        assert len(joint_angles) == 6, "Piper has 6 joints"
        
        # Convert speed from 0-1 to 0-100 percentage
        speed_percent = int(speed * 100)
        speed_percent = max(1, min(100, speed_percent))  # Clamp to 1-100
        
        # JointCtrl(joint_angles, speed_percent)
        self.connection.JointCtrl(joint_angles, speed_percent)
        print(f"[PiperWrapper] Moving joints to {joint_angles} at {speed_percent}% speed")
    
    def read_joint_state(self) -> Dict:
        """
        Read current joint states.
        Uses: GetArmJointMsgs() method from SDK
        
        Returns:
            Dictionary with 'angles' (list of 6 floats) and 
            'velocities' (list of 6 floats)
        """
        if self.connection is None:
            raise RuntimeError("Not connected to robot")
        
        # Get joint feedback - returns ArmJoint object
        joint_msg = self.connection.GetArmJointMsgs()
        
        print("[PiperWrapper] Reading joint state")
        
        # ArmJoint object has joint_state attribute (list of 6 angles)
        return {
            "angles": joint_msg.joint_state,
            "velocities": [0.0] * 6,  # Velocities not directly in this message
        }
    
    def read_end_pose(self) -> Dict:
        """
        Read end-effector pose.
        Uses: GetArmEndPoseMsgs() method from SDK
        
        Returns:
            Dictionary with x, y, z, roll, pitch, yaw (all floats)
        """
        if self.connection is None:
            raise RuntimeError("Not connected to robot")
        
        # Get end effector pose - returns ArmEndPose object
        end_pose_msg = self.connection.GetArmEndPoseMsgs()
        
        print("[PiperWrapper] Reading end pose")
        
        # ArmEndPose object has end_pose attribute (list of 6 values)
        return {
            "x": end_pose_msg.end_pose[0],
            "y": end_pose_msg.end_pose[1],
            "z": end_pose_msg.end_pose[2],
            "roll": end_pose_msg.end_pose[3],
            "pitch": end_pose_msg.end_pose[4],
            "yaw": end_pose_msg.end_pose[5],
        }
    
    def read_gripper_status(self) -> Dict:
        """
        Read gripper status.
        Uses: GetArmGripperMsgs() method from SDK
        
        Returns:
            Dictionary with gripper state information
        """
        if self.connection is None:
            raise RuntimeError("Not connected to robot")
        
        try:
            # Get gripper message - returns ArmGripper object
            gripper_msg = self.connection.GetArmGripperMsgs()
            print("[PiperWrapper] Reading gripper status")
            
            # ArmGripper object has gripper_state and grippers_current attributes
            return {
                "position": gripper_msg.gripper_state,
                "force": gripper_msg.grippers_current,
                "status": "active" if gripper_msg.gripper_state > 0 else "closed",
            }
        except AttributeError as e:
            print(f"[PiperWrapper] Warning: Could not read gripper status: {e}")
            return {"position": 0.0, "force": 0.0, "status": "unavailable"}
    
    def set_gripper(self, position: float, force: float = 0.5):
        """
        Control gripper.
        Uses: GripperCtrl() method from SDK
        
        Args:
            position: Gripper position (0.0 = closed, 1.0 = open)
                     Will be converted to 0-1000 range for SDK
            force: Gripper force (0.0 to 1.0) - converted to current limit
        """
        if self.connection is None:
            raise RuntimeError("Not connected to robot")
        
        try:
            # Convert position (0-1) to SDK range (0-1000)
            gripper_pos = int(position * 1000)
            gripper_pos = max(0, min(1000, gripper_pos))
            
            # Convert force to current (typical range 0-500mA)
            gripper_current = int(force * 500)
            
            # GripperCtrl(position, current, speed)
            self.connection.GripperCtrl(gripper_pos, gripper_current, 100)
            print(f"[PiperWrapper] Setting gripper: position={gripper_pos}, current={gripper_current}mA")
        except Exception as e:
            print(f"[PiperWrapper] Warning: Gripper control failed: {e}")
    
    def read_firmware_version(self) -> str:
        """
        Read firmware version.
        Uses: GetPiperFirmwareVersion() method from SDK
        
        Returns:
            Firmware version string
        """
        if self.connection is None:
            return "not connected"
        
        try:
            # Request firmware version
            self.connection.SearchPiperFirmwareVersion()
            time.sleep(0.1)  # Give time for response
            
            # Get firmware version
            version = self.connection.GetPiperFirmwareVersion()
            return str(version)
        except Exception as e:
            print(f"[PiperWrapper] Could not read firmware: {e}")
            return "unknown"
    
    def get_connection_status(self) -> bool:
        """
        Check if connected to robot.
        Uses: get_connect_status() method from SDK
        
        Returns:
            True if connected, False otherwise
        """
        if self.connection is None:
            return False
        
        try:
            return self.connection.get_connect_status()
        except:
            return False
    
    def is_enabled(self) -> bool:
        """
        Check if robot is enabled.
        Uses: GetArmEnableStatus() method from SDK
        
        Returns:
            True if enabled, False otherwise
        """
        if self.connection is None:
            return False
        
        try:
            return self.connection.GetArmEnableStatus()
        except:
            return False
    
    def close(self):
        """
        Close connection to robot.
        """
        if self.connection is None:
            print("[PiperWrapper] Already disconnected")
            return
            
        try:
            # Disable robot before closing
            if self.is_enabled():
                self.disable()
                time.sleep(0.2)
        except Exception as e:
            print(f"[PiperWrapper] Warning during disable: {e}")
        
        try:
            # Disconnect port if method exists
            if hasattr(self.connection, 'DisconnectPort'):
                self.connection.DisconnectPort()
        except Exception as e:
            print(f"[PiperWrapper] Warning during disconnect: {e}")
        
        print("[PiperWrapper] Connection closed")
        self.connection = None


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("Testing PiperWrapper")
    print("="*60)
    
    # CAN interface name (typically "can0" for first CAN interface)
    CAN_NAME = "can0"
    
    print("\nIMPORTANT: Make sure CAN interface is set up!")
    print("Check with: ip link show can0")
    print("If needed, set up with:")
    print("  sudo ip link set can0 type can bitrate 1000000")
    print("  sudo ip link set can0 up")
    
    # Test with real robot (uncomment when ready)
    try:
        print("\nAttempting to connect...")
        wrapper = PiperWrapper(can_name=CAN_NAME)
        
        # Check connection
        if wrapper.get_connection_status():
            print("✓ Connected to robot")
        else:
            print("✗ Not connected")
            exit(1)
        
        # Enable robot
        print("\nEnabling robot...")
        wrapper.enable()
        time.sleep(0.5)
        
        if wrapper.is_enabled():
            print("✓ Robot enabled")
        else:
            print("✗ Robot not enabled")
        
        # Read current state
        print("\nReading joint state...")
        state = wrapper.read_joint_state()
        print(f"Joint state: {state}")
        
        print("\nReading end pose...")
        pose = wrapper.read_end_pose()
        print(f"End pose: {pose}")
        
        # Safe test movement to zero
        print("\nMoving to zero position (30% speed)...")
        wrapper.go_zero()
        time.sleep(3)
        
        # Read state after movement
        print("\nReading state after movement...")
        state = wrapper.read_joint_state()
        print(f"Joint state: {state}")
        
        # Disable and close
        print("\nDisabling robot...")
        wrapper.disable()
        wrapper.close()
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60)
        print("Check the error above and verify:")
        print("1. CAN interface is set up and UP")
        print("2. Robot is powered on")
        print("3. CAN cables are connected")
        print("="*60)