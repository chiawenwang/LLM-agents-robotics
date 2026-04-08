"""
Mock Sawyer Robot Interface
Mimics the Intera SDK API structure for testing agent logic.

This is a SIMULATION/MOCK implementation. Replace marked sections with real
Intera SDK calls when you have access to the physical robot.

References:
- Intera SDK: https://rethinkrobotics.github.io/intera_sdk_docs/5.0.4/intera_interface/html/index.html
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import json


class Limb:
    """
    Mock implementation of intera_interface.Limb
    
    Real implementation: from intera_interface import Limb
    """
    
    def __init__(self, limb_name: str = "right"):
        self.name = limb_name
        self._joint_names = [
            f"{limb_name}_j0",
            f"{limb_name}_j1", 
            f"{limb_name}_j2",
            f"{limb_name}_j3",
            f"{limb_name}_j4",
            f"{limb_name}_j5",
            f"{limb_name}_j6"
        ]
        # Mock current joint angles (neutral position)
        self._joint_angles = {name: 0.0 for name in self._joint_names}
        
    def joint_names(self) -> List[str]:
        """Get list of joint names."""
        return self._joint_names
    
    def joint_angles(self) -> Dict[str, float]:
        """
        Get current joint angles.
        
        REPLACE WITH REAL ROBOT:
        This will return actual joint encoder readings from the robot.
        """
        # MOCK: Return simulated joint angles
        return self._joint_angles.copy()
    
    def move_to_joint_positions(self, positions: Dict[str, float], timeout: float = 15.0):
        """
        Move to specified joint positions.
        
        REPLACE WITH REAL ROBOT:
        from intera_interface import Limb
        limb = Limb('right')
        limb.move_to_joint_positions(positions)
        
        Args:
            positions: Dict mapping joint names to target angles (radians)
            timeout: Maximum time to wait for movement completion
        """
        # MOCK: Simulate movement by updating internal state
        print(f"[MOCK] Moving to joint positions: {positions}")
        time.sleep(0.1)  # Simulate movement time
        self._joint_angles.update(positions)
        print(f"[MOCK] Movement complete")
        
    def set_joint_position_speed(self, speed: float):
        """
        Set joint position movement speed (0.0 - 1.0).
        
        REPLACE WITH REAL ROBOT:
        limb.set_joint_position_speed(0.3)
        """
        # MOCK: Just store the speed setting
        self._speed = speed
        print(f"[MOCK] Joint speed set to {speed}")
    
    def endpoint_pose(self) -> Dict[str, any]:
        """
        Get current end-effector pose.
        
        REPLACE WITH REAL ROBOT:
        This will return actual endpoint position and orientation.
        
        Returns:
            Dict with 'position' (x,y,z) and 'orientation' (quaternion)
        """
        # MOCK: Calculate simulated endpoint pose from joint angles
        # In reality, this uses forward kinematics
        mock_position = np.array([0.5, 0.2, 0.3])  # x, y, z in meters
        mock_orientation = np.array([0.0, 1.0, 0.0, 0.0])  # quaternion (x,y,z,w)
        
        return {
            'position': mock_position,
            'orientation': mock_orientation
        }
    
    def endpoint_velocity(self) -> Dict[str, any]:
        """
        Get current end-effector velocity.
        
        REPLACE WITH REAL ROBOT:
        Returns actual endpoint linear and angular velocities.
        """
        # MOCK: Return zero velocity
        return {
            'linear': np.array([0.0, 0.0, 0.0]),
            'angular': np.array([0.0, 0.0, 0.0])
        }
    
    def joint_efforts(self) -> Dict[str, float]:
        """
        Get current joint torques/efforts.
        
        REPLACE WITH REAL ROBOT:
        Returns actual torque readings from joint torque sensors.
        """
        # MOCK: Return simulated joint efforts
        return {name: 0.0 for name in self._joint_names}


class Camera:
    """
    Mock implementation of camera interface for Sawyer head camera.
    
    REPLACE WITH REAL ROBOT:
    Use intera_interface.Cameras or ROS image topics
    """
    
    def __init__(self, camera_name: str = "head_camera"):
        self.name = camera_name
        self.resolution = (1280, 800)  # Sawyer head camera resolution
        
    def get_frame(self) -> np.ndarray:
        """
        Capture a frame from the camera.
        
        REPLACE WITH REAL ROBOT:
        from intera_interface import Cameras
        cameras = Cameras()
        cameras.start_streaming('head_camera')
        img = cameras.get_image('head_camera')
        
        Returns:
            np.ndarray: Image array (H, W, 3) in RGB format
        """
        # MOCK: Return blank image
        print(f"[MOCK] Capturing frame from {self.name}")
        return np.zeros((*self.resolution[::-1], 3), dtype=np.uint8)
    
    def get_parameters(self) -> Dict:
        """
        Get camera intrinsic parameters.
        
        REPLACE WITH REAL ROBOT:
        Get actual camera calibration parameters.
        """
        # MOCK: Return approximate Sawyer camera parameters
        return {
            'width': 1280,
            'height': 800,
            'fx': 900,  # focal length x
            'fy': 900,  # focal length y
            'cx': 640,  # principal point x
            'cy': 400   # principal point y
        }


class Gripper:
    """
    Mock implementation of gripper interface.
    
    REPLACE WITH REAL ROBOT:
    from intera_interface import Gripper
    """
    
    def __init__(self, gripper_name: str = "right_gripper"):
        self.name = gripper_name
        self._position = 100.0  # Fully open (0-100 scale)
        
    def open(self):
        """
        Open the gripper.
        
        REPLACE WITH REAL ROBOT:
        gripper = Gripper('right_gripper')
        gripper.open()
        """
        print(f"[MOCK] Opening gripper")
        self._position = 100.0
        time.sleep(0.5)
        
    def close(self):
        """
        Close the gripper.
        
        REPLACE WITH REAL ROBOT:
        gripper.close()
        """
        print(f"[MOCK] Closing gripper")
        self._position = 0.0
        time.sleep(0.5)
        
    def set_position(self, position: float):
        """
        Set gripper to specific position.
        
        Args:
            position: 0 (closed) to 100 (open)
            
        REPLACE WITH REAL ROBOT:
        gripper.set_position(50.0)
        """
        print(f"[MOCK] Setting gripper position to {position}")
        self._position = position
        time.sleep(0.3)
        
    def get_position(self) -> float:
        """
        Get current gripper position.
        
        REPLACE WITH REAL ROBOT:
        Returns actual gripper encoder position.
        """
        return self._position
    
    def is_gripping(self) -> bool:
        """
        Check if gripper is holding something.
        
        REPLACE WITH REAL ROBOT:
        Uses force/position feedback to detect grip.
        """
        # MOCK: Simulate gripping if partially closed
        return self._position < 50.0


class SawyerRobot:
    """
    High-level interface for Sawyer robot control.
    Combines limb, camera, and gripper interfaces.
    
    This is your main interface for robot control.
    """
    
    def __init__(self, limb_name: str = "right", mock: bool = True):
        """
        Initialize Sawyer robot interface.
        
        Args:
            limb_name: Which arm to control ('right' for Sawyer)
            mock: If True, use mock simulation. If False, use real robot.
        """
        self.mock = mock
        self.limb = Limb(limb_name)
        self.gripper = Gripper(f"{limb_name}_gripper")
        self.camera = Camera("head_camera")
        
        # Workspace bounds (safety limits in meters)
        self.workspace_bounds = {
            'x': [0.3, 0.9],   # Forward/back
            'y': [-0.5, 0.5],  # Left/right
            'z': [-0.1, 0.5]   # Up/down from table
        }
        
        if not mock:
            # WHEN USING REAL ROBOT: Add initialization code here
            # import rospy
            # rospy.init_node('sawyer_control')
            # from intera_interface import RobotEnable
            # self._rs = RobotEnable()
            # self._rs.enable()
            pass
        
        print(f"[{'MOCK' if mock else 'REAL'}] Sawyer robot initialized")
    
    def move_to_cartesian(self, position: List[float], orientation: Optional[List[float]] = None):
        """
        Move end-effector to cartesian position.
        
        PLACEHOLDER FOR REAL ROBOT:
        Implement inverse kinematics solution here.
        You'll need to:
        1. Use IK solver to convert position → joint angles
        2. Call self.limb.move_to_joint_positions()
        
        For now, this is a high-level placeholder.
        
        Args:
            position: [x, y, z] in meters
            orientation: [x, y, z, w] quaternion (optional)
        """
        print(f"[MOCK] Moving to cartesian position: {position}")
        
        # TODO: When you have the real robot, implement IK here
        # from intera_interface import get_kinematics
        # kin = get_kinematics()
        # joint_angles = kin.inverse_kinematics(position, orientation)
        # self.limb.move_to_joint_positions(joint_angles)
        
        # MOCK: Just simulate the movement
        time.sleep(0.5)
        print(f"[MOCK] Reached target position")
    
    def get_endpoint_state(self) -> Dict:
        """
        Get full endpoint state (position, velocity, force).
        
        Returns:
            Dict containing pose, velocity, and effort information
        """
        pose = self.limb.endpoint_pose()
        velocity = self.limb.endpoint_velocity()
        efforts = self.limb.joint_efforts()
        
        return {
            'pose': pose,
            'velocity': velocity,
            'efforts': efforts,
            'timestamp': time.time()
        }
    
    def capture_scene(self) -> Dict:
        """
        Capture complete scene data (image + robot state).
        
        PLACEHOLDER FOR SLINKY EXPERIMENTS:
        This is where you'll capture data for your experiments.
        Add any additional sensors or measurements needed.
        
        Returns:
            Dict with image, robot state, and metadata
        """
        return {
            'image': self.camera.get_frame(),
            'robot_state': self.get_endpoint_state(),
            'gripper_position': self.gripper.get_position(),
            'timestamp': time.time()
        }
    
    def is_position_safe(self, position: List[float]) -> bool:
        """
        Check if a position is within safe workspace bounds.
        
        MODIFY FOR YOUR SETUP:
        Adjust workspace_bounds based on your table/environment setup.
        """
        x, y, z = position
        return (
            self.workspace_bounds['x'][0] <= x <= self.workspace_bounds['x'][1] and
            self.workspace_bounds['y'][0] <= y <= self.workspace_bounds['y'][1] and
            self.workspace_bounds['z'][0] <= z <= self.workspace_bounds['z'][1]
        )
    
    def home_position(self):
        """
        Move to safe home position.
        
        CUSTOMIZE FOR YOUR SETUP:
        Define a safe home position for your experiments.
        """
        # MOCK: Simulate moving to home
        print("[MOCK] Moving to home position")
        home_joints = {name: 0.0 for name in self.limb.joint_names()}
        self.limb.move_to_joint_positions(home_joints)
    
    def emergency_stop(self):
        """
        Emergency stop - halt all motion.
        
        CRITICAL FOR REAL ROBOT:
        Implement proper e-stop behavior.
        """
        if not self.mock:
            # REAL ROBOT: Implement e-stop
            # self._rs.disable()
            pass
        print("[EMERGENCY STOP] Robot stopped")


# ============================================================================
# HELPER FUNCTIONS FOR COMMON OPERATIONS
# ============================================================================

def create_trajectory(start: List[float], end: List[float], num_points: int = 10) -> List[List[float]]:
    """
    Create a linear trajectory between two points.
    
    PLACEHOLDER:
    You might want more sophisticated trajectories (splines, etc.)
    for smooth Slinky manipulation.
    
    Args:
        start: Starting position [x, y, z]
        end: Ending position [x, y, z]
        num_points: Number of waypoints
        
    Returns:
        List of waypoint positions
    """
    start = np.array(start)
    end = np.array(end)
    
    # Linear interpolation
    trajectory = []
    for i in range(num_points):
        alpha = i / (num_points - 1)
        point = start + alpha * (end - start)
        trajectory.append(point.tolist())
    
    return trajectory


def save_robot_data(data: Dict, filepath: str):
    """
    Save robot sensor data to file.
    
    CUSTOMIZE:
    Add any additional data formats needed for your analysis.
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            serializable_data[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_data[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        else:
            serializable_data[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"[MOCK] Saved robot data to {filepath}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Mock Sawyer Robot Interface - Test")
    print("="*80)
    
    # Initialize robot
    robot = SawyerRobot(mock=True)
    
    # Test basic movements
    print("\n1. Testing joint movement...")
    test_joints = {name: 0.1 for name in robot.limb.joint_names()}
    robot.limb.move_to_joint_positions(test_joints)
    
    print("\n2. Testing cartesian movement...")
    robot.move_to_cartesian([0.6, 0.0, 0.2])
    
    print("\n3. Testing gripper...")
    robot.gripper.open()
    robot.gripper.close()
    robot.gripper.set_position(50.0)
    
    print("\n4. Testing camera...")
    frame = robot.camera.get_frame()
    print(f"   Captured frame: {frame.shape}")
    
    print("\n5. Testing scene capture...")
    scene_data = robot.capture_scene()
    print(f"   Scene data keys: {scene_data.keys()}")
    
    print("\n6. Testing safety check...")
    safe = robot.is_position_safe([0.6, 0.0, 0.2])
    print(f"   Position safe: {safe}")
    
    print("\n7. Testing home position...")
    robot.home_position()
    
    print("\n" + "="*80)
    print("Mock interface test complete!")
    print("="*80)
