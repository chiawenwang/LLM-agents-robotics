"""
Robot API - Abstraction layer for robot control
Provides a unified interface for the Scientist agent to interact with physical robots.
Currently supports: Piper robotic arm
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import time


class RobotAPI:
    """
    High-level API for robot control, designed to bridge scientific state representations
    with low-level robot commands.
    
    Attributes:
        backend (str): Robot backend type ('piper', etc.)
        initial_state (dict): Initial scientific state (position, velocity)
        current_state (dict): Current scientific state
        piper_interface: Piper SDK interface instance (if using Piper backend)
    """
    
    def __init__(self, backend: str, can_name: Optional[str] = None, **kwargs):
        """
        Initialize the robot API with specified backend.
        
        Args:
            backend: Robot type ('piper')
            can_name: CAN interface name (e.g., 'can0'). None for simulation mode.
            **kwargs: Additional backend-specific arguments
        """
        self.backend = backend
        self.can_name = can_name
        self.initial_state = None
        self.current_state = None
        self.piper_interface = None
        self.is_simulation = (can_name is None)
        self.kwargs = kwargs
        
        # State mapping parameters (to be tuned for specific DLO experiments)
        self.state_to_joint_mapping = {
            'position_scale': 1.0,  # Scale factor for position -> joint angle
            'workspace_offset': 0.0,  # Offset in workspace
            'controlled_joints': [0, 1, 2, 3, 4, 5]  # Which joints to control
        }

        if self.backend == "piper":
            print(f"[RobotAPI] Initializing with Piper backend (Simulation: {self.is_simulation})")
            self._initialize_piper()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _initialize_piper(self):
        """Initialize the Piper SDK interface."""
        try:
            # Import Piper wrapper
            if not self.is_simulation:
                from piper_wrapper import PiperWrapper
                self.piper_interface = PiperWrapper(
                    can_name=self.can_name,
                    **self.kwargs
                )
                
                # Enable the robot
                self.piper_interface.enable()
                print("[RobotAPI] Piper robot enabled and ready")
                
                # Read initial robot state
                self._update_current_state_from_robot()
            else:
                print("[RobotAPI] Running in simulation mode - no physical robot connected")
                # Initialize simulated state
                self.current_state = {
                    'position': 0.0,
                    'velocity': 0.0,
                    'joint_angles': [0.0] * 6,
                    'end_effector_pose': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'timestamp': time.time()
                }
                
        except ImportError as e:
            print(f"[RobotAPI] Warning: Could not import Piper SDK: {e}")
            print("[RobotAPI] Falling back to simulation mode")
            self.is_simulation = True
            self.current_state = {
                'position': 0.0,
                'velocity': 0.0,
                'joint_angles': [0.0] * 6,
                'end_effector_pose': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'timestamp': time.time()
            }

    def reset(self):
        """
        Reset robot to baseline/home state.
        For Piper: uses piper_ctrl_reset.py and piper_ctrl_go_zero.py functionality
        """
        print("[RobotAPI] Resetting robot to baseline state")

        if self.backend == "piper":
            if not self.is_simulation and self.piper_interface is not None:
                try:
                    # Stop any ongoing motion
                    self.piper_interface.stop()
                    time.sleep(0.5)
                    
                    # Reset the robot
                    self.piper_interface.reset()
                    time.sleep(1.0)
                    
                    # Re-enable after reset (as per SDK documentation)
                    self.piper_interface.enable()
                    self.piper_interface.enable()  # Enable twice as recommended
                    time.sleep(0.5)
                    
                    # Go to zero position
                    self.piper_interface.go_zero()
                    time.sleep(2.0)
                    
                    print("[RobotAPI] Piper robot reset complete")
                    
                except Exception as e:
                    print(f"[RobotAPI] Error during reset: {e}")
                    raise RuntimeError(f"Failed to reset Piper robot: {e}")
            else:
                # Simulation mode reset
                print("[RobotAPI] Simulation reset - returning to zero state")
                self.current_state = {
                    'position': 0.0,
                    'velocity': 0.0,
                    'joint_angles': [0.0] * 6,
                    'end_effector_pose': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'timestamp': time.time()
                }
            
            # Clear initial state
            self.initial_state = None
            
        else:
            raise RuntimeError("Reset called with unknown backend")

    def set_initial_state(self, state: Dict):
        """
        Set the initial scientific state and configure robot accordingly.
        
        Args:
            state: Dictionary with 'position' and 'velocity' keys (scientific coordinates)
        
        For DLO manipulation, this might represent:
        - position: grasp position along the DLO
        - velocity: initial velocity of manipulation
        """
        if not isinstance(state, dict):
            raise TypeError("Initial state must be a dictionary")

        if "position" not in state or "velocity" not in state:
            raise ValueError("State must include 'position' and 'velocity'")

        print(f"[RobotAPI] Setting initial state: {state}")
        self.initial_state = state.copy()

        if self.backend == "piper":
            # Map scientific state to robot configuration
            joint_config = self._state_to_joint_configuration(state)
            
            if not self.is_simulation and self.piper_interface is not None:
                try:
                    # Move to initial configuration
                    self.piper_interface.move_joints(
                        joint_angles=joint_config['joint_angles'],
                        speed=0.3  # Moderate speed for initial positioning
                    )
                    
                    # Wait for motion to complete
                    time.sleep(2.0)
                    
                    # Update current state
                    self._update_current_state_from_robot()
                    
                    print(f"[RobotAPI] Robot moved to initial configuration: {joint_config}")
                    
                except Exception as e:
                    print(f"[RobotAPI] Error setting initial state: {e}")
                    raise RuntimeError(f"Failed to set initial state: {e}")
            else:
                # Simulation mode
                self.current_state = {
                    'position': state['position'],
                    'velocity': state['velocity'],
                    'joint_angles': joint_config['joint_angles'],
                    'end_effector_pose': joint_config.get('end_effector_pose', {'x': 0.0, 'y': 0.0, 'z': 0.0}),
                    'timestamp': time.time()
                }
                print(f"[RobotAPI] Simulation state set to: {self.current_state}")
                
        else:
            raise RuntimeError("set_initial_state called with unknown backend")

    def _state_to_joint_configuration(self, state: Dict) -> Dict:
        """
        Convert scientific state (position, velocity) to robot joint configuration.
        
        This is experiment-specific and will need to be adapted based on:
        - The DLO being manipulated (elastic strip, Slinky, etc.)
        - The manipulation task
        - The workspace setup
        
        Args:
            state: Scientific state dictionary
            
        Returns:
            Dictionary with 'joint_angles' and 'end_effector_pose'
        """
        # Example mapping (to be customized for specific experiments)
        # For a 1D problem, we might map position to a single joint or end-effector coordinate
        
        position = state['position']
        velocity = state['velocity']
        
        # Simple linear mapping (placeholder - adjust for your setup)
        # This assumes position corresponds to a linear motion along one axis
        base_joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Map position to joint space (example: affects first two joints)
        # This is a PLACEHOLDER - you'll need to define the actual mapping
        scale = self.state_to_joint_mapping['position_scale']
        offset = self.state_to_joint_mapping['workspace_offset']
        
        # Example: map position to end-effector y-coordinate
        # and compute inverse kinematics (simplified here)
        joint_angles = base_joint_angles.copy()
        joint_angles[0] = np.clip(position * scale + offset, -np.pi, np.pi)
        
        # Compute approximate end-effector pose (would use forward kinematics in practice)
        end_effector_pose = {
            'x': 0.3,  # Fixed x position (example)
            'y': position * 0.1,  # y varies with position
            'z': 0.2,  # Fixed height (example)
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0
        }
        
        return {
            'joint_angles': joint_angles,
            'end_effector_pose': end_effector_pose
        }

    def _update_current_state_from_robot(self):
        """Read current robot state and update internal representation."""
        if not self.is_simulation and self.piper_interface is not None:
            try:
                # Read joint states
                joint_state = self.piper_interface.read_joint_state()
                
                # Read end-effector pose
                end_pose = self.piper_interface.read_end_pose()
                
                # Convert to scientific state representation
                scientific_state = self._joint_configuration_to_state(
                    joint_angles=joint_state['angles'],
                    joint_velocities=joint_state.get('velocities', [0.0] * 6)
                )
                
                self.current_state = {
                    'position': scientific_state['position'],
                    'velocity': scientific_state['velocity'],
                    'joint_angles': joint_state['angles'],
                    'end_effector_pose': end_pose,
                    'timestamp': time.time()
                }
                
            except Exception as e:
                print(f"[RobotAPI] Error reading robot state: {e}")

    def _joint_configuration_to_state(self, joint_angles: List[float], 
                                     joint_velocities: List[float]) -> Dict:
        """
        Convert robot joint configuration back to scientific state.
        Inverse of _state_to_joint_configuration.
        
        Args:
            joint_angles: Current joint angles
            joint_velocities: Current joint velocities
            
        Returns:
            Scientific state dictionary
        """
        # Inverse mapping (placeholder)
        scale = self.state_to_joint_mapping['position_scale']
        offset = self.state_to_joint_mapping['workspace_offset']
        
        position = (joint_angles[0] - offset) / scale if scale != 0 else 0.0
        velocity = joint_velocities[0] / scale if scale != 0 else 0.0
        
        return {
            'position': position,
            'velocity': velocity
        }

    def get_current_state(self) -> Dict:
        """
        Get the current scientific state of the system.
        
        Returns:
            Dictionary with 'position', 'velocity', and additional robot info
        """
        if not self.is_simulation and self.piper_interface is not None:
            self._update_current_state_from_robot()
        
        return self.current_state.copy() if self.current_state else None

    def apply_control(self, control_input: Dict):
        """
        Apply control input to the robot.
        
        Args:
            control_input: Dictionary specifying the control (e.g., force, position target, etc.)
        """
        if self.backend == "piper":
            if not self.is_simulation and self.piper_interface is not None:
                # Convert control to joint-space command
                # This depends on your control scheme (position, velocity, force, etc.)
                if 'target_position' in control_input:
                    target_state = {
                        'position': control_input['target_position'],
                        'velocity': control_input.get('target_velocity', 0.0)
                    }
                    joint_config = self._state_to_joint_configuration(target_state)
                    
                    self.piper_interface.move_joints(
                        joint_angles=joint_config['joint_angles'],
                        speed=control_input.get('speed', 0.5)
                    )
                    
            else:
                # Simulation: update state based on simple dynamics
                if 'target_position' in control_input:
                    self.current_state['position'] = control_input['target_position']
                    self.current_state['velocity'] = control_input.get('target_velocity', 0.0)
                    self.current_state['timestamp'] = time.time()

    def close(self):
        """Safely close the robot connection."""
        print("[RobotAPI] Closing robot connection")
        
        if self.backend == "piper" and not self.is_simulation:
            if self.piper_interface is not None:
                try:
                    self.piper_interface.disable()
                    print("[RobotAPI] Piper robot disabled")
                except Exception as e:
                    print(f"[RobotAPI] Error during close: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("Testing RobotAPI with Piper backend (Simulation Mode)")
    print("="*60)
    
    # Initialize robot in simulation mode (can_name=None)
    robot = RobotAPI(backend="piper", can_name=None)
    
    # Reset to baseline
    robot.reset()
    
    # Set initial state for 1D DLO experiment
    initial_state = {
        "position": 0.1,  # Initial position along DLO
        "velocity": 0.0   # Starting from rest
    }
    robot.set_initial_state(initial_state)
    
    # Get current state
    current = robot.get_current_state()
    print(f"\nCurrent state: {current}")
    
    # Apply a control input (move to new position)
    control = {
        "target_position": 0.2,
        "target_velocity": 0.05,
        "speed": 0.5
    }
    robot.apply_control(control)
    
    # Check new state
    new_state = robot.get_current_state()
    print(f"New state after control: {new_state}")
    
    # Clean up
    robot.close()
    
    print("\n" + "="*60)
    print("RobotAPI test complete!")
    print("="*60)