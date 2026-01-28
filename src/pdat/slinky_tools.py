"""
Slinky Manipulation Tools for Agent Framework

High-level functions for Slinky experiments that agents can call.
These wrap the low-level Sawyer API into experiment-specific operations.

CUSTOMIZE THESE for your specific Slinky experiments!
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from sawyer_interface import SawyerRobot, create_trajectory


class SlinkyExperiment:
    """
    High-level interface for Slinky manipulation experiments.
    This is what your Scientist agent will use.
    """
    
    def __init__(self, robot: SawyerRobot):
        """
        Initialize Slinky experiment interface.
        
        Args:
            robot: SawyerRobot instance
        """
        self.robot = robot
        self.data_buffer = []  # Store experiment data
        
        # CUSTOMIZE: Slinky-specific parameters
        self.slinky_length_relaxed = 0.15  # meters (relaxed length)
        self.slinky_diameter = 0.08  # meters
        self.slinky_mass = 0.05  # kg (approximate)
        
    def setup_slinky(self, attachment_height: float = 0.3):
        """
        Setup Slinky for experiment.
        
        PLACEHOLDER FOR REAL SETUP:
        This is where you'll:
        1. Attach Slinky to fixture/gripper
        2. Verify attachment is secure
        3. Initialize vision tracking
        
        Args:
            attachment_height: Height to hold Slinky (meters)
        """
        print("[SLINKY] Setting up Slinky experiment...")
        
        # MOCK: Move to attachment point
        attachment_point = [0.6, 0.0, attachment_height]
        self.robot.move_to_cartesian(attachment_point)
        
        # MOCK: Close gripper to hold Slinky
        self.robot.gripper.close()
        
        # TODO REAL ROBOT: Add vision verification
        # - Check that Slinky is properly attached
        # - Verify no obstructions
        # - Calibrate camera for tracking
        
        time.sleep(0.5)
        print("[SLINKY] Setup complete")
    
    def stretch_slinky(self, stretch_ratio: float, duration: float = 2.0) -> Dict:
        """
        Stretch Slinky to specified ratio.
        
        CUSTOMIZE FOR YOUR EXPERIMENTS:
        This is a key manipulation primitive for Slinky.
        You'll measure forces, track coil positions, etc.
        
        Args:
            stretch_ratio: How much to stretch (1.0 = relaxed, 2.0 = 2x length)
            duration: Time to complete stretch (seconds)
            
        Returns:
            Dict with sensor data during stretch
        """
        print(f"[SLINKY] Stretching to {stretch_ratio}x length...")
        
        # Calculate target position
        current_pose = self.robot.limb.endpoint_pose()
        current_z = current_pose['position'][2]
        
        # Stretch vertically (simplest case)
        stretch_distance = self.slinky_length_relaxed * (stretch_ratio - 1.0)
        target_z = current_z - stretch_distance  # Move down to stretch
        
        target_position = [
            current_pose['position'][0],
            current_pose['position'][1],
            target_z
        ]
        
        # SAFETY CHECK
        if not self.robot.is_position_safe(target_position):
            print("[ERROR] Target position is unsafe!")
            return {'status': 'error', 'reason': 'unsafe_position'}
        
        # Execute stretch movement
        # TODO REAL ROBOT: Add force control here
        # - Monitor gripper force
        # - Stop if force exceeds threshold
        # - Use impedance control for smooth stretch
        
        self.robot.move_to_cartesian(target_position)
        
        # Collect data during stretch
        data = self._collect_stretch_data(stretch_ratio, duration)
        
        print(f"[SLINKY] Stretch complete")
        return data
    
    def oscillate_slinky(self, amplitude: float, frequency: float, duration: float = 5.0) -> Dict:
        """
        Create oscillations in the Slinky.
        
        PLACEHOLDER FOR REAL EXPERIMENTS:
        This is for studying wave propagation in the Slinky.
        
        Args:
            amplitude: Oscillation amplitude (meters)
            frequency: Oscillation frequency (Hz)
            duration: Total oscillation time (seconds)
            
        Returns:
            Dict with oscillation data
        """
        print(f"[SLINKY] Oscillating at {frequency} Hz, amplitude {amplitude} m...")
        
        # Get current position
        current_pose = self.robot.limb.endpoint_pose()
        center = current_pose['position']
        
        # Calculate number of oscillations
        num_cycles = int(frequency * duration)
        samples_per_cycle = 20
        total_samples = num_cycles * samples_per_cycle
        
        oscillation_data = []
        
        # Execute oscillation
        for i in range(total_samples):
            t = i / (samples_per_cycle * frequency)
            offset = amplitude * np.sin(2 * np.pi * frequency * t)
            
            target = [center[0], center[1], center[2] + offset]
            
            # TODO REAL ROBOT: This needs to be much faster
            # Use velocity control instead of position control
            # self.robot.limb.set_joint_velocities(velocities)
            
            self.robot.move_to_cartesian(target)
            
            # Collect data point
            state = self.robot.get_endpoint_state()
            oscillation_data.append({
                'time': t,
                'position': target,
                'actual_position': state['pose']['position'],
                'velocity': state['velocity']
            })
        
        print("[SLINKY] Oscillation complete")
        return {
            'status': 'success',
            'oscillation_data': oscillation_data,
            'amplitude': amplitude,
            'frequency': frequency,
            'duration': duration
        }
    
    def bend_slinky(self, curvature: float, direction: str = 'horizontal') -> Dict:
        """
        Bend Slinky to create curvature.
        
        PLACEHOLDER FOR REAL EXPERIMENTS:
        For studying bending mechanics and material properties.
        
        Args:
            curvature: Target curvature (1/radius in m^-1)
            direction: 'horizontal' or 'vertical' bending
            
        Returns:
            Dict with bending data
        """
        print(f"[SLINKY] Bending with curvature {curvature}, direction {direction}...")
        
        # TODO REAL ROBOT: Implement bending control
        # This is complex - might need:
        # - Two-arm manipulation, or
        # - Fixtures to hold one end, or
        # - Vision-guided control
        
        # MOCK: Simulate bending movement
        time.sleep(1.0)
        
        data = {
            'status': 'success',
            'curvature': curvature,
            'direction': direction,
            'final_shape': 'mock_data'
        }
        
        print("[SLINKY] Bending complete")
        return data
    
    def release_and_observe(self, observation_time: float = 3.0) -> Dict:
        """
        Release Slinky and observe dynamics.
        
        IMPORTANT FOR YOUR RESEARCH:
        This captures the natural dynamics for model validation.
        
        Args:
            observation_time: How long to observe (seconds)
            
        Returns:
            Dict with video and position tracking data
        """
        print(f"[SLINKY] Releasing and observing for {observation_time}s...")
        
        # Release gripper
        self.robot.gripper.open()
        
        # Capture video and track positions
        observation_data = []
        start_time = time.time()
        
        while time.time() - start_time < observation_time:
            # Capture frame and robot state
            scene = self.robot.capture_scene()
            
            # TODO REAL ROBOT: Add Slinky tracking here
            # - Use computer vision to track coil positions
            # - Extract 3D positions from stereo/depth
            # - Compute velocities and accelerations
            
            observation_data.append({
                'time': time.time() - start_time,
                'image': scene['image'],
                'robot_state': scene['robot_state']
            })
            
            time.sleep(0.033)  # ~30 fps
        
        print("[SLINKY] Observation complete")
        return {
            'status': 'success',
            'observation_data': observation_data,
            'duration': observation_time
        }
    
    def _collect_stretch_data(self, stretch_ratio: float, duration: float) -> Dict:
        """
        Collect sensor data during stretch.
        
        CUSTOMIZE FOR YOUR SENSORS:
        Add all sensor modalities you need:
        - Force/torque at gripper
        - Joint torques
        - Camera images
        - Slinky coil positions (from vision)
        """
        # TODO REAL ROBOT: Replace with actual data collection
        
        # MOCK: Generate synthetic data
        num_samples = int(duration * 30)  # 30 Hz sampling
        
        mock_data = {
            'stretch_ratio': stretch_ratio,
            'duration': duration,
            'num_samples': num_samples,
            'timestamps': np.linspace(0, duration, num_samples).tolist(),
            'positions': [self.robot.limb.endpoint_pose()['position'].tolist()] * num_samples,
            'forces': np.random.randn(num_samples, 3).tolist(),  # Mock force data
            'status': 'success'
        }
        
        return mock_data
    
    def run_parameterized_experiment(self, params: Dict) -> Dict:
        """
        Run a complete experiment based on parameters.
        
        THIS IS WHAT YOUR SCIENTIST AGENT WILL CALL!
        
        The agent designs experiments by setting parameters,
        and this function executes them.
        
        Args:
            params: Dict with experiment parameters
                - experiment_type: 'stretch', 'oscillate', 'bend', etc.
                - specific parameters for that experiment type
                
        Returns:
            Dict with complete experiment results
        """
        print(f"\n{'='*60}")
        print(f"Running Experiment: {params.get('experiment_type', 'unknown')}")
        print(f"{'='*60}\n")
        
        experiment_type = params.get('experiment_type', 'stretch')
        results = {'experiment_params': params}
        
        try:
            # Setup
            self.setup_slinky()
            
            # Execute based on type
            if experiment_type == 'stretch':
                stretch_ratio = params.get('stretch_ratio', 1.5)
                data = self.stretch_slinky(stretch_ratio)
                results['stretch_data'] = data
                
            elif experiment_type == 'oscillate':
                amplitude = params.get('amplitude', 0.05)
                frequency = params.get('frequency', 2.0)
                duration = params.get('duration', 5.0)
                data = self.oscillate_slinky(amplitude, frequency, duration)
                results['oscillation_data'] = data
                
            elif experiment_type == 'bend':
                curvature = params.get('curvature', 0.5)
                direction = params.get('direction', 'horizontal')
                data = self.bend_slinky(curvature, direction)
                results['bending_data'] = data
                
            elif experiment_type == 'multi_step':
                # PLACEHOLDER: Complex multi-step experiments
                # Example: stretch → oscillate → release → observe
                steps = params.get('steps', [])
                step_results = []
                for step in steps:
                    # Execute each step recursively
                    step_result = self.run_parameterized_experiment(step)
                    step_results.append(step_result)
                results['step_results'] = step_results
                
            else:
                results['status'] = 'error'
                results['reason'] = f'Unknown experiment type: {experiment_type}'
                return results
            
            # Final observation
            if params.get('observe_after', False):
                obs_data = self.release_and_observe()
                results['observation_data'] = obs_data
            
            results['status'] = 'success'
            
        except Exception as e:
            print(f"[ERROR] Experiment failed: {e}")
            results['status'] = 'error'
            results['reason'] = str(e)
            
        finally:
            # Always return to safe position
            self.robot.home_position()
        
        print(f"\n{'='*60}")
        print(f"Experiment Complete: {results.get('status', 'unknown')}")
        print(f"{'='*60}\n")
        
        return results


# ============================================================================
# CONVENIENCE FUNCTIONS FOR AGENTS
# ============================================================================

def create_slinky_experiment(mock: bool = True) -> SlinkyExperiment:
    """
    Factory function to create SlinkyExperiment instance.
    
    Args:
        mock: If True, use mock robot. If False, use real robot.
        
    Returns:
        SlinkyExperiment instance ready to use
    """
    robot = SawyerRobot(mock=mock)
    experiment = SlinkyExperiment(robot)
    return experiment


def quick_stretch_test(stretch_ratio: float = 1.5, mock: bool = True) -> Dict:
    """
    Quick test of stretch experiment.
    
    Useful for debugging and testing.
    """
    exp = create_slinky_experiment(mock=mock)
    params = {
        'experiment_type': 'stretch',
        'stretch_ratio': stretch_ratio,
        'observe_after': False
    }
    return exp.run_parameterized_experiment(params)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Slinky Manipulation Tools - Test")
    print("="*80)
    
    # Create experiment interface
    exp = create_slinky_experiment(mock=True)
    
    # Test 1: Simple stretch
    print("\n--- Test 1: Stretch Experiment ---")
    stretch_params = {
        'experiment_type': 'stretch',
        'stretch_ratio': 1.5,
        'observe_after': False
    }
    result1 = exp.run_parameterized_experiment(stretch_params)
    print(f"Result: {result1['status']}")
    
    # Test 2: Oscillation
    print("\n--- Test 2: Oscillation Experiment ---")
    oscillate_params = {
        'experiment_type': 'oscillate',
        'amplitude': 0.05,
        'frequency': 2.0,
        'duration': 3.0,
        'observe_after': False
    }
    result2 = exp.run_parameterized_experiment(oscillate_params)
    print(f"Result: {result2['status']}")
    
    # Test 3: Multi-step experiment
    print("\n--- Test 3: Multi-Step Experiment ---")
    multi_params = {
        'experiment_type': 'multi_step',
        'steps': [
            {'experiment_type': 'stretch', 'stretch_ratio': 1.3},
            {'experiment_type': 'oscillate', 'amplitude': 0.03, 'frequency': 1.5, 'duration': 2.0}
        ],
        'observe_after': True
    }
    result3 = exp.run_parameterized_experiment(multi_params)
    print(f"Result: {result3['status']}")
    
    print("\n" + "="*80)
    print("Slinky tools test complete!")
    print("="*80)
