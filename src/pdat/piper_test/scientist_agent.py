"""
Simple Scientist Agent - For Testing Robot Connection
Minimal code to verify robot_api and piper_wrapper work correctly.
"""

import time
from robot_api import RobotAPI


class SimpleScientist:
    """
    Minimal scientist agent for testing robot connection and basic movements.
    """
    
    def __init__(self, can_name: str = "can0"):
        """
        Initialize with robot connection.
        
        Args:
            can_name: CAN interface name ("can0" for real robot, None for simulation)
        """
        print(f"[SimpleScientist] Initializing...")
        
        # Create robot connection
        self.robot = RobotAPI(backend="piper", can_name=can_name)
        
        print(f"[SimpleScientist] Connected to robot")
    
    def test_connection(self):
        """Test if we can communicate with the robot."""
        print("\n" + "="*60)
        print("TEST 1: Connection")
        print("="*60)
        
        try:
            state = self.robot.get_current_state()
            print(f"✓ Current state: {state}")
            return True
        except Exception as e:
            print(f"✗ Failed to get state: {e}")
            return False
    
    def test_reset(self):
        """Test robot reset."""
        print("\n" + "="*60)
        print("TEST 2: Reset")
        print("="*60)
        
        try:
            self.robot.reset()
            time.sleep(1)
            print("✓ Reset successful")
            return True
        except Exception as e:
            print(f"✗ Reset failed: {e}")
            return False
    
    def test_set_state(self):
        """Test setting initial state."""
        print("\n" + "="*60)
        print("TEST 3: Set State")
        print("="*60)
        
        try:
            # Set a simple state
            test_state = {"position": 0.1, "velocity": 0.0}
            self.robot.set_initial_state(test_state)
            time.sleep(1)
            
            # Read back the state
            current = self.robot.get_current_state()
            print(f"✓ Set state to: {test_state}")
            print(f"  Current state: {current}")
            return True
        except Exception as e:
            print(f"✗ Set state failed: {e}")
            return False
    
    def test_small_movement(self):
        """Test a small safe movement."""
        print("\n" + "="*60)
        print("TEST 4: Small Movement")
        print("="*60)
        
        try:
            # Read initial position
            initial = self.robot.get_current_state()
            print(f"Initial state: {initial}")
            
            # Move slightly
            new_state = {"position": 0.05, "velocity": 0.0}
            self.robot.set_initial_state(new_state)
            time.sleep(2)
            
            # Read new position
            final = self.robot.get_current_state()
            print(f"Final state: {final}")
            print("✓ Movement successful")
            return True
        except Exception as e:
            print(f"✗ Movement failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all basic tests."""
        print("\n" + "="*60)
        print("RUNNING ALL TESTS")
        print("="*60)
        
        results = {
            "Connection": self.test_connection(),
            "Reset": self.test_reset(),
            "Set State": self.test_set_state(),
            "Movement": self.test_small_movement(),
        }
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{test_name:20s}: {status}")
        
        all_passed = all(results.values())
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! Robot connection is working!")
        else:
            print("\n⚠️  Some tests failed. Check errors above.")
        
        return all_passed
    
    def close(self):
        """Clean up and close connection."""
        print("\n[SimpleScientist] Closing connection...")
        self.robot.close()
        print("[SimpleScientist] Done!")


# Main test script
if __name__ == "__main__":
    print("="*60)
    print("Simple Scientist Agent - Robot Connection Test")
    print("="*60)
    
    # For simulation: can_name=None
    # For real robot: can_name="can0"
    CAN_NAME = "can0"  # Change to None for simulation
    
    try:
        # Create scientist
        scientist = SimpleScientist(can_name=CAN_NAME)
        
        # Run all tests
        scientist.run_all_tests()
        
        # Clean up
        scientist.close()
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure:")
        print("1. CAN interface is set up (sudo ip link set can0 up)")
        print("2. Piper simulation is running")
        print("3. piper_wrapper.py is in the same directory")