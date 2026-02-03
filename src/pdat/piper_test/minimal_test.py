"""
Minimal Test Script - Quick Robot Connection Check
Run this first to make sure everything is connected properly.
"""

import time
from robot_api import RobotAPI

print("="*60)
print("MINIMAL ROBOT CONNECTION TEST")
print("="*60)

# Configuration
CAN_NAME = "can0"  # Change to None for simulation mode

try:
    # Step 1: Connect
    print("\n[1/5] Connecting to robot...")
    robot = RobotAPI(backend="piper", can_name=CAN_NAME)
    print("✓ Connected!")
    
    # Step 2: Reset
    print("\n[2/5] Resetting robot...")
    robot.reset()
    time.sleep(1)
    print("✓ Reset complete!")
    
    # Step 3: Read state
    print("\n[3/5] Reading current state...")
    state = robot.get_current_state()
    print(f"✓ State: {state}")
    
    # Step 4: Set a state
    print("\n[4/5] Setting initial state...")
    robot.set_initial_state({"position": 0.05, "velocity": 0.0})
    time.sleep(2)
    print("✓ State set!")
    
    # Step 5: Read final state
    print("\n[5/5] Reading final state...")
    final_state = robot.get_current_state()
    print(f"✓ Final state: {final_state}")
    
    # Success!
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - ROBOT IS WORKING!")
    print("="*60)
    
    # Clean up
    robot.close()
    
except Exception as e:
    print("\n" + "="*60)
    print("❌ TEST FAILED")
    print("="*60)
    print(f"Error: {e}")
    print("\nTroubleshooting:")
    print("1. Is Piper simulation running?")
    print("2. Is CAN interface up? (ip link show can0)")
    print("3. Try: sudo ip link set can0 type can bitrate 1000000")
    print("        sudo ip link set can0 up")
    import traceback
    traceback.print_exc()