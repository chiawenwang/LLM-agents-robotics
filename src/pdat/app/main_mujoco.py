import sys
sys.path.append("/home/sciarm/piper_robot/LLM-agents-robotics/src/")

from pdat.env.piper_mujoco_interface import PiperMujocoEnv, PiperMujocoConfig





env = PiperMujocoEnv(PiperMujocoConfig(
    joint_state="/joint_states",
    # action_topic="/piper/cmd_joint",
    # reset_service="/reset_sim",
))

# obs = env.reset()
print("reset obs:", env._wait_for_obs())

# # send a dummy action
# action = {"names": obs["joint_names"], "positions": obs["joint_pos"]}
# obs2 = env.step(action)
# print("step obs:", obs2)