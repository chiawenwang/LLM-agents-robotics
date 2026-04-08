# src/pdat/env/piper_mujoco_interface.py
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

import rospy
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty

# Replace these with your actual message/service types if custom:
# from piper_msgs.msg import ...
# from piper_msgs.srv import ...

@dataclass
class PiperMujocoConfig:
    joint_state: str = "/mujoco_joint_states_pub"
    # action_topic: str = "/piper/cmd_joint"     # TODO: replace
    # reset_service: str = "/reset_sim"          # TODO: replace
    control_rate_hz: int = 20
    obs_timeout_s: float = 1.0


class PiperMujocoEnv:
    """
    Thin adapter: provides reset/step/observe by talking to ROS.
    """
    def __init__(self, cfg: PiperMujocoConfig):
        self.cfg = cfg
        self._last_joint_state: Optional[JointState] = None

        if not rospy.core.is_initialized():
            rospy.init_node("pdat_piper_mujoco_env", anonymous=True, disable_signals=True)

        self._js_sub = rospy.Subscriber(self.cfg.joint_state, JointState, self._on_joint_state)

        # # Publisher example (replace msg type!)
        # # self._act_pub = rospy.Publisher(self.cfg.action_topic, YourActionMsg, queue_size=1)
        # self._act_pub = rospy.Publisher(self.cfg.action_topic, JointState, queue_size=1)  # placeholder

        # # Reset service example
        # self._reset_srv = None
        # try:
        #     rospy.wait_for_service(self.cfg.reset_service, timeout=3.0)
        #     self._reset_srv = rospy.ServiceProxy(self.cfg.reset_service, Empty)
        # except Exception:
        #     self._reset_srv = None

        self._rate = rospy.Rate(self.cfg.control_rate_hz)

    def _on_joint_state(self, msg: JointState):
        self._last_joint_state = msg

    def _wait_for_obs(self) -> JointState:
        t0 = time.time()
        while self._last_joint_state is None:
            if time.time() - t0 > self.cfg.obs_timeout_s:
                raise TimeoutError(f"No joint_state received from {self.cfg.joint_state} within {self.cfg.obs_timeout_s} seconds")
            self._rate.sleep()
        return self._last_joint_state

    def reset(self) -> Dict[str, Any]:
        if self._reset_srv is not None:
            self._reset_srv()
        # wait for fresh observation
        js = self._wait_for_obs()
        return self._make_obs(js)

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        action could be dict: {"names":[...], "positions":[...]} or whatever your agents produce
        """
        # TODO: map your action dict -> ROS message your sim expects
        msg = JointState()
        msg.name = action.get("names", [])
        msg.position = action.get("positions", [])
        # self._act_pub.publish(msg)

        self._rate.sleep()
        js = self._wait_for_obs()
        return self._make_obs(js)

    def _make_obs(self, js: JointState) -> Dict[str, Any]:
        return {
            "joint_names": list(js.name),
            "joint_pos": list(js.position),
            "joint_vel": list(js.velocity),
            "joint_eff": list(js.effort),
            "stamp": js.header.stamp.to_sec() if js.header.stamp else None,
        }