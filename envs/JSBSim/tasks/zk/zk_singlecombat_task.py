import numpy as np
import torch

from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.reward_functions.zk.zk_altitude_reward import ZKAltitudeReward
from envs.JSBSim.reward_functions.zk.zk_event_driven_reward import ZKEventDrivenReward
from envs.JSBSim.tasks import HierarchicalSingleCombatTask, SingleCombatShootMissileTask, SingleCombatTask

from gymnasium import spaces

from envs.JSBSim.termination_conditions.zk.zk_safe_return import ZKSafeReturn
from envs.JSBSim.termination_conditions.zk.zk_timeout import ZKTimeout
from envs.JSBSim.utils.utils import get_root_dir, LLA2NEU, get_AO_TA_R


class ZKHierarchicalSingleCombatShootTask(SingleCombatTask):
    def __init__(self, config: str):
        super().__init__(config)
        self.lowlevel_policy = BaselineActor()
        self.lowlevel_policy.load_state_dict(
            torch.load(get_root_dir() + '/model/baseline_model.pt', map_location=torch.device('cpu')))
        self.lowlevel_policy.eval()
        self.norm_delta_altitude = np.array([0.1, 0, -0.1])
        self.norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])
        self.norm_delta_velocity = np.array([0.05, 0, -0.05])


        self.reward_functions = [
            ZKAltitudeReward(self.config),
            ZKEventDrivenReward(self.config)
        ]
        self.termination_conditions = [
            ZKSafeReturn(self.config),
            ZKTimeout(self.config),
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(21,))

    def load_action_space(self):
        # altitude control + heading control + velocity control + shoot control
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([3, 5, 3]), spaces.Discrete(2)])

    def get_obs(self, env, agent_id):
        """
               Convert simulation states into the format of observation_space

               ------
               Returns: (np.ndarray)
               - ego info
                   - [0] ego altitude           (unit: 5km)
                   - [1] ego_roll_sin
                   - [2] ego_roll_cos
                   - [3] ego_pitch_sin
                   - [4] ego_pitch_cos
                   - [5] ego v_body_x           (unit: mh)
                   - [6] ego v_body_y           (unit: mh)
                   - [7] ego v_body_z           (unit: mh)
                   - [8] ego_vc                 (unit: mh)
               - relative enm info
                   - [9] delta_v_body_x         (unit: mh)
                   - [10] delta_altitude        (unit: km)
                   - [11] ego_AO                (unit: rad) [0, pi]
                   - [12] ego_TA                (unit: rad) [0, pi]
                   - [13] relative distance     (unit: 10km)
                   - [14] side_flag             1 or 0 or -1
               - relative missile info
                   - [15] delta_v_body_x
                   - [16] delta altitude
                   - [17] ego_AO
                   - [18] ego_TA
                   - [19] relative distance
                   - [20] side flag
               """
        norm_obs = np.zeros(21)
        # (1) ego info normalization
        agent = env.agents[agent_id]
        agent_feature = np.hstack([agent.get_position(), agent.get_velocity()])
        geodetic = agent.get_geodetic()
        position = agent.get_position()
        velocity = agent.get_velocity()
        rpy = agent.get_rpy()

        norm_obs[0] = geodetic[2] / 5000  # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(rpy[0])  # 1. ego_roll_sin
        norm_obs[2] = np.cos(rpy[0])  # 2. ego_roll_cos
        norm_obs[3] = np.sin(rpy[1])  # 3. ego_pitch_sin
        norm_obs[4] = np.cos(rpy[1])  # 4. ego_pitch_cos
        norm_obs[5] = agent.get("velocities/u-fps") / 1116.44  # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = agent.get("velocities/v-fps") / 1116.44  # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = agent.get("velocities/w-fps") / 1116.44  # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = agent.get("velocities/ve-fps") / 1116.44 # 8. ego vc   (unit: mh)(unit: 5G)
        # (2) relative inof w.r.t partner+enemies state
        offset = 8
        sim = env.agents[agent_id].enemies[0]
        sim_geodetic = sim.get_geodetic()
        # cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
        # feature = np.array([*cur_ned, *(state[6:9])])
        sim_feature = np.hstack([sim.get_position(), sim.get_velocity()])
        AO, TA, R, side_flag = get_AO_TA_R(agent_feature, sim_feature, return_side=True)
        # print("距离R:{}".format(R))
        norm_obs[offset + 1] = (sim.get("velocities/u-fps") - agent.get("velocities/u-fps")) / 1116.44
        norm_obs[offset + 2] = (sim_geodetic[2] - geodetic[2]) / 1000
        norm_obs[offset + 3] = AO
        norm_obs[offset + 4] = TA
        norm_obs[offset + 5] = R / 10000
        norm_obs[offset + 6] = side_flag
        offset += 6
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        # (3) missile info TODO: multiple missile and parnter's missile?
        missile_sim = env.agents[agent_id].check_missile_warning()  #
        if missile_sim is not None:
            missile_sim_geodetic = missile_sim.get_geodetic()
            missile_feature = np.hstack([missile_sim.get_position(), missile_sim.get_velocity()])
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(agent_feature, missile_feature, return_side=True)
            norm_obs[offset + 1] = (missile_sim.get("Speed") - agent.get("velocities/u-fps")) / 1116.44
            norm_obs[offset + 2] = (missile_sim_geodetic[2] - geodetic[2]) / 1000
            norm_obs[offset + 3] = ego_AO
            norm_obs[offset + 4] = ego_TA
            norm_obs[offset + 5] = R / 10000
            norm_obs[offset + 6] = side_flag
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        action = action.astype(np.int32)
        shoot = action[3] > 0 if 1 else 0
        """Convert high-level action into low-level action.
        """
        # generate low-level input_obs
        raw_obs = self.get_obs(env, agent_id)
        input_obs = np.zeros(12)
        # print("action:{}".format(action))
        # (1) delta altitude/heading/velocity
        input_obs[0] = self.norm_delta_altitude[action[0]]
        input_obs[1] = self.norm_delta_heading[action[1]]
        input_obs[2] = self.norm_delta_velocity[action[2]]
        # (2) ego info
        input_obs[3:12] = raw_obs[:9]
        input_obs = np.expand_dims(input_obs, axis=0)
        # output low-level action
        _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
        action = _action.detach().cpu().numpy().squeeze(0)
        self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
        # normalize low-level action
        norm_act = np.zeros(5)
        norm_act[0] = action[0] / 20 - 1.
        norm_act[1] = action[1] / 20 - 1.
        norm_act[2] = action[2] / 20 - 1.
        norm_act[3] = action[3] / 58 + 0.4
        norm_act[4] = shoot
        return norm_act

    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return super().reset(env)

    def step(self, env):
        return