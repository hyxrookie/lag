import numpy as np
from gymnasium import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, HeadingReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, UnreachHeading


class HeadingTask(BaseTask):
    '''
    Control target heading with discrete action space
    '''
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            HeadingReward(self.config),
            AltitudeReward(self.config),
        ]
        self.termination_conditions = [
            UnreachHeading(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]
        self.sim_freq = getattr(self.config, 'sim_freq', 60)
        dt = 1.0 / self.sim_freq
        self.eta1 = dt / (2.0 + dt)
        self.eta2 = (2.0 - dt) / (2.0 + dt)
        self.filter_states = {}

    def reset(self, env):
        self.filter_states = {}
        return super().reset(env)
    @property
    def num_agents(self):
        return 1

    def load_variables(self):
        self.state_var = [
            c.delta_altitude,  # 0. delta_h   (unit: m)
            c.delta_heading,  # 1. delta_heading  (unit: °)
            c.delta_velocities_u,  # 2. delta_v   (unit: m/s)
            c.position_h_sl_m,  # 3. altitude  (unit: m)
            c.attitude_roll_rad,  # 4. roll      (unit: rad)
            c.attitude_pitch_rad,  # 5. pitch     (unit: rad)
            c.velocities_u_mps,  # 6. v_body_x   (unit: m/s)
            c.velocities_v_mps,  # 7. v_body_y   (unit: m/s)
            c.velocities_w_mps,  # 8. v_body_z   (unit: m/s)
            c.velocities_vc_mps,  # 9. vc        (unit: m/s)
            # 新增角速度参数 (通常单位是 rad/s)
            c.velocities_p_rad_sec,  # 10. p (roll rate)  (unit: rad/s)
            c.velocities_q_rad_sec,  # 11. q (pitch rate) (unit: rad/s)
            c.velocities_r_rad_sec,  # 12. r (yaw rate)   (unit: rad/s)
            c.fcs_left_aileron_pos_norm,  # 13 left aileron position
            c.fcs_right_aileron_pos_norm,  # 14 right aileron position
            c.fcs_elevator_pos_norm,  # 15 elevator position
            c.fcs_rudder_pos_norm,  # 16 rudder position
            c.fcs_throttle_pos_norm,  # 17 throttle position
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(20,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space.

        observation(dim 20):
            0. ego delta altitude      (unit: km)
            1. ego delta heading       (unit rad)
            2. ego delta velocities_u  (unit: mh)
            3. ego_altitude            (unit: 5km)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego v_body_x            (unit: mh)
            9. ego v_body_y            (unit: mh)
            10. ego v_body_z           (unit: mh)
            11. ego_vc                 (unit: mh)
            12.p (roll rate)  (unit: rad/s)
            13. q (pitch rate) (unit: rad/s)
            14. r (yaw rate)   (unit: rad/s)
             15 left aileron position
             16 right aileron position
            17 elevator position
             18 rudder position
            19 throttle position
        """
        obs = np.array(env.agents[agent_id].get_property_values(self.state_var))
        norm_obs = np.zeros(20)
        norm_obs[0] = obs[0] / 1000         # 0. ego delta altitude (unit: 1km)
        norm_obs[1] = obs[1] / 180 * np.pi  # 1. ego delta heading  (unit rad)
        norm_obs[2] = obs[2] / 340          # 2. ego delta velocities_u (unit: mh)
        norm_obs[3] = obs[3] / 5000         # 3. ego_altitude   (unit: 5km)
        norm_obs[4] = np.sin(obs[4])        # 4. ego_roll_sin
        norm_obs[5] = np.cos(obs[4])        # 5. ego_roll_cos
        norm_obs[6] = np.sin(obs[5])        # 6. ego_pitch_sin
        norm_obs[7] = np.cos(obs[5])        # 7. ego_pitch_cos
        norm_obs[8] = obs[6] / 340          # 8. ego_v_north    (unit: mh)
        norm_obs[9] = obs[7] / 340          # 9. ego_v_east     (unit: mh)
        norm_obs[10] = obs[8] / 340         # 10. ego_v_down    (unit: mh)
        norm_obs[11] = obs[9] / 340         # 11. ego_vc        (unit: mh)
        norm_obs[12] = obs[10]           # 12.p (roll rate)  (unit: rad/s)
        norm_obs[13] = obs[11]           # 13. q (pitch rate) (unit: rad/s)
        norm_obs[14] = obs[12]           # 14. r (yaw rate)   (unit: rad/s)
        norm_obs[15] = obs[13]           # 15 left aileron position
        norm_obs[16] = obs[14]           # 16 right aileron position
        norm_obs[17] = obs[15]           # 17 elevator position
        norm_obs[18] = obs[16]           # 18 rudder position
        norm_obs[19] = obs[17]           # 19 throttle position

        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        raw_act = np.zeros(4)
        raw_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.  # Aileron
        raw_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.  # Elevator
        raw_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.  # Rudder
        raw_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4 # Throttle
        if agent_id not in self.filter_states:
            self.filter_states[agent_id] = {
                'last_raw': raw_act.copy(),  # 上一时刻的原始指令 (\bar{a}_{t-1})
                'last_smoothed': raw_act.copy()  # 上一时刻的平滑输出 (a_{t-1})
            }
            # 第一步直接返回，不滤波
            return raw_act

            # 2.2 获取历史状态
        state = self.filter_states[agent_id]
        last_raw = state['last_raw']
        last_smoothed = state['last_smoothed']

        # 2.3 应用论文公式 (7)
        # Smoothed_t = eta1 * (Raw_t + Raw_{t-1}) + eta2 * Smoothed_{t-1}
        # 这是一个双线性变换 (Bilinear Transform) 实现的一阶滤波器
        smoothed_act = (self.eta1 * (raw_act + last_raw) +
                        self.eta2 * last_smoothed)

        # 2.4 更新历史状态
        state['last_raw'] = raw_act
        state['last_smoothed'] = smoothed_act

        # 2.5 返回平滑后的动作
        return smoothed_act
