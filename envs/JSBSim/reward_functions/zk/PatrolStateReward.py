import math
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.core.catalog import Catalog as c


class PatrolStateReward(BaseRewardFunction):
    """
    奖励飞机进入并维持一个理想的巡航状态。
    核心控制变量为等效空速 (EAS)，以保证气动性能的稳定。
    """

    def __init__(self, config):
        super().__init__(config)
        # --- 目标区间定义 (单位: 英尺 ft, 英尺/秒 fps) ---
        # 目标高度：约9500m-10500m
        self.H_MIN = getattr(config, 'H_MIN', 31000.0)  # ft
        self.H_MAX = getattr(config, 'H_MAX', 34500.0)  # ft

        # 目标速度(EAS)：对应 F-16 在该高度层约 0.85-0.95马赫的巡航速度
        self.V_MIN_EAS = getattr(config, 'V_MIN_EAS', 470.0)  # fps
        self.V_MAX_EAS = getattr(config, 'V_MAX_EAS', 525.0)  # fps

        # --- 奖励和惩罚权重 ---
        self.w_alt = getattr(config, 'w_alt', 1.0)
        self.w_vel = getattr(config, 'w_vel', 1.0)
        self.w_stab = getattr(config, 'w_stab', 0.5)  # 飞行稳定性权重

        # --- 奖励值 ---
        self.in_zone_bonus = getattr(config, 'in_zone_bonus', 2.0)

        # --- 惩罚缩放系数 ---
        # 用于将状态误差转化为合适的惩罚值
        self.alt_error_scale = getattr(config, 'alt_error_scale', 1 / 1000)
        self.vel_error_scale = getattr(config, 'vel_error_scale', 1 / 100)

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        # 任务开始前或飞机坠毁时不给奖励
        if not agent.is_alive:
            return 0

        # 如果探测到敌人，此奖励函数不生效
        if any(enm.is_alive for enm in agent.share_detected_enemies):
            return 0

        # --- 1. 获取状态 ---
        altitude_ft = agent.get('position/h-sl-ft')
        velocity_eas_fps = agent.get('velocities/ve-fps')
        roll_rad = agent.get('attitude/roll-rad')
        pitch_rad = agent.get('attitude/pitch-rad')

        # --- 2. 计算各分项奖励 ---

        # 高度奖励
        if self.H_MIN <= altitude_ft <= self.H_MAX:
            R_alt = self.in_zone_bonus
        else:
            alt_error = min(abs(altitude_ft - self.H_MIN), abs(altitude_ft - self.H_MAX))
            R_alt = -((alt_error * self.alt_error_scale) ** 2)

        # 速度奖励 (基于EAS)
        if self.V_MIN_EAS <= velocity_eas_fps <= self.V_MAX_EAS:
            R_vel = self.in_zone_bonus
        else:
            vel_error = min(abs(velocity_eas_fps - self.V_MIN_EAS), abs(velocity_eas_fps - self.V_MAX_EAS))
            R_vel = -((vel_error * self.vel_error_scale) ** 2)

        # 稳定性奖励 (鼓励平飞)
        penalty_roll = math.cos(roll_rad) - 1
        penalty_pitch = math.cos(pitch_rad) - 1  # 不再需要乘以2

        # 给予权重
        w_roll_penalty = 0.5
        w_pitch_penalty = 0.3
        R_stab = w_roll_penalty * penalty_roll + w_pitch_penalty * penalty_pitch

        # --- 3. 组合最终奖励 ---
        reward = (self.w_alt * R_alt +
                  self.w_vel * R_vel +
                  self.w_stab * R_stab)

        return self._process(reward, agent_id)