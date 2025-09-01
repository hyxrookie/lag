import numpy as np
import math
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R, _calculate_tactical_score


class EvasionReward(BaseRewardFunction):
    """
    奖励智能体规避来自最具威胁敌机的攻击。
    - 这是一个综合评估敌方威胁的函数，与 TacticalAdvantageReward 的进攻哲学相对应。
    - 综合评估四个维度来量化“威胁分数”：
      1. 角度威胁 (Angle): 敌机是否对准我机。
      2. 距离威胁 (Range): 我机是否处于敌机最佳射程。
      3. 高度威胁 (Altitude): 敌机是否拥有高度优势。
      4. 速度威胁 (Velocity): 敌机是否拥有速度优势。
    - 最终奖励是基于对“最大威胁源”的“综合威胁分数”的负值。
    """

    def __init__(self, config):
        super().__init__(config)
        # --- 几何威胁参数 (从敌方视角) ---
        self.min_attack_range = getattr(self.config, 'min_attack_range', 4000.0)
        self.max_attack_range = getattr(self.config, 'max_attack_range', 14000.0)
        # 衰减因子决定了在范围外奖励下降的速度，值越小，下降越平缓
        self.range_decay_factor = getattr(self.config, 'range_decay_factor', 0.0005)
        # 敌方武器的最大视角
        self.max_ao_rad = math.radians(getattr(self.config, 'max_missile_attack_angle', 60.0))

        # --- 能量威胁参数 ---
        self.altitude_advantage_ref = getattr(self.config, 'altitude_advantage_ref', 1000.0)
        self.velocity_advantage_ref = getattr(self.config, 'velocity_advantage_ref', 100.0)

        # --- 权重 ---
        # 几何威胁通常更直接，所以权重更高
        self.w_geometry = getattr(self.config, 'w_evasion_geometry', 0.7)
        self.w_energy = getattr(self.config, 'w_evasion_energy', 0.3)

        self.w_ta_angle = getattr(self.config, 'w_ta_angle', 0.5) # TA角最重要，代表战术态势
        self.w_ao_angle = getattr(self.config, 'w_ao_angle', 0.3) # AO角其次，代表攻击窗口
        self.w_range = getattr(self.config, 'w_range_geom', 0.2)   # 距离权重
        self.w_altitude = getattr(self.config, 'w_evasion_altitude', 0.5)
        self.w_velocity = getattr(self.config, 'w_evasion_velocity', 0.5)

        # --- 惩罚放大系数 ---
        # 惩罚的scale，使其与进攻奖励的大小相匹配
        self.penalty_scale = getattr(self.config, 'evasion_penalty_scale', -15.0)
        # 当有导弹来袭时，威胁的放大倍数
        self.missile_threat_multiplier = getattr(self.config, 'missile_threat_multiplier', 5.0)

    def get_reward(self, task, env, agent_id):
        ego_agent = env.agents[agent_id]
        if not ego_agent.is_alive:
            return 0

        ego_feature = np.hstack([ego_agent.get_position(), ego_agent.get_velocity()])
        ego_pos = ego_agent.get_position()
        ego_vel_norm = np.linalg.norm(ego_agent.get_velocity())

        max_threat_score = 0.0

        for enm in ego_agent.enemies:
            if not enm.is_alive:
                continue

            current_threat_score = _calculate_tactical_score(enm, ego_agent, self.config)

            if current_threat_score > max_threat_score:
                max_threat_score = current_threat_score

        # 奖励是最大威胁分数的负值
        new_reward = self.penalty_scale * max_threat_score

        return self._process(new_reward, agent_id)