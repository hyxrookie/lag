import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.utils.utils import get_AO_TA_R, _calculate_tactical_score
from ..core.catalog import Catalog as c

class TacticalAdvantageReward(BaseRewardFunction):
    """
    奖励智能体对最具战术优势的敌方目标进行占位。
    - 这是一个增强版本，综合评估四个维度：
      1. 角度优势 (Angle): TA角，代表了占位的几何优势。
      2. 距离优势 (Range): 是否在最佳攻击距离上。
      3. 高度优势 (Altitude): 是否比敌机飞得更高。
      4. 速度优势 (Velocity): 是否比敌机飞得更快。
    - 最终奖励是基于对最佳目标的“综合战术优势分数”。
    """

    def __init__(self, config):
        super().__init__(config)
        # --- 几何优势参数 ---
        self.min_attack_range = getattr(self.config, 'min_attack_range', 4000.0)
        self.max_attack_range = getattr(self.config, 'max_attack_range', 14000.0)
        # 衰减因子决定了在范围外奖励下降的速度，值越小，下降越平缓
        self.range_decay_factor = getattr(self.config, 'range_decay_factor', 0.0005)
        self.max_ao_rad = math.radians(getattr(self.config, 'max_missile_attack_angle', 60.0))

        # --- 能量优势参数 ---
        # 高度优势的参考值（米）。超过这个高度差，优势得分达到最大。
        self.altitude_advantage_ref = getattr(self.config, 'altitude_advantage_ref', 1000.0)
        # 速度优势的参考值（米/秒）。超过这个速度差，优势得分达到最大。
        self.velocity_advantage_ref = getattr(self.config, 'velocity_advantage_ref', 100.0)

        # --- 权重 ---
        # 将总优势分为两部分：几何优势 和 能量优势
        self.w_geometry = getattr(self.config, 'w_geometry', 0.6)  # 占位本身更重要
        self.w_energy = getattr(self.config, 'w_energy', 0.4)  # 能量是实现占位的基础

        # 在各自部分内部的权重
        self.w_ta_angle = getattr(self.config, 'w_ta_angle', 0.5) # TA角最重要，代表战术态势
        self.w_ao_angle = getattr(self.config, 'w_ao_angle', 0.3) # AO角其次，代表攻击窗口
        self.w_range = getattr(self.config, 'w_range_geom', 0.2)   # 距离权重
        self.w_altitude = getattr(self.config, 'w_altitude', 0.5)
        self.w_velocity = getattr(self.config, 'w_velocity', 0.5)

        # 最终奖励的放大系数
        # self.reward_scale = getattr(self.config, 'advantage_reward_scale', 15.0)  # 因为维度更多，可以适当提高

    def get_reward(self, task, env, agent_id):
        ego_agent = env.agents[agent_id]
        if not ego_agent.is_alive:
            return 0

        ego_feature = np.hstack([ego_agent.get_position(), ego_agent.get_velocity()])
        ego_pos = ego_agent.get_position()
        ego_vel_norm = np.linalg.norm(ego_agent.get_velocity())

        max_tactical_score = 0.0

        for enm in ego_agent.enemies:
            if not enm.is_alive:
                continue

            current_tactical_score = _calculate_tactical_score(ego_agent, enm, self.config)

            if current_tactical_score > max_tactical_score:
                max_tactical_score = current_tactical_score

        new_reward = max_tactical_score

        alive_missile = list(filter(lambda x: x.is_alive, env.agents[agent_id].check_all_missile_warning()))
        #如果被锁定了，这时应该优先考虑躲避
        if len(alive_missile) > 0:
            new_reward = 0.1 * new_reward

        return self._process(new_reward, agent_id)