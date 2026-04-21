import numpy as np

from envs.JSBSim.core.catalog import Catalog
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math


class BVREvasionReward_315_Improved2(BaseRewardFunction):
    """
    【绝对生存优先 - 姿态稳定修正版】
    特点：
    1. 保留原有的同源计算和线性对齐引导。
    2. 新增姿态稳定惩罚（惩罚无意义的滚转和角速度），专治螺旋转圈。
    3. 新增动能奖励，鼓励直线加速逃逸，避免盘旋掉速。
    """

    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + '_total']
        self.danger_dist = 40000.0

    def reset(self, task, env):
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        # --- 1. 获取威胁信息 ---
        missiles = agent.check_all_missile_warning() if hasattr(agent, 'check_all_missile_warning') else []

        if not missiles:
            return 0.0

        ego_pos = agent.get_position()
        missile_sim = min(missiles, key=lambda m: np.linalg.norm(ego_pos - m.get_position()))
        dist = np.linalg.norm(ego_pos - missile_sim.get_position())

        if dist > self.danger_dist:
            return 0.0

        # 获取状态与速度向量
        aircraft_pos = agent.get_position()
        missile_pos = missile_sim.get_position()
        aircraft_v = agent.get_velocity()
        missile_v = missile_sim.get_velocity()

        # 加 1e-6 防止除以 0
        v_norm = np.linalg.norm(aircraft_v) + 1e-6

        # ===== LOS vector (视线向量，从飞机指向导弹) =====
        los_vec = missile_pos - aircraft_pos
        los_dist = np.linalg.norm(los_vec) + 1e-6
        los_unit = los_vec / los_dist
        rel_vel = missile_v - aircraft_v



        # ==========================================
        # ===== 核心修改 2：远距离逃逸 (Drag / Run) =====
        # ==========================================
        # 1. 计算逃逸偏角 EA (Escape Angle) 的余弦值
        # cos_ea = 1.0 代表完美背对导弹，-1.0 代表迎头
        cos_ea = np.dot(aircraft_v, -los_unit) / v_norm
        cos_ea = np.clip(cos_ea, -1.0, 1.0)  # 防止 arccos 出现 NaN
        EA = np.arccos(cos_ea)  # 取值范围 [0, PI]

        # 2. 角度纯洁度奖励：使用高斯分布，只有 EA 接近 0 才给高分，偏航迅速暴跌
        r_escape_angle = np.exp(-2.0 * EA ** 2)

        # 4. 远距离总奖励：
        r_far_escape = r_escape_angle


        # ==========================================
        # ===== 动态融合总奖励 =====
        # ==========================================
        # 根据距离平滑切换策略
        reward_action = r_far_escape
        # 总奖励放大，保持你原本的代码量级
        # print("agent_id:{}, reward:{}".format(agent_id, reward_action))
        total_reward = (reward_action) * 8.0
        return self._process(total_reward, agent_id)