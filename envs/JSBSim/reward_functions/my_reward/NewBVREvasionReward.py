import numpy as np
from envs.JSBSim.core.catalog import Catalog
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
# 必须引入这个工具，确保和你的 Obs 计算逻辑完全一致
from envs.JSBSim.utils.utils import get_AO_TA_R


class NewBVREvasionReward(BaseRewardFunction):
    """
    【绝对生存优先 - Obs同源修正版】
    特点：
    1. 数据同源：直接使用 get_AO_TA_R 计算 AO，确保 Obs 和 Reward 逻辑一致。
    2. 离散引导：保留了 +3/-3 的离散奖惩机制，训练稳定。
    3. 零阈值响应：只要 AO 在变大（变好），立刻给奖励，防止飞机“绝望直飞”。
    """

    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + '_total']
        self.danger_dist = 40000.0
        # 记录上一帧的 AO (Angle Off)
        self.last_ao = {}

    def reset(self, task, env):
        self.last_ao.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        # --- 1. 获取威胁信息 ---
        missiles = agent.check_all_missile_warning() if hasattr(agent, 'check_all_missile_warning') else []

        # 逻辑：如果有多个导弹，取最近的一个计算 AO
        # 这也是为了和 Obs 对齐 (Obs通常只看最近的威胁)
        if not missiles:
            self.last_ao.pop(agent_id, None)
            return 0.0

        ego_pos = agent.get_position()
        nearest_m = min(missiles, key=lambda m: np.linalg.norm(ego_pos - m.get_position()))
        dist = np.linalg.norm(ego_pos - nearest_m.get_position())

        if dist > self.danger_dist:
            self.last_ao.pop(agent_id, None)
            return 0.0

        # --- 2. 计算几何关系 (使用 Obs 同源算法) ---
        # 构造特征向量，格式需与 get_AO_TA_R 要求一致
        ego_feature = np.hstack([agent.get_position(), agent.get_velocity()])
        m_feature = np.hstack([nearest_m.get_position(), nearest_m.get_velocity()])

        # AO 范围: [0, PI]
        # 0 = 迎头 (极度危险)
        # PI = 背对 (安全)
        current_ao, _, _ = get_AO_TA_R(ego_feature, m_feature)

        # 获取上一帧 AO
        prev_ao = self.last_ao.get(agent_id, current_ao)

        # --- 3. 计算奖励 (离散逻辑) ---
        total_reward = 0.0

        # 计算改善量 (AO 越大越安全，所以正数代表改善)
        delta = current_ao - prev_ao

        # 定义阈值 (弧度制)
        # 1.57 (PI/2) = 90度，侧身
        # 0.52 ≈ 30度，迎头

        # === A. 状态分 (告诉它处境) ===
        if current_ao < 1.6:  # 小于90度 (处于前半球，危险)
            # 【危险区】
            # 基础惩罚 -5.0
            total_reward -= 5.0

            # === B. 引导分 (告诉它动作) ===
            # 关键修改：移除正向阈值！只要 delta > 0 就给分！
            if delta > 0.000001:
                # 动作正确：正在把机头转离导弹
                # 即使现在是 -5 分，加上这 3 分变成 -2 分，也是一种“巨大的解脱”
                total_reward += 3.0
            else:
                # 动作错误：直飞(delta=0) 或 转反了(delta<0)
                # 罪加一等，迫使它必须动起来
                total_reward -= 3.0

        else:
            # 【安全区】 (AO > 90度)
            # 活下来了，给固定正分
            total_reward += 2.0

            # 安全区内依然鼓励继续转到 180度 (纯背对)
            if delta > 0:
                total_reward += 1.0

        # --- 4. 存活/距离修正 ---
        # 距离极近 (<10km) 且 AO < 90度，追加惩罚
        if dist < 10000.0 and current_ao < 1.6:
            total_reward -= 5.0

        # 更新历史
        self.last_ao[agent_id] = current_ao

        return self._process(total_reward, agent_id)

    def _get_threat_info(self, agent, missiles):
        # 这个辅助函数主要用来判断是否有威胁存在，具体计算在主函数里用 get_AO_TA_R 做了
        if not missiles: return None, None, float('inf')
        ego_pos = agent.get_position()
        nearest_m = min(missiles, key=lambda m: np.linalg.norm(ego_pos - m.get_position()))
        dist = np.linalg.norm(ego_pos - nearest_m.get_position())
        if dist > self.danger_dist: return None, None, float('inf')
        return nearest_m.get_position(), nearest_m.get_velocity(), dist