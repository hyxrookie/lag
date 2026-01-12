import numpy as np

from envs.JSBSim.core.catalog import Catalog
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R


class BVREvasionReward(BaseRewardFunction):
    """
    【防御模块 - Mach 版】
    目标：在被导弹攻击且距离较近时，全加力加速逃逸。
    使用 Mach 作为速度指标，适应全高度空域。
    """

    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + '_total']

        # --- 距离参数 ---
        self.danger_dist = 40000.0  # 40km 开始预警

        # --- 速度参数 (Mach) ---
        # 逃命时的目标马赫数
        # Mach 1.5 是一个非常好的逃逸速度，既快又在大气层内可达
        self.target_evade_mach = 1.5
    #
    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        # 1. 获取马赫数 (直接读取属性)
        # 务必确认这就是你列表中对应的那个属性
        mach = agent.get_property_value(Catalog.velocities_mach)

        # 2. 获取威胁信息
        missiles = agent.check_all_missile_warning() if hasattr(agent, 'check_all_missile_warning') else []
        threat_pos, threat_vel, dist = self._get_threat_info(agent, missiles)
        # print("missiles:{}, threat_pos:{}, threat_vel:{}, dist:{}".format(missiles, threat_pos, threat_vel, dist))
        if threat_pos is None:
            return 0.0

        # 3. 计算几何关系 (AO 依然需要向量计算，这部分不能省)
        ego_pos = agent.get_position()
        ego_vel = agent.get_velocity()  # 这里依然需要向量来算角度

        ego_feature = np.hstack([ego_pos, ego_vel])
        threat_feature = np.hstack([threat_pos, threat_vel])
        # AO, _, _ = get_AO_TA_R(ego_feature, threat_feature)
        aircraft_v = agent.get_velocity()

        # 4. 计算紧迫感 (Urgency)
        # 距离越近，权重越大

        # urgency = self.compute_threat_from_distance(dist)
        # 5. 核心计算
        dir_factor = np.dot(threat_vel, aircraft_v) / (np.linalg.norm(threat_vel) * np.linalg.norm(aircraft_v) + 1e-6)

        if dir_factor > 0:
            # --- 正确逃逸方向 ---

            # 使用 Mach 计算速度得分
            # 逻辑：Mach 越接近 target_evade_mach (1.5) 分越高
            # 如果 Mach > 1.5，保持满分 (逃命不怕快)
            if mach >= self.target_evade_mach:
                speed_score = 1.0
            else:
                # 线性插值: 0.4(起步) -> 1.5(满分)
                # 加上 max(0, ...) 防止静止时出现负分
                speed_score = max(0.0, (mach - 0.4) / (self.target_evade_mach - 0.4))

            # 组合奖励
            # total_reward = (dir_factor * (0.3 + 0.7 * speed_score)) * (0.2 + 0.8 *urgency)
            total_reward = dir_factor * (0.3 + 0.7 * speed_score)
        else:
            # --- 错误方向 (迎头冲向导弹) ---
            # 直接惩罚，不需要乘速度
            total_reward = dir_factor

        # print("agentID:{},躲避奖励:{}".format(agent_id, total_reward))

        return self._process(total_reward, agent_id)

    def _get_threat_info(self, agent, missiles):
        # ... (保持原有的距离筛选逻辑) ...
        if not missiles:
            return None, None, float('inf')
        ego_pos = agent.get_position()
        nearest_m = min(missiles, key=lambda m: np.linalg.norm(ego_pos - m.get_position()))
        dist = np.linalg.norm(ego_pos - nearest_m.get_position())
        if dist > self.danger_dist:
            return None, None, float('inf')
        return nearest_m.get_position(), nearest_m.get_velocity(), dist

    def compute_threat_from_distance(self, dist):
        d_launch = 40000.0
        d_critical = 30000.0

        if dist >= d_launch:
            return 0.0
        if dist <= d_critical:
            return 1.0

        # 归一化
        x = (d_launch - dist) / (d_launch - d_critical)

        # 指数压缩（前慢后快）
        threat = 1.0 - np.exp(-4.0 * x)
        return np.clip(threat, 0.0, 1.0)