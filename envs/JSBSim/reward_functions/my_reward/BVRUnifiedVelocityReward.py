import numpy as np

from envs.JSBSim.core.catalog import Catalog
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction


class BVRUnifiedVelocityReward(BaseRewardFunction):
    """
【全高度通用速度奖励】
    核心思想：
    1. 惩罚逻辑：看表速 (Vc)。无论在哪，表速太低就会失速。
    2. 奖励逻辑：看马赫 (Mach)。BVR 空战讲究的是马赫数 (0.9巡航, 1.2+战斗)。
    """

    def __init__(self, config):
        super().__init__(config)

        # --- 1. 生存底线 (使用 Vc - m/s) ---
        # F-16 无论在什么高度，表速低于 100 m/s (约195节) 都非常危险
        self.vc_stall_threshold = 100.0

        # --- 2. 战术目标 (使用 Mach) ---
        # 这些数值在 1000m 和 10000m 都是通用的战术指标
        self.mach_cruise = 0.9  # 高亚音速巡航 (省油且快)
        self.mach_combat = 1.2  # 超音速截击 (导弹射程加成)
        self.mach_max = 2.0  # 物理极限

        # 距离参数 (单位: 米)
        self.d_cruise_start = 60000.0
        self.d_combat_start = 50000.0

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        # --- 关键修改：获取正确的速度参数 ---
        # 假设 agent.sim 或 agent 提供了访问属性的方法
        # 你需要确保这里能取到你在列表中列出的那两个值
        # 伪代码：
        vc = agent.get_property_value(Catalog.velocities_vc_mps)  # #12
        mach = agent.get_property_value(Catalog.velocities_mach)  # #19

        # 如果你的 get_velocity() 只能返回 vector，你需要去改环境接口
        # 或者暂时用 TAS 估算 Mach，但强烈建议直接读属性

        target_enm, dist = self._get_target_and_dist(agent)

        # -----------------------------------------------------------
        # 1. 绝对惩罚：防失速 (使用 Vc)
        # -----------------------------------------------------------
        # 这是物理铁律：Vc 不够，飞机就挂不住。和高度无关。
        if vc < self.vc_stall_threshold:
            # 线性或指数惩罚
            return self._process(-1.0 * (1.0 - vc / self.vc_stall_threshold), agent_id)

        # -----------------------------------------------------------
        # 2. 战术奖励 (使用 Mach)
        # -----------------------------------------------------------

        # 确定当前的战术目标是 巡航 还是 战斗
        if target_enm is None:
            w_cruise = 1.0
        else:
            if dist >= self.d_cruise_start:
                w_cruise = 1.0
            elif dist <= self.d_combat_start:
                w_cruise = 0.0
            else:
                w_cruise = (dist - self.d_combat_start) / (self.d_cruise_start - self.d_combat_start)
        w_combat = 1.0 - w_cruise

        # A. 巡航奖励 (目标 Mach 0.9)
        # 无论在 5000m 还是 10000m，0.9 都是很好的巡航速度
        r_cruise = np.exp(-((mach - self.mach_cruise) ** 2) / (2 * 0.1 ** 2))  # sigma=0.1 mach

        # B. 空战奖励 (目标 Mach 1.2+)
        # 鼓励飞得越快越好，直到 Mach 1.5 左右边际递减
        target_mach = self.mach_combat

        r_combat = 0.0
        if mach < target_mach:
            # 处于追赶阶段 (例如 0.6 -> 1.2)
            # 0.4 是大概的起飞/低速 Mach
            r_combat = (mach - 0.4) / (target_mach - 0.4)
        else:
            # 达标
            r_combat = 1.0
            # 超速惩罚 (Mach > 2.0)
            if mach > self.mach_max:
                r_combat -= (mach - self.mach_max)

        total_reward = w_cruise * r_cruise + w_combat * r_combat

        return self._process(total_reward, agent_id)

    def _get_target_and_dist(self, agent):
        """ 复用之前的选敌逻辑 """
        enemies = agent.enemies
        if not enemies: return None, float('inf')
        ego_pos = agent.get_position()

        # 简单取最近
        best_target = min(enemies, key=lambda e: np.linalg.norm(ego_pos - e.get_position()))
        if hasattr(best_target, 'is_alive') and not best_target.is_alive:
            return None, float('inf')  # 虽有对象但已死

        dist = np.linalg.norm(ego_pos - best_target.get_position())
        return best_target, dist