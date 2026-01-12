import numpy as np

from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction


class BVRRelativeSpeedReward(BaseRewardFunction):
    """
    【相对速度奖励 - 鲁棒版】
    目标：建立相对于敌机的能量优势，但严格受限于自身的飞行包线。

    逻辑核心：
    1. 安全底线：无论敌机如何，自身速度绝不能低于失速速度 (V_stall)。
    2. 战术底线：无论敌机多慢，自身速度应保持在最佳机动速度 (V_corner) 以上。
    3. 压制目标：试图比敌机快 10%~20%。
    4. 物理上限：如果敌机速度超过我方极速，则以我方极速为目标，不惩罚无法超越的部分。
    """

    def __init__(self, config):
        super().__init__(config)
        # --- 自身性能参数 (m/s) ---
        self.v_stall = getattr(self.config, 'v_stall', 150.0)  # 失速/危险速度
        self.v_corner = getattr(self.config, 'v_corner', 400.0)  # 最佳机动速度 (下限保底)
        self.v_max = getattr(self.config, 'v_max', 600.0)  # 物理极速 (上限封顶)

        self.reward_item_names = [self.__class__.__name__ + '_total']

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]
        if len(agent.check_all_missile_warning()) != 0:
            return 0
        # 1. 获取最近敌机
        enemies = agent.enemies
        best_enemy = None
        min_dist = float('inf')
        ego_pos = agent.get_position()

        for enm in enemies:
            if hasattr(enm, 'is_alive') and not enm.is_alive: continue
            d = np.linalg.norm(ego_pos - enm.get_position())
            if d < min_dist:
                min_dist = d
                best_enemy = enm
        if min_dist > 50000:
            return 0
        # 2. 获取速度数据
        v_self = np.linalg.norm(agent.get_velocity())
        v_enemy = 0
        if best_enemy:
            v_enemy = np.linalg.norm(best_enemy.get_velocity())

        # 3. 计算“动态目标速度” (Dynamic Target Speed)
        # 逻辑：我要比敌机快一点(x1.1)，但不能慢于我的Corner Speed，也不能快过我的Max Speed
        target_v = max(self.v_corner, v_enemy * 1.1)
        target_v = min(target_v, self.v_max)

        reward = 0.0

        # --- 4. 分段奖励计算 ---

        # 区域 A: 危险区 (V < V_stall)
        # 绝对惩罚，不论敌机在干嘛
        if v_self < self.v_stall:
            # 线性惩罚: 速度越低分越低 (-1.0 ~ 0.0)
            reward = -1.0 * (1.0 - (v_self / self.v_stall))

        # 区域 B: 追赶区 (V_stall <= V < Target)
        # 此时还没有达到我们设定的战术目标
        elif v_self < target_v:
            # 计算完成度 (Progress)
            # 分母是 (目标 - 失速)，即有效操作区间
            # 结果范围 0.0 ~ 1.0 (线性增长)
            progress = (v_self - self.v_stall) / (target_v - self.v_stall + 1e-6)

            # 如果比敌机慢(v_self < v_enemy)，奖励给低一点 (0~0.4)
            # 如果比敌机快但没到目标(v_enemy < v_self < target)，奖励给高一点 (0.4~1.0)
            if v_self < v_enemy:
                reward = 0.4 * progress
            else:
                reward = 0.4 + 0.6 * progress

        # 区域 C: 达标区 (V >= Target)
        # 已经达到理想速度，保持满分
        else:
            reward = 1.0

            # [可选] 过速惩罚
            # 如果速度不仅达到了目标，还严重超过了物理极速(比如俯冲超速)，可以给一点惩罚
            if v_self > self.v_max:
                overspeed = max(0.0, v_self - self.v_max)
                reward -= 0.5 * (overspeed / self.v_max)

        return self._process(reward, agent_id)