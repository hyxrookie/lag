import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.utils.utils import get_AO_TA_R


class ApproachAndOrientReward(BaseRewardFunction):
    """
    一个用于远距离引导的奖励函数，是 ComputeClosenessReward 的升级版。
    - 聚焦于最近的敌机，而非所有敌机的平均距离，避免在多目标下决策混乱。
    - 同时奖励两个行为：
      1. 朝向改善 (减小与最近敌机的AO角)。
      2. 距离拉近 (减小与最近敌机的距离)。
    - 奖励值是连续的，与改善的程度相关，提供了更平滑的学习梯度。
    """

    def __init__(self, config):
        super().__init__(config)
        # 权重，用于平衡“朝向”和“接近”的重要性
        self.w_orient = getattr(self.config, 'w_orient_approach', 0.4)
        self.w_approach = getattr(self.config, 'w_approach_approach', 0.6)

        self.optimal_combat_range = getattr(self.config, 'optimal_combat_range', 20000)

        # 奖励的放大系数
        self.reward_scale = getattr(self.config, 'approach_reward_scale', 5.0)

        # 存储上一帧的信息
        self.prev_info = {}

    def reset(self, task, env):
        self.prev_info.clear()
        return super().reset(task, env)
        # 注意: super().reset() 的调用方式可能依赖于基类实现
        # 如果基类有reset, 应该这样调用: super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        ego_agent = env.agents[agent_id]
        if not ego_agent.is_alive:
            # 清理死亡智能体的信息，避免内存泄漏
            if agent_id in self.prev_info:
                del self.prev_info[agent_id]
            return 0

        # --- 1. 找到最近的存活敌人 ---
        ego_feature = np.hstack([ego_agent.get_position(), ego_agent.get_velocity()])
        closest_enm = None
        min_dist = float('inf')

        for enm in ego_agent.enemies:
            if enm.is_alive:
                enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
                _, _, dist = get_AO_TA_R(ego_feature, enm_feature)
                if dist < min_dist:
                    min_dist = dist
                    closest_enm = enm

        if closest_enm is None:
            return 0  # 没有存活的敌人

        # --- 2. 计算与最近敌人的当前态势 ---
        enm_feature = np.hstack([closest_enm.get_position(), closest_enm.get_velocity()])
        current_AO, _, current_R = get_AO_TA_R(ego_feature, enm_feature)

        if current_R <= self.optimal_combat_range:
            # 同样需要更新信息，以防飞出范围后计算出错
            self.prev_info[agent_id] = {'ao_abs': abs(current_AO), 'r': current_R}
            return 0

        new_reward = 0

        # --- 3. 与上一帧比较，计算奖励 ---
        if agent_id in self.prev_info:
            prev_AO_abs = self.prev_info[agent_id]['ao_abs']
            prev_R = self.prev_info[agent_id]['r']

            # 计算AO角的改善程度。变化量归一化到 [-1, 1] 附近
            # (prev - curr) / pi，这样如果从pi变到0，得分最高
            orient_improvement = (prev_AO_abs - abs(current_AO)) / math.pi

            # 计算距离的改善程度。变化量用一个参考值归一化
            ego_speed = np.linalg.norm(ego_agent.get_velocity())
            enm_speed = np.linalg.norm(closest_enm.get_velocity())
            # 0.2 是仿真环境的时间步长
            max_closure_dist_per_step = (ego_speed + enm_speed) * 0.2

            approach_improvement = 0
            if max_closure_dist_per_step > 1e-6:  # 避免除以零
                # 距离改善量 / 单步最大可能改善量
                distance_change = prev_R - current_R
                approach_improvement = np.clip(distance_change / max_closure_dist_per_step, -1.0, 1.0)

            # 只有在改善时才给予奖励，避免因为情况恶化而惩罚
            # 这使得这个奖励函数更专注于“正向激励”
            orient_reward = orient_improvement
            approach_reward = approach_improvement

            # 加权求和
            new_reward = self.reward_scale * (self.w_orient * orient_reward + self.w_approach * approach_reward)

        # --- 4. 更新状态用于下一帧的计算 ---
        self.prev_info[agent_id] = {'ao_abs': abs(current_AO), 'r': current_R}

        return self._process(new_reward, agent_id)