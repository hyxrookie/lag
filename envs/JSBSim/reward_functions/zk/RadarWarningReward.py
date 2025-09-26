import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c


class RadarWarningReward(BaseRewardFunction):
    """
    当无人机被锁定后，提供宽限期，超时则惩罚。
    成功摆脱锁定会给予一次性奖励，但该奖励有较长的冷却时间，
    以防止AI通过反复被锁/解锁来刷分。
    """

    def __init__(self, config):
        super().__init__(config)

        self.reward_on_break_lock = getattr(self.config, 'reward_on_break_lock', 20.0)
        self.penalty_per_step_under_lock = getattr(self.config, 'penalty_per_step_under_lock', -1.0)
        self.grace_period_steps = getattr(self.config, 'grace_period_steps', 25)  # 5秒反应时间

        # 【关键】为“摆脱锁定”奖励增加冷却期
        self.break_lock_cooldown_steps = getattr(self.config, 'break_lock_cooldown_steps', 100)  # 20秒冷却

        # 分别存储锁定开始时间和上次获得奖励的时间
        self.lock_start_steps = {}
        self.last_rewarded_break_steps = {}

    def reset(self, task, env):
        """在每个回合开始时，清空所有记录。"""
        self.lock_start_steps.clear()
        self.last_rewarded_break_steps.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """计算奖励和惩罚的核心逻辑。"""
        agent = env.agents[agent_id]
        current_step = env.current_step
        rwr_level = agent.get("WarningNumber")

        # --- 1. 检查是否处于高度威胁下 ---
        if rwr_level >= 2:
            if agent_id not in self.lock_start_steps:
                self.lock_start_steps[agent_id] = current_step
                return 0.0
            else:
                lock_duration = current_step - self.lock_start_steps[agent_id]
                if lock_duration > self.grace_period_steps:
                    return self._process(self.penalty_per_step_under_lock, agent_id)
                else:
                    return 0.0

        # --- 2. 如果威胁已解除 ---
        else:
            if agent_id in self.lock_start_steps:
                # 威胁刚刚解除，首先清除锁定计时
                del self.lock_start_steps[agent_id]

                # 【关键改动】在给予奖励前，检查冷却时间
                last_reward_step = self.last_rewarded_break_steps.get(agent_id, -100)

                if current_step - last_reward_step > self.break_lock_cooldown_steps:
                    # 冷却已过，可以给予奖励
                    self.last_rewarded_break_steps[agent_id] = current_step  # 更新奖励时间
                    return self._process(self.reward_on_break_lock, agent_id)
                else:
                    # 仍在冷却期，不给奖励
                    return 0.0

            return 0.0