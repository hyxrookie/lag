import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c


class TacticalDetectionReward(BaseRewardFunction):
    """
    当一个敌机从“未探测”变为“已探测”状态时，检查其冷却时间。
    如果冷却时间已过，则给予一次性的高额奖励。
    这个版本严格遵循“事件驱动”原则，并结合冷却机制，是最健壮的设计。
    """

    def __init__(self, config):
        super().__init__(config)
        self.reward_per_detection = getattr(self.config, 'reward_per_detection', 20.0)
        self.detection_cooldown_steps = getattr(self.config, 'detection_cooldown_steps', 100)

        # 为每个agent独立维护一个记忆体
        # 'last_rewarded': {enemy_id: step}
        # 'last_detected': {enemy_id_set}
        self.agent_memory = {}

    def reset(self, task, env):
        """在每个回合开始时，清空所有agent的记忆。"""
        self.agent_memory.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """计算奖励的核心逻辑。"""
        new_reward = 0
        current_step = env.current_step

        # --- 1. 初始化或获取该agent的记忆 ---
        if agent_id not in self.agent_memory:
            self.agent_memory[agent_id] = {
                'last_rewarded': {},
                'last_detected': set()
            }

        memory = self.agent_memory[agent_id]
        last_rewarded_steps = memory['last_rewarded']
        last_detected_ids = memory['last_detected']

        # --- 2. 获取当前探测到的所有敌人ID ---
        agent = env.agents[agent_id]
        current_detected_ids = {enemy.uid for enemy in agent.single_detected_enemies if enemy.is_alive}

        # --- 3. 遍历当前所有可见的敌人 ---
        for enemy_id in current_detected_ids:

            # 【核心条件1: 事件发生】检查这个敌人是否是“刚刚出现”的
            if enemy_id not in last_detected_ids:

                # 如果是刚刚出现，再检查冷却时间
                last_reward_step = last_rewarded_steps.get(enemy_id, -100)

                # 【核心条件2: 冷却完毕】
                if current_step - last_reward_step >= self.detection_cooldown_steps:
                    # 两个条件都满足，给予奖励！
                    new_reward += self.reward_per_detection

                    # 更新该敌人的奖励时间戳
                    last_rewarded_steps[enemy_id] = current_step

        # --- 4. 更新“上一帧”的探测列表，为下一帧做准备 ---
        memory['last_detected'] = current_detected_ids

        return self._process(new_reward, agent_id)