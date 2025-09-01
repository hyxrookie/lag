import numpy as np
import math
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R, _calculate_tactical_score


class TeamReward(BaseRewardFunction):
    """
    计算纯粹的团队奖励。
    这个奖励值对团队中所有存活的成员都是相同的，旨在引导智能体关注整个战局的宏观优劣。
    它由两部分组成：
    1. 数量优势 (Numerical Superiority): 基于存活的友方和敌方飞机数量差异。
    2. 态势优势 (Aggregate Advantage): 基于整个团队对敌方团队的战术总优势。
    """

    def __init__(self, config):
        super().__init__(config)
        # 团队奖励内部各部分的权重
        self.w_numerical = getattr(self.config, 'w_team_numerical', 0.6)
        self.w_advantage = getattr(self.config, 'w_team_advantage', 0.4)

        # 各部分奖励的放大系数
        self.numerical_scale = getattr(self.config, 'team_numerical_scale', 20.0)
        self.advantage_scale = getattr(self.config, 'team_advantage_scale', 10.0)

    def get_reward(self, task, env, agent_id):
        """
        计算并返回团队奖励。
        注意：虽然传入了 agent_id，但计算结果对于同一团队的所有成员都是一样的。
        """
        ego_agent = env.agents[agent_id]

        # --- 1. 计算数量优势奖励 ---
        alive_friends = sum(1 for agent in ego_agent.partners if agent.is_alive)
        alive_friends = alive_friends + (1 if ego_agent.is_alive else 0)
        alive_enemies = sum(1 for agent in ego_agent.enemies if agent.is_alive)

        # 数量差异归一化到 [-1, 1]
        # (我方存活数 - 敌方存活数) / 每队初始人数
        initial_squad_size = len(ego_agent.partners) + 1
        if initial_squad_size > 0:
            numerical_advantage_ratio = (alive_friends - alive_enemies) / initial_squad_size
        else:
            numerical_advantage_ratio = 0.0

        numerical_reward = self.numerical_scale * numerical_advantage_ratio

        # --- 2. 计算态势优势奖励 ---
        # a. 计算我方对敌方的总优势：
        #    对于每个友军，找出其对所有敌军的最大战术优势，然后将这些最大值相加。
        total_friend_advantage = 0
        for friend in ego_agent.partners + [ego_agent]:
            if friend.is_alive:
                max_advantage_for_friend = 0
                for enm in ego_agent.enemies:
                    if enm.is_alive:
                        score = _calculate_tactical_score(friend, enm, self.config)
                        if score > max_advantage_for_friend:
                            max_advantage_for_friend = score
                total_friend_advantage += max_advantage_for_friend

        # b. 计算敌方对我方的总威胁（即敌方的总优势）：
        #    对于每个敌军，找出其对所有我方单位的最大战术优势，然后相加。
        total_enemy_advantage = 0
        for enm in ego_agent.enemies:
            if enm.is_alive:
                max_advantage_for_enemy = 0
                for friend in ego_agent.partners + [ego_agent]:
                    if friend.is_alive:
                        score = _calculate_tactical_score(enm, friend, self.config)
                        if score > max_advantage_for_enemy:
                            max_advantage_for_enemy = score
                total_enemy_advantage += max_advantage_for_enemy

        # c. 计算优势差并归一化
        #    (我方总优势 - 敌方总优势) / 我方初始人数
        if initial_squad_size > 0:
            advantage_diff_ratio = (total_friend_advantage - total_enemy_advantage) / initial_squad_size
        else:
            advantage_diff_ratio = 0.0

        aggregate_advantage_reward = self.advantage_scale * advantage_diff_ratio

        # --- 3. 组合最终团队奖励 ---
        # 这个奖励值将被加到每个智能体的个人奖励上
        team_reward = (self.w_numerical * numerical_reward +
                       self.w_advantage * aggregate_advantage_reward)

        # 团队奖励是全局的，不需要经过 _process 处理
        return team_reward