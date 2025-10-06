from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction


class ZKEventDrivenReward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Shot down by missile: -200
    - Crash accidentally: -200
    - Shoot down other aircraft: +200
    """

    def __init__(self, config):
        super().__init__(config)
        # 初始化用于跟踪已处理事件的集合
        # 确保每个事件只被奖励/惩罚一次
        self.shotdown_agents = set()
        self.crashed_agents = set()
        self.rewarded_missiles = set()

    def reset(self, task, env):
        """
        在每个回合开始时重置状态，清空已记录的事件。
        """
        self.shotdown_agents.clear()
        self.crashed_agents.clear()
        self.rewarded_missiles.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        计算一次性的事件奖励。

        Args:
            task: task 实例
            env: environment 实例
            agent_id: 当前飞机的ID

        Returns:
            (float): 单步奖励值
        """
        reward = 0
        agent = env.agents[agent_id]

        # 检查飞机是否被击落，并且这个事件是第一次发生
        if agent.is_shotdown and agent_id not in self.shotdown_agents:
            reward -= 300
            self.shotdown_agents.add(agent_id)

        # 检查飞机是否坠毁，并且这个事件是第一次发生
        elif agent.is_crash and agent_id not in self.crashed_agents:
            reward -= 500
            self.crashed_agents.add(agent_id)

        # 遍历飞机发射的所有导弹
        for missile in agent.launch_missiles:
            # 检查导弹是否成功命中，并且这个成功事件是第一次被奖励
            # 我们用导弹的唯一ID (missile.id) 来做标识
            if missile.is_success and missile.uid not in self.rewarded_missiles:
                reward += 300
                self.rewarded_missiles.add(missile.uid)

        # 注意：原代码中的 _process 方法依然保留，用于后续处理
        return self._process(reward, agent_id)
