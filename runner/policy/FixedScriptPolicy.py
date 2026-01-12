import numpy as np
import torch


class FixedScriptPolicy:
    def __init__(self, act_space, num_agents=4, device=None):
        self.act_space = act_space
        self.device = device
        # 初始化 4 个直线飞行的 Agent
        self.agents = [
            StraightLineAgent(agent_id=i, target_alt=6000, target_vel=243)
            for i in range(num_agents)
        ]

    def prep_rollout(self):
        pass

    def act(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Runner 传入的 obs 通常形状是 [Batch, Dim] 或 [Batch, Num_Agents, Dim]
        这里假设 input obs 已经是 numpy 数组
        """
        # 如果 obs 是 tensor，转 numpy (兼容性处理)
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()

        batch_size = obs.shape[0]
        actions_list = []

        # 这里的逻辑取决于 Runner 如何 split 环境。
        # 如果 Runner 是一次性传进所有并行环境的数据 (Batch Size = n_rollout_threads)
        # 且每个环境里的 agent 是通过 Loop 处理的：

        # 假设 Runner 中的逻辑是：
        # for policy in opponent_policy:
        #    policy.act(obs[env_idx])
        # 这里的 obs[env_idx] 形状是 [N_Threads, Obs_Dim] (如果对手是单个)
        # 或者 [N_Threads * Num_Opponents, Obs_Dim]

        # 但要注意：我们现在是一个 Policy 控制 4 个 Agent (如果是多智能体且对手共享策略)
        # 或者 Runner 分别调用。

        # 既然你说要控制 4 架飞机，通常有两种情况：
        # 1. 这是一个 "Shared Policy"，所有 4 架飞机共用这个 act。
        #    此时 obs 的第一维通常是 N_Threads * Num_Agents (如果是拼接的)
        # 2. 或者 Runner 知道这是 4 个不同的 Agent。

        # 针对 JSBSim 代码常见结构，act 接收的是 batch 数据。
        # 我们需要知道当前这个 obs 属于哪架飞机。
        # 但通常 Policy.act 不知道 agent_id。

        # 简易处理：假设 obs 的顺序就是 agent 0, 1, 2, 3 的顺序循环
        # 或者我们简单地让所有飞机都执行同样的 "直线逻辑" (因为直线逻辑是无状态的，除了RNN)

        actions = []
        next_rnn_states = []

        # 遍历 Batch 中的每一行数据
        for i in range(batch_size):
            # 简单的 Round-Robin 映射：假设 batch 中的数据顺序对应 agent 0,1,2,3...
            # 如果 batch_size = n_threads，说明这是一个 agent 的数据。
            # 为了简化，我们只使用 agents[0] 的逻辑 (因为大家都是飞直线，逻辑是一样的)

            # 这里的 single_obs 是 [Obs_Dim]
            single_obs = obs[i]

            # 使用第一个 agent 的逻辑处理所有输入 (因为参数 target_alt 都一样)
            # 如果每架飞机目标不同，需要更复杂的 batch index 映射
            action = self.agents[0].get_action(single_obs)

            actions.append(action)
            # RNN state 处理略显复杂，这里简化为返回 zeros 或 current
            next_rnn_states.append(self.agents[0].rnn_states)

        actions = np.array(actions)

        # 确保维度匹配 [Batch, Action_Dim]
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, axis=-1)

        # 伪装成 Tensor 返回，因为 Runner 需要
        return torch.tensor(actions), torch.tensor(np.array(next_rnn_states))

    def load_state_dict(self, state_dict):
        pass