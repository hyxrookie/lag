import numpy as np
from gymnasium import spaces
from typing import List, Tuple
from abc import ABC, abstractmethod
from ..core.catalog import Catalog as c
from ..termination_conditions.mylog import event_logger


class BaseTask(ABC):
    """
    Base Task class.
    A class to subclass in order to create a task with its own observation variables,
    action variables, termination conditions and reward functions.
    """
    def __init__(self, config):
        self.config = config
        self.reward_functions = []
        self.termination_conditions = []
        self.load_variables()
        self.load_observation_space()
        self.load_action_space()

    @property
    def num_agents(self):
        return 1

    @abstractmethod
    def load_variables(self):
        self.state_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,
            c.fcs_elevator_cmd_norm,
            c.fcs_rudder_cmd_norm,
            c.fcs_throttle_cmd_norm,
        ]

    @abstractmethod
    def load_observation_space(self):
        """
        Load observation space
        """
        self.observation_space = spaces.Discrete(5)

    @abstractmethod
    def load_action_space(self):
        """
        Load action space
        """
        self.action_space = spaces.Discrete(5)

    def reset(self, env):
        """Task-specific reset

        Args:
            env: environment instance
        """
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)

    def step(self, env):
        """ Task-specific step

        Args:
            env: environment instance
        """
        pass

    # def get_reward(self, env, agent_id, info={}) -> Tuple[float, dict]:
    #     """
    #     Aggregate reward functions
    #
    #     Args:
    #         env: environment instance
    #         agent_id: current agent id
    #         info: additional info
    #
    #     Returns:
    #         (tuple):
    #             reward(float): total reward of the current timestep
    #             info(dict): additional info
    #     """
    #     total_reward = 0.0
    #     reward_details = []  # 存储每个奖励函数的详细信息
    #     per_function_rewards = {}  # 记录每个函数的奖励值，用于info返回
    #
    #     # 遍历所有奖励函数，计算并记录每个函数的奖励
    #     for idx, reward_function in enumerate(self.reward_functions):
    #         # 获取单个奖励函数的奖励值
    #         func_reward = reward_function.get_reward(self, env, agent_id)
    #         total_reward += func_reward
    #
    #         # 获取奖励函数的名称（增强可读性）
    #         func_name = reward_function.__class__.__name__
    #         # func_name = getattr(reward_function, '__name__', f'reward_function_{idx}')
    #         # if hasattr(reward_function, 'name'):
    #         #     func_name = reward_function.name  # 如果有自定义名称则使用
    #
    #         # 记录奖励值
    #         per_function_rewards[func_name] = func_reward
    #         reward_details.append(f"{func_name}: {func_reward:.6f}")
    #
    #     # 拼接奖励信息字符串并打印
    #     reward_str = f"Agent {agent_id} 奖励明细 | " + " | ".join(reward_details)
    #     reward_str += f" | 总奖励: {total_reward:.6f}"
    #     event_logger.info(reward_str)
    #     # 将每个函数的奖励信息添加到info中返回
    #     info['per_function_rewards'] = per_function_rewards
    #     info['total_reward'] = total_reward
    #
    #     return total_reward, info
    def get_reward(self, env, agent_id, info={}) -> Tuple[float, dict]:
        """
        Aggregate reward functions

        Args:
            env: environment instance
            agent_id: current agent id
            info: additional info

        Returns:
            (tuple):
                reward(float): total reward of the current timestep
                info(dict): additional info
        """
        reward = 0.0
        for reward_function in self.reward_functions:
            reward += reward_function.get_reward(self, env, agent_id)
        return reward, info

    def get_termination(self, env, agent_id, info={}) -> Tuple[bool, dict]:
        """
        Aggregate termination conditions

        Args:
            env: environment instance
            agent_id: current agent id
            info: additional info

        Returns:
            (tuple):
                done(bool): whether the episode has terminated
                info(dict): additional info
        """
        done = False
        success = True
        for condition in self.termination_conditions:
            d, s, info = condition.get_termination(self, env, agent_id, info)
            done = done or d
            success = success and s
            if done:
                break
        return done, info

    def get_obs(self, env, agent_id):
        """Extract useful informations from environment for specific agent_id.
        """
        return np.zeros(2)

    def normalize_action(self, env, agent_id, action):
        """Normalize action to be consistent with action space.
        """
        return np.array(action)
