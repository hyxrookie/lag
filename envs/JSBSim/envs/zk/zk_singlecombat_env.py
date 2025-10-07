from typing import Tuple, Dict

import numpy as np
from envs.JSBSim.envs.env_base import BaseEnv
from envs.JSBSim.envs.zk.zk_env_base import ZKBaseEnv
from envs.JSBSim.tasks import SingleCombatTask, SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, \
    HierarchicalSingleCombatShootTask, SingleCombatShootMissileTask, HierarchicalSingleCombatTask
from envs.JSBSim.human_task.HumanSingleCombatTask import  HumanSingleCombatTask
from envs.JSBSim.tasks.zk.zk_singlecombat_task import ZKHierarchicalSingleCombatShootTask


class ZKSingleCombatEnv(ZKBaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str, port):
        super().__init__(config_name, port)
        # Env-Specific initialization here!




    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'singlecombat':
            self.task = SingleCombatTask(self.config)
        elif taskname == 'hierarchical_singlecombat':
            self.task = HierarchicalSingleCombatTask(self.config)
        elif taskname == 'singlecombat_dodge_missile':
            self.task = SingleCombatDodgeMissileTask(self.config)
        elif taskname == 'singlecombat_shoot':
            self.task = SingleCombatShootMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_dodge_missile':
            self.task = HierarchicalSingleCombatDodgeMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_shoot':
            self.task = HierarchicalSingleCombatShootTask(self.config)
        elif taskname == 'HumanSingleCombat':
            self.task = HumanSingleCombatTask(self.config)
        elif taskname == 'zk_hierarchical_singlecombat_shoot':
            self.task = ZKHierarchicalSingleCombatShootTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (dict): {agent_id: initial observation}
            share_obs (dict): {agent_id: initial state}
        """
        self.current_step = 0
        self._zk_sims.clear()
        self._zk_missiles.clear()
        # self.reset_simulators()

        red_x, red_y, red_psi, red_v, blue_x, blue_y, blue_psi, blue_v, h = self.get_common_init_pos()
        reset_attribute = self.reset_variable(red_x, red_y, red_psi, red_v, blue_x,
                                              blue_y, blue_psi, blue_v, h, self.red_num, self.blue_num)
        init_info = {'red': reset_attribute['red'],
                     'blue': reset_attribute['blue']}
        if self.INITIAL is False:
            self.INITIAL = True
            init_info['flag'] = {'init': {'render': self.RENDER, 'save': 0}}
        else:
            # print("reset-------------")
            init_info['flag'] = {'reset': {'render': self.RENDER}}
        self._send_condition(init_info)

        zk_obs = self._accept_from_socket()
        self.update_from_obs(zk_obs)
        self.task.reset(self)
        obs = self.get_obs()
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in self.agents.keys()}
        return self._pack(obs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (dict): the agents' actions, each key corresponds to an agent_id

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                share_obs: agents' share observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}
        # print("self.current_step:{}".format(self.current_step))
        # apply actions
        action = self._unpack(action)
        send_action = self.postprocess_action(action)

        self._send_condition(send_action)
        zk_obs = self._accept_from_socket()
        # print("zk_obs:{}".format(zk_obs) )
        self.update_from_obs(zk_obs)
        self.task.step(self)
        obs = self.get_obs()

        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]
        #
        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]

        return self._pack(obs), self._pack(rewards), self._pack(dones), info

