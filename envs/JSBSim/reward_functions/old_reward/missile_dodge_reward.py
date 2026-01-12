import logging

import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R

class MissileDodgeReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.pre_missiles = {}
        self.pre_missiles_v = {}
        self.prev_velocity_component = {}
    def reset(self, task, env):
        self.pre_missiles.clear()
        self.pre_missiles_v.clear()
        self.prev_velocity_component.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        new_reward = 0
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        missile_sims = env.agents[agent_id].check_all_missile_warning()
        agent = env.agents[agent_id]
        for sim in missile_sims:
            if not sim.is_alive :
                #成功躲避导弹给比较大的奖励
                if env.agents[agent_id].is_alive:
                    new_reward += 100
                continue

            sim_feature = np.hstack([sim.get_position(), sim.get_velocity()])
            relative_angle = np.degrees(np.arccos(np.dot(sim.get_velocity(), env.agents[agent_id].get_velocity()) /
                                                  (np.linalg.norm(sim.get_velocity()) * np.linalg.norm(env.agents[agent_id].get_velocity()))))

            if 60 <= relative_angle <= 110:
                new_reward += 7
            elif 30 <= relative_angle <= 60:
                new_reward += 12
            elif 0 <= relative_angle <= 30:
                new_reward += 15
            else:
                new_reward -= 20


        return self._process(new_reward, agent_id)



