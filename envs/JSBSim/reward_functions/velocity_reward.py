import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R

class VelocityReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.optimal_combat_range = getattr(self.config, 'optimal_combat_range')


    def get_reward(self, task, env, agent_id):
        new_reward = 0
        v = np.linalg.norm(env.agents[agent_id].get_velocity())
        if  v<= 150:
            new_reward -= 20


        return self._process(new_reward, agent_id)



