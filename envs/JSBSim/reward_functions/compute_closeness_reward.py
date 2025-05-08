import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R

class ComputeClosenessReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.optimal_combat_range = getattr(self.config, 'optimal_combat_range')


    def get_reward(self, task, env, agent_id):
        new_reward = 0
        ego_range = set()
        # feature: (north, east, down, vn, ve, vd)
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                     enm.get_velocity()])
            AO, _, R = get_AO_TA_R(ego_feature, enm_feature)
            if 2000 < R < self.optimal_combat_range:
                ego_range.add(enm.uid)
            if R < 2000:
                new_reward -= 20

        if len(ego_range) > 1:
            new_reward += 4
        elif len(ego_range) == 1:
            new_reward += 2
        else:
            new_reward -= 5

        return new_reward
        # return self._process(new_reward, agent_id)



