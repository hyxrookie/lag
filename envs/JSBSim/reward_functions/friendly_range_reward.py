import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R

class FriendlyRangeReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.optimal_combat_range = getattr(self.config, 'optimal_combat_range')


    def get_reward(self, task, env, agent_id):
        new_reward = 0
        ego_range = set()
        # feature: (north, east, down, vn, ve, vd)
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        for partner in env.agents[agent_id].partners:
            if not partner.is_alive:
                continue
            partner_feature = np.hstack([partner.get_position(),
                                         partner.get_velocity()])
            AO, _, R = get_AO_TA_R(ego_feature, partner_feature)
            if 5000 < R > 8000:
                new_reward -= 10
            elif R > 8000:
                new_reward -=20


        return self._process(new_reward, agent_id)



