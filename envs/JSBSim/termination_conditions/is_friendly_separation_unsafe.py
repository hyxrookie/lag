from .termination_condition_base import BaseTerminationCondition
from ..core.catalog import Catalog as c
import numpy as np

from ..utils.utils import get_AO_TA_R


class FriendlySeparationUnsafe(BaseTerminationCondition):
    """
    LowAltitude
    End up the simulation if altitude are too low.
    """

    def __init__(self, config):
        super().__init__(config)
        self.friendly_separation_unsafe = getattr(config, 'friendly_separation_unsafe', 50)  # unit: m

    #目前仅能用于2v2
    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if altitude are too low.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        done = False
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        for partner in env.agents[agent_id].partners:
            partner_feature = np.hstack([partner.get_position(),
                                     partner.get_velocity()])
            AO, _, R = get_AO_TA_R(ego_feature, partner_feature)
            if R < self.friendly_separation_unsafe:
                done = True
        if done:
            env.agents[agent_id].crash()
            if env.agents[agent_id].partners is not None:
                env.agents[agent_id].partners[0].crash()
            self.log(f'{agent_id} friendly separation unsafe. Total Steps={env.current_step}')
        success = False
        return done, success, info
