import numpy as np
from .EpsGreedyPolicy import EpsGreedyPolicy


class MaskedEpsGreedyPolicy(EpsGreedyPolicy):

    def __init__(self, config, mask):

        super(MaskedEpsGreedyPolicy, self).__init__(config)

        self.mask = None

        if mask is not None:
            self.set_mask(mask)

    def act(self, state, qs, timestep, evaluation=False):

        if self.exploration_schedule is not None and np.random.rand() < \
                self.exploration_schedule.value(timestep) and not evaluation:

            if np.random.choice([True, False]):
                action = np.random.choice(self.good_actions)
            else:
                action = np.random.choice(self.bad_actions)

        else:

            action = np.argmax(qs)

        return action

    def get_partition_(self):

        good_actions = np.where(self.mask.reshape(-1) == True)[0]
        bad_actions = np.where(self.mask.reshape(-1) == False)[0]

        return good_actions, bad_actions

    def set_mask(self, mask):

        self.mask = mask
        assert self.mask.dtype == np.bool

        self.good_actions, self.bad_actions = self.get_partition_()
