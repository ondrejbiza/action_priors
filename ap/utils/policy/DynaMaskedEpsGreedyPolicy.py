import numpy as np
from .MaskedEpsGreedyPolicy import MaskedEpsGreedyPolicy


class DynaMaskedEpsGreedyPolicy(MaskedEpsGreedyPolicy):

    def __init__(self, config, get_mask):

        super(DynaMaskedEpsGreedyPolicy, self).__init__(config, None)

        self.get_mask = get_mask

    def act(self, state, qs, timestep, evaluation=False):

        self.set_mask(self.get_mask(state))

        if self.exploration_schedule is not None and np.random.rand() < \
                self.exploration_schedule.value(timestep) and not evaluation:

            # 50% chance of selecting good / bad actions
            # but, make sure there's at least one good / bad action to select
            # TODO: make p a parameter
            if (np.random.choice([True, False], p=[0.8, 0.2]) and len(self.good_actions) > 0) or len(self.bad_actions) == 0:
                action = np.random.choice(self.good_actions)
            else:
                action = np.random.choice(self.bad_actions)

        else:

            action = np.argmax(qs)

        return action
