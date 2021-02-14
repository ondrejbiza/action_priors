import numpy as np
from .PartitionedEpsGreedyPolicy import PartitionedEpsGreedyPolicy


class DynaPartitionedEpsGreedyPolicy(PartitionedEpsGreedyPolicy):

    def __init__(self, config, get_partition, probs=None):

        super(DynaPartitionedEpsGreedyPolicy, self).__init__(config, None)

        self.get_partition = get_partition
        self.probs = probs

    def act(self, state, qs, timestep, evaluation=False):

        self.set_partition(self.get_partition(state))

        if self.exploration_schedule is not None and np.random.rand() < \
                self.exploration_schedule.value(timestep) and not evaluation:

            if self.probs is not None:
                if len(self.probs.shape) == 2:
                    assert self.probs.shape[0] == 2
                    hand_state = int(state[2][0].detach().cpu().numpy())
                    if hand_state == 0:
                        probs = self.probs[0, self.blocks]
                    elif hand_state == 1:
                        probs = self.probs[1, self.blocks]
                    else:
                        raise ValueError("Hand bit problem.")
                else:
                    probs = self.probs[self.blocks]

                norm = np.sum(probs)
                if norm == 0:
                    # the available blocks have a cumulative probability of 0
                    # select them with uniform probability
                    probs += 1 / len(probs)
                else:
                    probs = probs / np.sum(probs)
                block = np.random.choice(list(range(len(self.blocks))), p=probs)
            else:
                block = np.random.choice(list(range(len(self.blocks))))

            action = np.random.choice(self.action_sets[block])

        else:

            action = np.argmax(qs)

        return action
