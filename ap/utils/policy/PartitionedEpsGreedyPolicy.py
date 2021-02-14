import numpy as np
from .EpsGreedyPolicy import EpsGreedyPolicy


class PartitionedEpsGreedyPolicy(EpsGreedyPolicy):

    def __init__(self, config, partition):

        super(PartitionedEpsGreedyPolicy, self).__init__(config)

        self.partition = None
        self.blocks = None
        self.action_sets = None

        if partition is not None:
            self.set_partition(partition)

    def act(self, state, qs, timestep, evaluation=False):

        if self.exploration_schedule is not None and np.random.rand() < \
                self.exploration_schedule.value(timestep) and not evaluation:

            block = np.random.choice(self.blocks)
            action = np.random.choice(self.action_sets[block])

        else:

            action = np.argmax(qs)

        return action

    def set_partition(self, partition):

        self.partition = partition
        assert self.partition.dtype == np.int32

        self.blocks = np.unique(self.partition)
        self.action_sets = []

        for b in self.blocks:
            self.action_sets.append(np.where(self.partition == b)[0])
