import numpy as np
from scipy.special import softmax
from .SoftmaxPolicy import SoftmaxPolicy


class PartitionedSoftmaxPolicy(SoftmaxPolicy):

    def __init__(self, config, partition):

        super(PartitionedSoftmaxPolicy, self).__init__(config)

        self.partition = None
        self.blocks = None
        self.action_sets = None

        if partition is not None:
            self.set_partition(partition)

    def act(self, state, qs, timestep, evaluation=False):

        if evaluation:
            return np.argmax(qs)

        block = self.select_block_(qs)
        return self.select_action_(qs, block)

    def select_block_(self, qs):

        agg_qs = []

        for i in range(len(self.blocks)):
            tmp_agg_qs = np.mean(qs[self.action_sets[i]])
            agg_qs.append(tmp_agg_qs)

        agg_s = softmax(agg_qs)
        return np.random.choice(list(range(len(self.blocks))), p=agg_s)

    def select_action_(self, qs, block):

        block_qs = qs[self.action_sets[block]]
        block_s = softmax(block_qs)

        return np.random.choice(self.action_sets[block], p=block_s)

    def set_partition(self, partition):

        self.partition = partition

        if self.partition.dtype == np.bool:
            self.partition = self.partition.astype(np.int32)

        assert self.partition.dtype == np.int32

        self.blocks = np.unique(self.partition)
        self.action_sets = []

        for b in self.blocks:
            self.action_sets.append(np.where(self.partition == b)[0])
