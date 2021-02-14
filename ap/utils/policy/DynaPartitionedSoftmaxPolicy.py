import numpy as np
from .PartitionedSoftmaxPolicy import PartitionedSoftmaxPolicy


class DynaPartitionedSoftmaxPolicy(PartitionedSoftmaxPolicy):

    def __init__(self, config, get_partition):

        super(DynaPartitionedSoftmaxPolicy, self).__init__(config, None)

        self.get_partition = get_partition

    def act(self, state, qs, timestep, evaluation=False):

        self.set_partition(self.get_partition(state))

        if evaluation:
            return np.argmax(qs)

        block = self.select_block_(qs)
        return self.select_action_(qs, block)
