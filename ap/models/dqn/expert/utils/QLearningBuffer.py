from copy import deepcopy
import numpy as np
import numpy.random as npr


class Experience:
    def __init__(self, state, image, num_orientation, num_height):
        self.num_orientation = num_orientation
        self.num_height = num_height

        self.image = image
        self.state = state
        self.xy = None
        self.primitive = None

        self.rewards = []
        self.next_states = []
        self.next_obs = []
        self.dones = []
        self.phis = []

    def setXY(self, x, y):
        self.xy = x, y

    def setPXY(self, p, x, y):
        self.primitive = p
        self.xy = x, y

    def addOutcome(self, orientation_id, height_id, reward, next_state, next_obs, done):
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.next_obs.append(next_obs)
        self.dones.append(done)
        self.phis.append(orientation_id*self.num_height+height_id)


class QLearningBuffer:
    def __init__(self, size):
        self._storage = []
        self._max_size = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._max_size

    def sample(self, batch_size):
        batch_indexes = npr.choice(self.__len__(), batch_size).tolist()
        batch = [self._storage[idx] for idx in batch_indexes]
        return batch

    def getSaveState(self):
        return {
            'storage': self._storage,
            'max_size': self._max_size,
            'next_idx': self._next_idx
        }

    def loadFromState(self, save_state):
        self._storage = save_state['storage']
        self._max_size = save_state['max_size']
        self._next_idx = save_state['next_idx']


class QLearningBufferExpert(QLearningBuffer):
    def __init__(self, size):
        super().__init__(size)
        self._expert_idx = []

    def add(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
            idx = len(self._storage)-1
            self._next_idx = (self._next_idx + 1) % self._max_size
        else:
            self._storage[self._next_idx] = data
            idx = deepcopy(self._next_idx)
            self._next_idx = (self._next_idx + 1) % self._max_size
            while self._storage[self._next_idx].expert:
                self._next_idx = (self._next_idx + 1) % self._max_size
        if data.expert:
            self._expert_idx.append(idx)

    def sample(self, batch_size):
        if len(self._expert_idx) < batch_size/2 or len(self._storage) - len(self._expert_idx) < batch_size/2:
            return super().sample(batch_size)
        expert_indexes = npr.choice(self._expert_idx, int(batch_size / 2)).tolist()
        non_expert_mask = np.ones(self.__len__(), dtype=np.bool)
        non_expert_mask[np.array(self._expert_idx)] = 0
        non_expert_indexes = npr.choice(np.arange(self.__len__())[non_expert_mask], int(batch_size/2)).tolist()
        batch_indexes = expert_indexes + non_expert_indexes
        batch = [self._storage[idx] for idx in batch_indexes]
        return batch

    def getSaveState(self):
        save_state = super().getSaveState()
        save_state['expert_idx'] = self._expert_idx
        return save_state

    def loadFromState(self, save_state):
        super().loadFromState(save_state)
        self._expert_idx = save_state['expert_idx']
