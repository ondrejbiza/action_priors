import numpy.random as npr
import torch


class ReplayBuffer(object):
  def __init__(self, size):
    '''
    Create replay buffer.

    Args:
      - size: Max number of transitions to store in the buffer
    '''
    self._storage = []
    self._max_size = size
    self._next_idx = 0

  def __len__(self):
    return len(self._storage)

  def add(self, obs_t, actions, rewards, obs_tp1, masks):
    '''
    Add experience to replay buffer.

    Args:
      - obs_t: Observations
      - actions: Actions
      - rewards: Reward for taking action in current state
      - obs_tp: Next observations for taking action in current state
      - masks: Episode finished masks
    '''
    for obs_t, action, reward, obs_tp1, mask in zip(obs_t, actions, rewards, obs_tp1, masks):
      data = (obs_t, action, reward, obs_tp1, mask)
      if self._next_idx >= len(self._storage):
        self._storage.append(data)
      else:
        self._storage[self._next_idx] = data
      self._next_idx = (self._next_idx + 1) % self._max_size

  def _encode_sample(self, batch_indexes):
    '''
    Get the batch from a list of indexes

    Args:
      - batch_indexes: Indexes of samples to pull out of buffer

    Returns: (obs_t_batch, action_batch, reward_batch, obs_tp1_batch, mask_batch)
      - obs_t_batch: Torch tensor of observations
      - action_batch: Torch tensor of actions
      - reward_batch: Torch tensor of rewards
      - obs_tp1_batch: Torch tensor of next observations
      - mask_batch: Torch tensor of episode done masks
    '''

    batch = list(zip(*[self._storage[idx] for idx in batch_indexes]))
    obs_t_batch = torch.stack(batch[0])
    action_batch = torch.stack(batch[1])
    reward_batch = torch.stack(batch[2])
    obs_tp1_batch = torch.stack(batch[3])
    mask_batch = torch.stack(batch[4]).long()

    return obs_t_batch, action_batch, reward_batch, obs_tp1_batch, mask_batch

  def sample(self, batch_size):
    '''
    Sample a batch of experiences.

    Args:
      - batch_size: How many transitions to sample.

    Returns: (obs_t_batch, action_batch, reward_batch, obs_tp1_batch, mask_batch)
      - obs_t_batch: Torch tensor of observations
      - action_batch: Torch tensor of actions
      - reward_batch: Torch tensor of rewards
      - obs_tp1_batch: Torch tensor of next observations
      - mask_batch: Torch tensor of episode done masks
    '''
    batch_indexes = npr.choice(self.__len__(), batch_size).tolist()
    return self._encode_sample(batch_indexes)

  def update_montecarlo(self, gamma):
    '''
    Reassign the rewards of all steps in the last episode in the buffer to
    reflect the true experience monte carlo reward.
    '''
    idx = self._next_idx-1
    if idx < 0:
      idx = min(self._max_size,len(self._storage)) - 1

    obs_t, action, reward, obs_tp1, mask = self._storage[idx]
    if not mask.data[0]:
      print("replay_buffer.update_montecarlo ERROR! Last entry into buffer must have a positive done flag!")

    acc_reward = reward
    for i in range(idx-1, -self._max_size, -1):
      ii = i
      if i < 0:
          ii = i+min(self._max_size, len(self._storage))
      obs_t, action, reward, obs_tp1, mask = self._storage[ii]
      if mask.data[0]:
          return
      acc_reward = gamma * acc_reward + reward
      self._storage[ii] = (obs_t, action, acc_reward, obs_tp1, mask)
