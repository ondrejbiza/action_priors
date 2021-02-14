import numpy as np
import torch
from torch import nn
from .utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .utils.schedules import LinearSchedule
from ...constants import Constants


class DQN(nn.Module):

    def __init__(self, encoder, policy, config, logger):

        super(DQN, self).__init__()

        self.encoder = encoder
        self.policy = policy
        self.logger = logger

        c = config
        self.num_actions = c[Constants.NUM_ACTIONS]
        self.dueling = c[Constants.DUELING]
        self.discount = c[Constants.DISCOUNT]
        self.prioritized_replay = c[Constants.PRIORITIZED_REPLAY]
        self.exploration_steps = c[Constants.EXPLORATION_STEPS]
        self.prioritized_replay_max_steps = c[Constants.PRIORITIZED_REPLAY_MAX_STEPS]
        self.buffer_size = c[Constants.BUFFER_SIZE]

        self.pr_alpha = 0.6
        self.pr_beta = 0.4
        self.pr_eps = 1e-6
        self.init_explore = 1.0
        self.final_explore = 0.1

        # setup q prediction heads
        if self.dueling:
            self.fc_v, self.fc_a = self.create_double_head_(self.num_actions)
        else:
            self.fc_q = self.create_normal_head_(self.num_actions)

        # setup loss
        self.loss = nn.SmoothL1Loss(reduction="none") # TODO: this might not be exactly huber loss

    def forward(self, x):

        x = self.encoder(x)

        if self.dueling:
            q = self.get_dueling_q_(x, self.fc_v, self.fc_a)
        else:
            q = self.fc_q(x)

        return q

    def act(self, state, timestep, evaluation=False):
        # assume the state has a batch size of one
        with torch.no_grad():
            qs = self.forward(state)

        qs = qs.cpu().numpy()
        qs = qs[0]

        return self.policy.act(state, qs, timestep, evaluation=evaluation)

    def calculate_qs(self, states, actions):

        qs = self.forward(states)
        qs = qs[list(range(len(actions))), actions.long()]

        return qs

    def calculate_td_error_and_loss(self, states, actions, rewards, dones, target_qs, weights=None):

        qs = self.calculate_qs(states, actions)
        target = rewards + self.discount * target_qs * (1 - dones.float())

        td_error = torch.abs(qs - target)
        loss = self.loss(qs, target)

        if weights is None:
            loss = torch.mean(loss)
        else:
            loss = torch.mean(weights * loss)

        return td_error, loss

    def setup_for_training(self):

        self.setup_replay_buffer_()

    def remember(self, state, action, reward, next_state, done):

        self.replay_buffer.add(
            state, action, reward, next_state, done
        )

    def sample_buffer(self, batch_size, timestep):

        if self.prioritized_replay:
            beta = self.beta_schedule.value(timestep)
            states, actions, rewards, next_states, dones, weights, batch_indexes = \
                self.replay_buffer.sample(batch_size, beta)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
            weights, batch_indexes = np.ones_like(rewards), None

        return states, actions, rewards, next_states, dones, weights, batch_indexes

    def update_priorities(self, new_priorities, batch_indexes):

        if not self.prioritized_replay:
            return

        new_priorities += self.pr_eps
        self.replay_buffer.update_priorities(batch_indexes, new_priorities)

    def sync_weights(self, reference_net):

        self.load_state_dict(reference_net.state_dict())

    def create_normal_head_(self, num_actions):

        fc_q = nn.Linear(self.encoder.output_size, num_actions, bias=True)

        nn.init.kaiming_normal_(fc_q.weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(fc_q.bias, 0)

        return fc_q

    def create_double_head_(self, num_actions):

        fc_v = nn.Linear(self.encoder.output_size, 1, bias=True)
        fc_a = nn.Linear(self.encoder.output_size, num_actions, bias=True)

        nn.init.kaiming_normal_(fc_v.weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(fc_v.bias, 0)

        nn.init.kaiming_normal_(fc_a.weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(fc_a.bias, 0)

        return fc_v, fc_a

    def setup_replay_buffer_(self):

        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.pr_alpha)
            self.beta_schedule = LinearSchedule(
                self.prioritized_replay_max_steps, initial_p=self.pr_beta, final_p=self.final_explore
            )
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)
            self.beta_schedule = None

    def get_dueling_q_(self, x, fc_v, fc_a):

        v = fc_v(x)
        a = fc_a(x)
        a = a - a.mean(1)[:, None]
        return v + a

    def save(self, path):

        torch.save(self.state_dict(), path)

    def load(self, path):

        self.load_state_dict(torch.load(path, map_location=next(self.parameters()).device))

    def head_parameters(self):

        if self.dueling:
            return list(self.fc_v.parameters()) + list(self.fc_a.parameters())
        else:
            return list(self.fc_q.parameters())
