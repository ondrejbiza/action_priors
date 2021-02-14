import torch
from .DQN import DQN
from ...constants import Constants


class DQNH(DQN):

    def __init__(self, encoder, policy, config, logger):

        super(DQNH, self).__init__(encoder, policy, config, logger)

        c = config
        self.num_meta_actions = c[Constants.NUM_META_ACTIONS]

        # setup meta heads
        if self.dueling:
            self.meta_fc_v, self.meta_fc_a = self.create_double_head_(self.num_meta_actions)
        else:
            self.meta_fc_q = self.create_normal_head_(self.num_meta_actions)

    def forward(self, x):

        x = self.encoder(x)

        if self.dueling:
            q = self.get_dueling_q_(x, self.fc_v, self.fc_a)
            mq = self.get_dueling_q_(x, self.meta_fc_v, self.meta_fc_a)
        else:
            q = self.fc_q(x)
            mq = self.meta_fc_q(x)

        return q, mq

    def act(self, state, timestep, evaluation=False):
        # assume the state has a batch size of one
        with torch.no_grad():
            qs, mqs = self.forward(state)

        qs = qs.cpu().numpy()[0]
        mqs = mqs.cpu().numpy()[0]

        return self.policy.act(state, qs, mqs, timestep, evaluation=evaluation)

    def calculate_qs(self, states, actions):
        # TODO: not sure how this will be used
        qs = self.forward(states)
        qs = qs[list(range(len(actions))), actions.long()]

        return qs

    def calculate_td_error_and_loss(self, states, actions, meta_actions, rewards, dones, target_qs, target_meta_qs,
                                    weights=None):

        qs, mqs = self.calculate_qs(states, actions)
        target = rewards + self.discount * target_qs * (1 - dones.float())
        m_target = rewards + self.discount * target_meta_qs * (1 - dones.float())

        td_error = torch.abs(qs - target)
        loss = self.loss(qs, target)
        m_loss = torch.mean(self.loss(mqs, m_target))

        if weights is None:
            loss = torch.mean(loss)
        else:
            loss = torch.mean(weights * loss)

        return td_error, loss + m_loss

    def remember(self, state, action, meta_action, reward, next_state, done):

        self.replay_buffer.add(
            state, [action, meta_action], reward, next_state, done
        )

