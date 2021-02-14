import torch
from torch import nn


class FlatStatesOneHotActions(nn.Module):

    def __init__(self, num_actions):

        super(FlatStatesOneHotActions, self).__init__()
        self.num_actions = num_actions

    def forward(self, x):

        assert len(x) == 2
        states = x[0]
        actions = x[1]

        flat_states = self.flatten_states_(states)
        one_hot_actions = self.one_hot_actions_(actions)

        return self.concat_states_and_actions_(flat_states, one_hot_actions)

    def flatten_states_(self, states):

        return torch.flatten(states, 1)

    def one_hot_actions_(self, actions):

        shape = (actions.size()[0], self.num_actions)
        one_hot = torch.zeros(shape, dtype=torch.float32, device=actions.device)
        one_hot[list(range(len(actions))), actions] = 1.0
        return one_hot

    def concat_states_and_actions_(self, states, actions):

        return torch.cat((states, actions), dim=1)
