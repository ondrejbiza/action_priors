from .FlatStatesMultipleConcant import FlatStatesMultipleConcat
from .FlatStatesOneHotActions import FlatStatesOneHotActions


class FlatStatesMultipleConcatOneHotActions(FlatStatesOneHotActions, FlatStatesMultipleConcat):

    def __init__(self, num_actions):

        super(FlatStatesMultipleConcatOneHotActions, self).__init__(num_actions)

    def forward(self, x):

        states = x[0]
        actions = x[1]

        flats = []

        for state in states:
            flats.append(self.flatten_states_(state))

        flat_states = self.concat_(flats)
        one_hot_actions = self.one_hot_actions_(actions)

        return self.concat_states_and_actions_(flat_states, one_hot_actions)
