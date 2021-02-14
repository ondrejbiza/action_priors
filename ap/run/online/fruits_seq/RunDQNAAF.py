import numpy as np
import torch
from torch import nn
from ....modules.FlatStatesMultipleConcant import FlatStatesMultipleConcat
from ....modules.FCEncoder import FCEncoder
from ....models.dqn.DQN import DQN
from ....run.online.fruits_seq.RunDQN import RunDQN
from ....utils import runner as runner_utils
from ....utils.policy.DynaMaskedEpsGreedyPolicy import DynaMaskedEpsGreedyPolicy
from ....utils.policy.DynaPartitionedSoftmaxPolicy import DynaPartitionedSoftmaxPolicy
from ....constants import Constants


class RunDQNAAF(RunDQN):

    def __init__(self, runner_config, model_config, model, logger):

        super(RunDQNAAF, self).__init__(runner_config, model_config, logger)

        self.model = model

        rc = runner_config
        self.num_actions = rc[Constants.NUM_ACTIONS]
        self.policy_type = rc[Constants.POLICY]

    def create_mask_function_(self):

        def mask(state):
            # the state should be a pytorch tensor with the the batch dimension
            with torch.no_grad():

                pred_probs = self.model(
                    [state[0].repeat((self.num_actions, 1, 1, 1)), state[1].repeat((self.num_actions, 1, 1, 1))],
                    runner_utils.other_to_torch(np.array(list(range(self.num_actions)), dtype=np.long), self.device)
                )
                pred = (pred_probs > 0.0)

            return pred.cpu().numpy().astype(np.bool)

        return mask

    def build_dqn_(self):

        state_shape = [x.shape for x in self.env.get_state()]

        flatten_states = FlatStatesMultipleConcat()
        flatten_states.output_size = int(np.prod(state_shape[0]) + np.prod(state_shape[1]))

        fc_config = {
            Constants.INPUT_SIZE: flatten_states.output_size,
            Constants.NEURONS: [256, 256],
            Constants.USE_BATCH_NORM: False,
            Constants.USE_LAYER_NORM: False,
            Constants.ACTIVATION_LAST: True
        }

        fc_encoder = FCEncoder(fc_config, self.logger)

        encoder = nn.Sequential(flatten_states, fc_encoder)
        encoder.output_size = fc_encoder.output_size

        if self.policy_type == Constants.EPS:
            policy = DynaMaskedEpsGreedyPolicy(self.model_config, self.create_mask_function_())
        elif self.policy_type == Constants.SOFTMAX:
            policy = DynaPartitionedSoftmaxPolicy(self.model_config, self.create_mask_function_())
        else:
            raise ValueError("Wrong policy.")

        dqn = DQN(encoder, policy, self.model_config, self.logger)

        return dqn, encoder
