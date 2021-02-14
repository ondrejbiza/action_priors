import numpy as np
import torch
from torch import nn
from ....modules.ConvEncoder import ConvEncoder
from ....modules.FCEncoder import FCEncoder
from ....models.dqn.DQN import DQN
from ....run.online.fruits.RunDQN import RunDQN
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
                    state.repeat((self.num_actions, 1, 1, 1)),
                    runner_utils.other_to_torch(np.array(list(range(self.num_actions)), dtype=np.long), self.device)
                )
                pred = (pred_probs > 0.0)

            return pred.cpu().numpy().astype(np.bool)

        return mask

    def build_dqn_conv_fc_(self):

        conv_config = {
            Constants.INPUT_SIZE: self.env.get_state().shape,
            Constants.FILTER_SIZES: [],
            Constants.FILTER_COUNTS: [],
            Constants.STRIDES: [],
            Constants.USE_BATCH_NORM: False,
            Constants.ACTIVATION_LAST: True,
            Constants.FLAT_OUTPUT: True
        }

        conv_encoder = ConvEncoder(conv_config, self.logger)

        fc_config = {
            Constants.INPUT_SIZE: conv_encoder.output_size,
            Constants.NEURONS: [256, 256],
            Constants.USE_BATCH_NORM: False,
            Constants.USE_LAYER_NORM: False,
            Constants.ACTIVATION_LAST: True
        }

        fc_encoder = FCEncoder(fc_config, self.logger)

        encoder = nn.Sequential(conv_encoder, fc_encoder)
        encoder.input_size = conv_encoder.input_size
        encoder.output_size = fc_encoder.output_size

        if self.policy_type == Constants.EPS:
            policy = DynaMaskedEpsGreedyPolicy(self.model_config, self.create_mask_function_())
        elif self.policy_type == Constants.SOFTMAX:
            policy = DynaPartitionedSoftmaxPolicy(self.model_config, self.create_mask_function_())
        else:
            raise ValueError("Wrong policy.")

        dqn = DQN(encoder, policy, self.model_config, self.logger)

        return dqn, encoder
