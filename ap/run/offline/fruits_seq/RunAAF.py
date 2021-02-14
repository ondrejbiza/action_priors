import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from ....envs.fruits_seq import FruitsSeq
from ....models.homo.AAFFruits import AAFFruits
from ....modules.FlatStatesMultipleConcatOneHotActions import FlatStatesMultipleConcatOneHotActions
from ....modules.FCEncoder import FCEncoder
from ....run.offline.fruits.RunAAF import RunAAF
from ....utils import runner as runner_utils
from ....constants import Constants


class RunAAFSeq(RunAAF):

    def get_batch_(self, epoch_step, validation=False):

        b = np.index_exp[epoch_step * self.batch_size: (epoch_step + 1) * self.batch_size]

        if validation:
            states = self.valid_dataset[Constants.STATES][b]
            hand_states = self.valid_dataset[Constants.HAND_STATES][b]
            actions = self.valid_dataset[Constants.ACTIONS][b]
            qs = self.valid_dataset[Constants.QS][b]
        else:
            states = self.dataset[Constants.STATES][b]
            hand_states = self.dataset[Constants.HAND_STATES][b]
            actions = self.dataset[Constants.ACTIONS][b]
            qs = self.dataset[Constants.QS][b]

        if len(qs.shape) == 2:
            qs = qs[list(range(len(actions))), actions]

        return [runner_utils.other_to_torch(states, self.device),
                runner_utils.other_to_torch(hand_states, self.device)], \
            runner_utils.other_to_torch(actions, self.device).long(), \
            runner_utils.other_to_torch(qs, self.device)

    def get_accuracy_(self, validation=False):

        if validation:
            num = len(self.valid_dataset[Constants.STATES]) // self.batch_size
        else:
            num = len(self.dataset[Constants.STATES]) // self.batch_size

        accs = []

        for i in range(num):

            states, actions, qs = self.get_batch_(i, validation=validation)
            qs = qs.int() # TODO: ???

            with torch.no_grad():
                res = self.model.get_accuracy(states, actions, qs)
                res = res.cpu().numpy()

            accs.append(res)

        return np.mean(accs)

    def build_model_(self):

        reshape_encoder = FlatStatesMultipleConcatOneHotActions(self.num_actions)

        fc_config = {
            Constants.INPUT_SIZE: int(np.prod(self.input_size[0])) + int(np.prod(self.input_size[1])) + self.num_actions,
            Constants.NEURONS: [256, 256, 32],
            Constants.USE_BATCH_NORM: False,
            Constants.USE_LAYER_NORM: False,
            Constants.ACTIVATION_LAST: True
        }

        fc_encoder = FCEncoder(fc_config, self.logger)

        self.encoder = nn.Sequential(reshape_encoder, fc_encoder)
        self.encoder.output_size = fc_encoder.output_size

        self.model = AAFFruits(self.encoder, self.model_config, self.logger)
        self.model.to(self.device)

    def plot_dataset_examples_(self):

        for i in range(100):

            state = self.dataset[Constants.STATES][i]
            hand_state = self.dataset[Constants.HAND_STATES][i]
            next_state = self.dataset[Constants.NEXT_STATES][i]
            next_hand_state = self.dataset[Constants.NEXT_HAND_STATES][i]
            action = self.dataset[Constants.ACTIONS][i]
            reward = self.dataset[Constants.REWARDS][i]
            is_opt = self.dataset[Constants.QS][i]

            print("hand state: {:s}".format(str(hand_state)))
            print("next hand state: {:s}".format(str(next_hand_state)))
            print("action: {:d}, reward: {:.2f}, is opt {:.2f}".format(action, reward, is_opt))

            plt.figure(figsize=(16, 8))

            plt.subplot(1, 2, 1)
            plt.imshow(FruitsSeq.state_to_image(state))
            plt.subplot(1, 2, 2)
            plt.imshow(FruitsSeq.state_to_image(next_state))

            plt.show()

    def plot_prediction_examples_(self):

        for i in range(100):

            state = self.dataset[Constants.STATES][i]
            hand_state = self.dataset[Constants.HAND_STATES][i]
            next_state = self.dataset[Constants.NEXT_STATES][i]
            next_hand_state = self.dataset[Constants.NEXT_HAND_STATES][i]
            action = self.dataset[Constants.ACTIONS][i]
            reward = self.dataset[Constants.REWARDS][i]

            with torch.no_grad():
                pred_qs = self.model.get_prediction(
                    [runner_utils.other_to_torch(np.array([state] * self.num_actions, dtype=np.float32), self.device),
                     runner_utils.other_to_torch(np.array([hand_state] * self.num_actions, dtype=np.float32), self.device)],
                    runner_utils.other_to_torch(np.array(list(range(self.num_actions)), dtype=np.long), self.device),
                    no_sample=True, hard=True
                )
            pred_qs = pred_qs.cpu().numpy()
            size = int(np.sqrt(pred_qs.shape[0]))

            print("hand state: {:s}".format(str(hand_state)))
            print("next hand state: {:s}".format(str(next_hand_state)))
            print("action: {:d}, reward: {:.2f}".format(action, reward))

            plt.figure(figsize=(16, 8))

            plt.subplot(1, 3, 1)
            plt.imshow(FruitsSeq.state_to_image(state))

            plt.subplot(1, 3, 2)
            plt.imshow(FruitsSeq.state_to_image(next_state))

            plt.subplot(1, 3, 3)
            plt.imshow(pred_qs.reshape((size, size)))

            plt.show()
