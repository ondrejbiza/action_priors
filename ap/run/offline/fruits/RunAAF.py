import copy as cp
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from ....modules.FlatStatesOneHotActions import FlatStatesOneHotActions
from ....modules.FCEncoder import FCEncoder
from ....models.homo.AAFFruits import AAFFruits
from ....utils.dataset import ArrayDataset
from ....constants import Constants
from ....envs.fruits import Fruits
from ....utils import runner as runner_utils


class RunAAF:

    def __init__(self, runner_config, model_config, logger):

        super(RunAAF, self).__init__()

        rc = runner_config
        self.ignore_list = rc[Constants.IGNORE_LIST]
        self.load_path = rc[Constants.LOAD_PATH]
        self.encoder_learning_rate = rc[Constants.ENCODER_LEARNING_RATE]
        self.encoder_weight_decay = rc[Constants.ENCODER_WEIGHT_DECAY]
        self.validation_fraction = rc[Constants.VALIDATION_FRACTION]
        self.batch_size = rc[Constants.BATCH_SIZE]
        self.discount = rc[Constants.DISCOUNT]
        self.plot_dataset_examples = rc[Constants.PLOT_DATASET_EXAMPLES]
        self.plot_results = rc[Constants.PLOT_RESULTS]
        self.num_training_steps = rc[Constants.NUM_TRAINING_STEPS]
        self.device = rc[Constants.DEVICE]
        self.model_load_path = rc[Constants.MODEL_LOAD_PATH]
        self.model_save_path = rc[Constants.MODEL_SAVE_PATH]
        self.input_size = rc[Constants.INPUT_SIZE]
        self.num_actions = rc[Constants.NUM_ACTIONS]

        self.model_config = model_config
        self.logger = logger
        self.best_model = None
        self.best_val_loss = None

        self.load_dataset_()

        if self.plot_dataset_examples:
            self.plot_dataset_examples_()

        self.build_model_()

    def train_model(self):

        if self.model_load_path is not None:
            # no training when loading model
            self.model.load(self.model_load_path)
            self.model.eval()
            return

        self.model.train()

        params = list(self.encoder.parameters()) + [*self.model.fc_prob.parameters()]
        opt = optim.Adam(params, self.encoder_learning_rate, weight_decay=self.encoder_weight_decay)

        losses = []
        valid_losses = []

        for key, value in self.model.state_dict().items():
            print(key, value.size())

        for training_step in range(self.num_training_steps):

            epoch_step = training_step % self.epoch_size

            if epoch_step == 0:

                self.dataset.shuffle()

                if len(losses) > 0:
                    tmp_losses = losses[- self.epoch_size:]
                    tmp_losses = np.stack(tmp_losses, axis=0).mean(0)
                    print("({:d}, {:d}) total loss {:.4f}".format(
                        training_step // self.epoch_size, training_step, tmp_losses
                    ))

                tmp_valid_loss = self.validate_()
                valid_losses.append(tmp_valid_loss)
                self.maybe_save_best_model_(tmp_valid_loss)

            states, actions, qs = self.get_batch_(epoch_step)

            opt.zero_grad()

            total_loss = self.model.compute_loss(
                states, actions, qs
            )
            total_loss.backward()
            opt.step()
            losses.append(total_loss.detach().cpu().numpy())

        self.load_best_model_()

        if self.model_save_path is not None:
            torch.save(self.model.state_dict(), self.model_save_path)

        losses = np.stack(losses, axis=0)
        valid_losses = np.stack(valid_losses, axis=0)

        if self.plot_results:
            self.plot_losses_panel_(losses, valid_losses)

        self.final_valid_loss = self.validate_()

        self.model.eval()

    def maybe_save_best_model_(self, val_loss):

        if self.best_val_loss is None or self.best_val_loss > val_loss:
            self.best_val_loss = val_loss
            self.best_model = cp.deepcopy(self.model.state_dict())

    def load_best_model_(self):

        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
        else:
            self.logger.warning("Best model not saved.")

    @torch.no_grad()
    def validate_(self):

        self.model.eval()

        losses = []

        # throws away a bit of data if validation set size % batch size != 0
        num_steps = int(len(self.valid_dataset[Constants.STATES]) // self.batch_size)

        for step in range(num_steps):

            states, actions, qs = self.get_batch_(step, validation=True)
            total_loss = self.model.compute_loss(states, actions, qs)
            losses.append(total_loss.cpu().numpy())

        self.model.train()

        losses = np.mean(np.stack(losses, axis=0), axis=0)

        return losses

    def get_accuracy_(self, validation=False):

        if validation:
            states = self.valid_dataset[Constants.STATES]
            actions = self.valid_dataset[Constants.ACTIONS]
            qs = self.valid_dataset[Constants.QS]
        else:
            states = self.dataset[Constants.STATES]
            actions = self.dataset[Constants.ACTIONS]
            qs = self.dataset[Constants.QS]

        num = int(np.ceil(len(states) / self.batch_size))
        accs = []

        for i in range(num):

            batch_states = states[i * self.batch_size: (i + 1) * self.batch_size]
            batch_states = runner_utils.states_to_torch(batch_states, self.device)

            batch_actions = actions[i * self.batch_size: (i + 1) * self.batch_size]
            batch_actions = runner_utils.other_to_torch(batch_actions, self.device).long()

            batch_qs = qs[i * self.batch_size: (i + 1) * self.batch_size]
            batch_qs = runner_utils.other_to_torch(batch_qs, self.device).int()

            with torch.no_grad():
                res = self.model.get_accuracy(batch_states, batch_actions, batch_qs)
                res = res.cpu().numpy()

            accs.append(res)

        return np.mean(accs)

    def evaluate(self):

        self.final_valid_accuracy = self.get_accuracy_(validation=True)
        self.logger.info("final valid accuracy: {:.2f}%".format(self.final_valid_accuracy * 100))

    def plot_losses_panel_(self, losses, validation_losses):

        plt.subplot(1, 2, 1)
        runner_utils.plot_losses([losses], ["AAF loss"])
        runner_utils.plot_losses(
            [validation_losses], ["AAF valid loss"], validation=True, epoch_size=self.epoch_size
        )
        plt.subplot(1, 2, 2)
        runner_utils.plot_log_losses([np.abs(losses)], ["AAF log loss"])
        runner_utils.plot_log_losses(
            [np.abs(validation_losses)], ["AAF log valid loss"], validation=True, epoch_size=self.epoch_size
        )

        plt.show()

    def plot_prediction_examples_(self):

        for i in range(100):

            state = self.dataset[Constants.STATES][i]
            next_state = self.dataset[Constants.NEXT_STATES][i]
            action = self.dataset[Constants.ACTIONS][i]
            reward = self.dataset[Constants.REWARDS][i]

            with torch.no_grad():
                pred_qs = self.model.get_prediction(
                    runner_utils.states_to_torch(np.array([state] * self.num_actions, dtype=np.float32), self.device),
                    runner_utils.other_to_torch(np.array(list(range(self.num_actions)), dtype=np.long), self.device),
                    no_sample=True, hard=True
                )
            pred_qs = pred_qs.cpu().numpy()
            size = int(np.sqrt(pred_qs.shape[0]))

            print("action: {:d}, reward: {:.2f}".format(action, reward))

            plt.figure(figsize=(16, 8))

            plt.subplot(1, 3, 1)
            plt.imshow(Fruits.state_to_image(state))

            plt.subplot(1, 3, 2)
            plt.imshow(Fruits.state_to_image(next_state))

            plt.subplot(1, 3, 3)
            plt.imshow(pred_qs.reshape((size, size)))

            plt.show()

    def get_batch_(self, epoch_step, validation=False):

        b = np.index_exp[epoch_step * self.batch_size: (epoch_step + 1) * self.batch_size]

        if validation:
            states = self.valid_dataset[Constants.STATES][b]
            actions = self.valid_dataset[Constants.ACTIONS][b]
            qs = self.valid_dataset[Constants.QS][b]
        else:
            states = self.dataset[Constants.STATES][b]
            actions = self.dataset[Constants.ACTIONS][b]
            qs = self.dataset[Constants.QS][b]

        if len(qs.shape) == 2:
            qs = qs[list(range(len(actions))), actions]

        return runner_utils.states_to_torch(states, self.device), \
            runner_utils.other_to_torch(actions, self.device).long(), \
            runner_utils.other_to_torch(qs, self.device)

    def build_model_(self):

        reshape_encoder = FlatStatesOneHotActions(self.num_actions)

        fc_config = {
            Constants.INPUT_SIZE: int(np.prod(self.input_size)) + self.num_actions,
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

    def load_dataset_(self):

        self.dataset = ArrayDataset(None)
        self.dataset.load_hdf5(self.load_path)
        self.dataset.shuffle()

        tasks = runner_utils.parse_task_string(
            self.dataset.metadata[Constants.TASK_LIST], return_char_list=False
        )
        if self.ignore_list is not None:

            to_delete = []

            for task in self.ignore_list:
                indices = list(np.where(self.dataset[Constants.TASK_INDEX] == task))
                to_delete += indices

            self.dataset.delete(to_delete)
            self.delete_qs_for_tasks_()

            self.logger.info("deleted tasks: {:s}".format(
                str([task for i, task in enumerate(tasks) if i in self.ignore_list]))
            )
            self.logger.info("active tasks: {:s}".format(
                str([task for i, task in enumerate(tasks) if i not in self.ignore_list]))
            )
        else:
            self.logger.info(
                "active tasks: {:s}".format(str(tasks))
            )

        opt_actions = np.argmax(self.dataset[Constants.QS], axis=2)
        is_opt_action = np.any(opt_actions == self.dataset[Constants.ACTIONS][:, np.newaxis], axis=1)
        self.dataset[Constants.QS] = is_opt_action.astype(np.float32)

        self.num_samples = self.dataset.size
        self.valid_samples = int(self.num_samples * self.validation_fraction)
        self.valid_dataset = self.dataset.split(self.valid_samples)

        self.epoch_size = self.dataset[Constants.STATES].shape[0] // self.batch_size

    def delete_qs_for_tasks_(self):

        valid_indices = [i for i in range(self.dataset[Constants.QS].shape[1]) if i not in self.ignore_list]
        self.dataset[Constants.QS] = self.dataset[Constants.QS][:, valid_indices, :]
        self.logger.info("new qs shape: {:s}, valid indices: {:s}".format(
            str(self.dataset[Constants.QS].shape), str(valid_indices)
        ))

    def plot_dataset_examples_(self):

        for i in range(100):

            state = self.dataset[Constants.STATES][i]
            next_state = self.dataset[Constants.NEXT_STATES][i]
            action = self.dataset[Constants.ACTIONS][i]
            reward = self.dataset[Constants.REWARDS][i]
            qs = self.dataset[Constants.QS][i]

            print("action: {:d}, reward: {:.2f}".format(action, reward))

            plt.figure(figsize=(16, 8))
            plt.subplot(1, 3, 1)
            plt.imshow(Fruits.state_to_image(state))
            plt.subplot(1, 3, 2)
            plt.imshow(Fruits.state_to_image(next_state))
            plt.subplot(1, 3, 3)
            size = int(np.sqrt(qs.shape[0]))
            plt.imshow(qs.reshape((size, size)))
            plt.show()
