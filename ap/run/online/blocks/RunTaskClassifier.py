import os
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import torch
from torch import nn
from torch import optim
from ....constants import Constants
from ....utils.dataset import ArrayDataset
from ....utils import runner as runner_utils
from ....utils.result import Result
from ....models.misc.SoftmaxClassifier import SoftmaxClassifier
from ....modules.View import View
from ....modules.ConvEncoder import ConvEncoder
from ....modules.FCEncoder import FCEncoder
from ....modules.SplitConcat import SplitConcat


class RunTaskClassifier:

    def __init__(self, runner_config, logger, build=True):

        self.logger = logger

        rc = runner_config

        self.dataset_load_path = rc[Constants.DATASET_LOAD_PATH]
        self.validation_fraction = rc[Constants.VALIDATION_FRACTION]
        self.validation_freq = rc[Constants.VALIDATION_FREQ]
        self.batch_size = rc[Constants.BATCH_SIZE]
        self.device = rc[Constants.DEVICE]
        self.learning_rate = rc[Constants.LEARNING_RATE]
        self.weight_decay = rc[Constants.WEIGHT_DECAY]
        self.num_training_steps = rc[Constants.NUM_TRAINING_STEPS]
        self.plot_results = rc[Constants.PLOT_RESULTS]
        self.num_tasks = rc[Constants.NUM_TASKS]

        self.best_val_loss, self.best_model = None, None

        if build:
            self.load_dataset_()
            self.build_model_()
            self.build_optimizer_()

            if self.validation_freq is None:
                self.validation_freq = self.epoch_size

    def train(self):

        self.model.train()
        result = Result()
        result.register(Constants.TOTAL_LOSS)
        result.register(Constants.ACCURACY)
        result.register(Constants.TOTAL_VALID_LOSS)
        result.register(Constants.VALID_ACCURACY)

        for training_step in range(self.num_training_steps):

            epoch_step = training_step % self.epoch_size

            if epoch_step == 0:

                self.dataset.shuffle()

            if training_step % self.validation_freq == 0:

                valid_loss, valid_acc = self.validate()
                self.maybe_save_best_model_(valid_loss)
                result.add(Constants.TOTAL_VALID_LOSS, valid_loss)
                result.add(Constants.VALID_ACCURACY, valid_acc)
                self.logger.info("validation complete")

            if training_step % 100 == 0:

                self.logger.info("step {:d}".format(training_step))

            obs, hand_obs, task_indices = self.get_batch_(epoch_step)

            self.opt.zero_grad()
            loss, acc = self.model.compute_loss_and_accuracy([obs, hand_obs], task_indices)
            loss.backward()
            self.opt.step()

            result.add_pytorch(Constants.TOTAL_LOSS, loss)
            result.add(Constants.ACCURACY, acc)

        self.load_best_model_()

        losses = np.stack(result[Constants.TOTAL_LOSS], axis=0)
        valid_losses = np.stack(result[Constants.TOTAL_VALID_LOSS], axis=0)
        acc = np.stack(result[Constants.ACCURACY], axis=0)
        valid_acc = np.stack(result[Constants.VALID_ACCURACY], axis=0)

        if self.plot_results:
            self.plot_losses_panel_(losses, valid_losses, acc, valid_acc)

        self.final_valid_loss = self.validate()

        self.model.eval()

    @torch.no_grad()
    def validate(self):

        was_training = self.model.training

        self.model.eval()
        result = Result()
        result.register(Constants.TOTAL_LOSS)
        result.register(Constants.ACCURACY)

        # throws away a bit of data if validation set size % batch size != 0
        num_steps = int(len(self.valid_dataset[Constants.OBS]) // self.batch_size)

        for step in range(num_steps):

            obs, hand_obs, task_indices = self.get_batch_(step, validation=True)

            loss, acc = self.model.compute_loss_and_accuracy([obs, hand_obs], task_indices)
            result.add_pytorch(Constants.TOTAL_LOSS, loss)
            result.add(Constants.ACCURACY, acc)

        if was_training:
            self.model.train()

        return result.mean(Constants.TOTAL_LOSS), result.mean(Constants.ACCURACY)

    @torch.no_grad()
    def get_predictions(self, dataset):

        num = int(np.ceil(dataset.size / self.batch_size))
        probs_list = []

        for i in range(num):
            b = np.index_exp[i * self.batch_size: (i + 1) * self.batch_size]

            obs = dataset[Constants.OBS][b]
            hand_obs = dataset[Constants.HAND_OBS][b]

            obs = runner_utils.other_to_torch(obs[:, np.newaxis, :, :], self.device)
            hand_obs = runner_utils.other_to_torch(hand_obs[:, np.newaxis, :, :], self.device)

            probs = self.model.get_prediction([obs, hand_obs], logits=False, hard=False)
            probs = probs.cpu().numpy()
            probs_list.append(probs)

        return np.concatenate(probs_list, axis=0)

    def maybe_save_best_model_(self, val_loss):

        if self.best_val_loss is None or self.best_val_loss > val_loss:
            self.best_val_loss = val_loss
            self.best_model = cp.deepcopy(self.model.state_dict())

    def load_best_model_(self):

        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
        else:
            self.logger.warning("Best model not saved.")

    def plot_losses_panel_(self, total_loss, total_validation_loss, accuracy, validation_accuracy):

        plt.figure(figsize=(8, 6))

        plt.subplot(3, 1, 1)
        runner_utils.plot_losses([total_loss], ["loss"])
        runner_utils.plot_losses([total_validation_loss], ["valid loss"], validation=True, epoch_size=self.validation_freq)

        plt.subplot(3, 1, 2)
        runner_utils.plot_log_losses([total_loss], ["log loss"])
        runner_utils.plot_log_losses(
            [total_validation_loss], ["log valid loss"], validation=True, epoch_size=self.validation_freq
        )

        plt.subplot(3, 1, 3)
        runner_utils.plot_losses([accuracy], ["accuracy"])
        runner_utils.plot_losses([validation_accuracy], ["valid accuracy"], validation=True, epoch_size=self.validation_freq)

        plt.show()

    def get_batch_(self, epoch_step, validation=False):

        b = np.index_exp[epoch_step * self.batch_size: (epoch_step + 1) * self.batch_size]

        if validation:
            obs = self.valid_dataset[Constants.OBS][b]
            hand_obs = self.valid_dataset[Constants.HAND_OBS][b]
            task_indices = self.valid_dataset[Constants.TASK_INDEX][b]
        else:
            obs = self.dataset[Constants.OBS][b]
            hand_obs = self.dataset[Constants.HAND_OBS][b]
            task_indices = self.dataset[Constants.TASK_INDEX][b]

        return runner_utils.other_to_torch(obs[:, np.newaxis, :, :], self.device), \
            runner_utils.other_to_torch(hand_obs[:, np.newaxis, :, :], self.device), \
            runner_utils.other_to_torch(task_indices.astype(np.long), self.device)

    def load_dataset_(self):

        self.dataset = ArrayDataset(None)
        self.dataset.load_hdf5(self.dataset_load_path)
        self.dataset.shuffle()

        self.num_samples = self.dataset.size
        self.valid_samples = int(self.num_samples * self.validation_fraction)
        self.valid_dataset = self.dataset.split(self.valid_samples)

        self.epoch_size = self.dataset[Constants.OBS].shape[0] // self.batch_size

    def build_model_(self):

        # encodes obs of shape Bx1x90x90 into Bx128x5x5
        conv_obs = ConvEncoder({
            Constants.INPUT_SIZE: [90, 90, 1],
            Constants.FILTER_SIZES: [3, 3, 3, 3],
            Constants.FILTER_COUNTS: [32, 64, 128, 128],
            Constants.STRIDES: [2, 2, 2, 2],
            Constants.USE_BATCH_NORM: True,
            Constants.ACTIVATION_LAST: True,
            Constants.FLAT_OUTPUT: False
        }, self.logger)
        # average pool Bx128x5x5 into Bx128x1x1 and reshape that into Bx128
        conv_obs_avg_pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=0)
        conv_obs_view = View([128])
        conv_obs_encoder = nn.Sequential(conv_obs, conv_obs_avg_pool, conv_obs_view)

        # encodes hand obs of shape Bx1x24x24 into Bx128x1x1
        conv_hand_obs = ConvEncoder({
            Constants.INPUT_SIZE: [24, 24, 1],
            Constants.FILTER_SIZES: [3, 3, 3, 3],
            Constants.FILTER_COUNTS: [32, 64, 128, 128],
            Constants.STRIDES: [2, 2, 2, 2],
            Constants.USE_BATCH_NORM: True,
            Constants.ACTIVATION_LAST: True,
            Constants.FLAT_OUTPUT: False
        }, self.logger)
        # reshape Bx128x1x1 into Bx128
        conv_hand_obs_view = View([128])
        conv_hand_obs_encoder = nn.Sequential(conv_hand_obs, conv_hand_obs_view)
        # gets [obs, hand_obs], runs that through their respective encoders
        # and then concats [Bx128, Bx128] into Bx256
        conv_encoder = SplitConcat([conv_obs_encoder, conv_hand_obs_encoder], 1)

        fc = FCEncoder({
            Constants.INPUT_SIZE: 256,
            Constants.NEURONS: [256, 256],
            Constants.USE_BATCH_NORM: True,
            Constants.USE_LAYER_NORM: False,
            Constants.ACTIVATION_LAST: True
        }, self.logger)

        self.encoder = nn.Sequential(conv_encoder, fc)

        self.encoder.output_size = 256

        self.model = SoftmaxClassifier(self.encoder, self.num_tasks)
        self.model.to(self.device)
        self.model.train()

    def build_optimizer_(self):

        self.params = self.encoder.parameters()
        self.logger.info("num. parameter tensors: {:d}".format(len(list(self.encoder.parameters()))))
        self.opt = optim.Adam(self.params, lr=self.learning_rate, weight_decay=self.weight_decay)

    def plot_dataset_examples(self):

        for i in range(100):

            obs = self.dataset[Constants.OBS][i]
            hand_obs = self.dataset[Constants.HAND_OBS][i]
            hand_bit = self.dataset[Constants.HAND_BITS][i]
            task_index = self.dataset[Constants.TASK_INDEX][i]

            print("hand bit: {:d}, task index: {:d}".format(hand_bit, task_index))

            plt.figure(figsize=(8, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(obs)
            plt.subplot(1, 2, 2)
            plt.imshow(hand_obs)

            plt.tight_layout()
            plt.show()

    @torch.no_grad()
    def plot_prediction_examples(self):

        for i in range(100):
            obs = self.valid_dataset[Constants.OBS][i]
            hand_obs = self.valid_dataset[Constants.HAND_OBS][i]
            hand_bit = self.valid_dataset[Constants.HAND_BITS][i]
            task_index = self.valid_dataset[Constants.TASK_INDEX][i]

            pred = self.model(
                [runner_utils.other_to_torch(obs[None, None], self.device),
                 runner_utils.other_to_torch(hand_obs[None, None], self.device)]
            ).detach().cpu().numpy()[0]
            pred_hard = np.argmax(pred, axis=0)

            print("hand_bit: {:d}, task_index: {:d}, pred: {:d}".format(hand_bit, task_index, pred_hard))
            print("probs:" + str(softmax(pred)))

            plt.figure(figsize=(8, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(obs)

            plt.subplot(1, 2, 2)
            plt.imshow(hand_obs)

            plt.show()

    def save_model(self, save_path):

        dir_path = os.path.dirname(save_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path):

        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
