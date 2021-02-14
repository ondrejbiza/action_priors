import copy as cp
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import torch
from torch import optim
from ....constants import Constants
from ....models.homo.AAF import AAF
from ....modules.hand_obs.ResUCatEncoder import ResUCatEncoder
from ....utils.dataset import ArrayDataset
from ....utils import runner as runner_utils


class RunAAF:

    def __init__(self, runner_config, model_config, logger):

        self.model_config = model_config
        self.logger = logger

        rc = runner_config
        self.pos_amb_labels_load_path = rc[Constants.POS_AMB_LABELS_LOAD_PATH]
        self.use_amb_labels = rc[Constants.USE_AMB_LABELS]
        self.ignore_list = rc[Constants.IGNORE_LIST]
        self.load_path = rc[Constants.LOAD_PATH]
        self.encoder_learning_rate = rc[Constants.ENCODER_LEARNING_RATE]
        self.encoder_weight_decay = rc[Constants.ENCODER_WEIGHT_DECAY]
        self.validation_fraction = rc[Constants.VALIDATION_FRACTION]
        self.batch_size = rc[Constants.BATCH_SIZE]
        self.plot_dataset_examples = rc[Constants.PLOT_DATASET_EXAMPLES]
        self.num_training_steps = rc[Constants.NUM_TRAINING_STEPS]
        self.device = rc[Constants.DEVICE]
        self.model_load_path = rc[Constants.MODEL_LOAD_PATH]
        self.model_save_path = rc[Constants.MODEL_SAVE_PATH]
        self.plot_results = rc[Constants.PLOT_RESULTS]
        self.input_size = rc[Constants.INPUT_SIZE]
        self.num_actions = rc[Constants.NUM_ACTIONS]
        self.animate_latent = rc[Constants.ANIMATE_LATENT]
        self.limit = rc[Constants.LIMIT]

        self.final_valid_accuracy = None
        self.final_valid_loss = None
        self.best_val_loss = None
        self.best_model = None
        self.dataset = None
        self.model = None
        self.encoder = None

        self.load_dataset_()

        if self.plot_dataset_examples:
            self.plot_dataset_examples_()

        self.build_model_()

    def evaluate(self):

        self.final_valid_accuracy = self.get_accuracy_(validation=True)
        self.logger.info("final valid accuracy: {:.2f}%".format(self.final_valid_accuracy * 100))

    def get_accuracy_(self, validation=False):

        if validation:
            num = len(self.valid_dataset[Constants.OBS]) // self.batch_size
        else:
            num = len(self.dataset[Constants.OBS]) // self.batch_size

        accs = []

        for i in range(num):

            obs, hand_states, hand_bits, qs, _ = self.get_batch_(i, validation=validation)

            with torch.no_grad():
                res = self.model.get_accuracy([obs, hand_states, hand_bits], qs)
                res = res.cpu().numpy()

            accs.append(res)

        return np.mean(accs)

    def train_model(self):

        if self.model_load_path is not None:
            # no training when loading model
            self.model.load(self.model_load_path)
            self.model.eval()
            return

        self.model.train()
        opt = optim.Adam(self.model.parameters(), self.encoder_learning_rate, weight_decay=self.encoder_weight_decay)

        losses = []
        valid_losses = []

        for key, value in self.model.state_dict().items():
            print(key, value.size())

        for training_step in range(self.num_training_steps):

            epoch_step = training_step % self.epoch_size

            if epoch_step == 0:

                self.dataset.shuffle()

                if len(losses) > 0:
                    tmp_loss = np.mean(losses[- self.epoch_size:])
                    print("({:d}, {:d}) total loss {:.4f}".format(
                        training_step // self.epoch_size, training_step, tmp_loss
                    ))

                tmp_valid_loss = self.validate_()
                valid_losses.append(tmp_valid_loss)
                self.maybe_save_best_model_(tmp_valid_loss)
                self.logger.info("validation complete")

            elif training_step % 100 == 0:

                print("step {:d}".format(training_step))

            obs, hand_obs, hand_bits, qs, amb_qs = self.get_batch_(epoch_step)

            opt.zero_grad()

            total_loss = self.model.compute_loss(
                [obs, hand_obs, hand_bits], qs, amb_labels=amb_qs
            )

            losses.append(total_loss.detach().cpu().numpy())
            total_loss.backward()

            opt.step()

        self.load_best_model_()

        if self.model_save_path is not None:
            torch.save(self.model.state_dict(), self.model_save_path)

        losses = np.array(losses)
        valid_losses = np.array(valid_losses)

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

    def plot_losses_panel_(self, losses, validation_losses):

        plt.figure(figsize=(24, 16))

        plt.subplot(1, 2, 1)
        runner_utils.plot_losses([losses], ["total"])
        runner_utils.plot_log_losses([validation_losses], ["total"], epoch_size=self.epoch_size)

        plt.subplot(1, 2, 2)
        runner_utils.plot_log_losses([np.abs(losses)], ["|total|"])
        runner_utils.plot_log_losses(
            [np.abs(validation_losses)], ["|total|"], validation=True, epoch_size=self.epoch_size
        )

        plt.show()

    def validate_(self):

        self.model.eval()

        losses = []

        with torch.no_grad():

            # throws away a bit of data if validation set size % batch size != 0
            num_steps = int(len(self.valid_dataset[Constants.OBS]) // self.batch_size)

            for step in range(num_steps):

                obs, hand_obs, hand_bits, qs, amb_qs = self.get_batch_(step, validation=True)

                with torch.no_grad():
                    total_loss = self.model.compute_loss(
                        [obs, hand_obs, hand_bits], qs, amb_labels=amb_qs
                    )

                losses.append(total_loss.detach().cpu().numpy())

        self.model.train()

        return np.mean(losses)

    def get_batch_(self, epoch_step, validation=False):

        b = np.index_exp[epoch_step * self.batch_size: (epoch_step + 1) * self.batch_size]

        if validation:
            dataset = self.valid_dataset
        else:
            dataset = self.dataset

        obs = dataset[Constants.OBS][b]
        hand_obs = dataset[Constants.HAND_OBS][b]
        hand_bits = dataset[Constants.HAND_BITS][b]
        qs = dataset[Constants.QS][b].astype(np.float32)

        to_return = [
            runner_utils.other_to_torch(obs[:, np.newaxis, :, :], self.device),
            runner_utils.other_to_torch(hand_obs[:, np.newaxis, :, :], self.device),
            runner_utils.other_to_torch(hand_bits, self.device).long(),
            runner_utils.other_to_torch(qs, self.device)
        ]

        if self.use_amb_labels:
            amb_qs = dataset[Constants.AMB_QS][b].astype(np.float32)
            to_return.append(runner_utils.other_to_torch(amb_qs, self.device))
        else:
            to_return.append(None)

        return to_return

    def load_dataset_(self):

        self.dataset = ArrayDataset(None)
        self.dataset.load_hdf5(self.load_path)

        tasks = runner_utils.parse_task_string(
            self.dataset.metadata[Constants.TASK_LIST], return_char_list=False
        )

        # make sure all arrays in the dataset have the same size
        assert self.pos_amb_labels_load_path is not None
        self.load_pos_amb_labels_()

        # now the shape should be |D|x|A|
        if self.limit is not None:
            self.dataset.limit(self.limit)

        if self.ignore_list is not None:

            to_delete = []

            for task in self.ignore_list:
                indices = list(np.where(self.dataset[Constants.TASK_INDEX] == task))
                to_delete += indices

            self.dataset.delete(to_delete)

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

        self.dataset[Constants.ABSTRACT_ACTIONS] = np.zeros(self.dataset.size, dtype=np.int32)

        self.dataset.shuffle()

        self.num_samples = self.dataset.size
        self.valid_samples = int(self.num_samples * self.validation_fraction)
        self.valid_dataset = self.dataset.split(self.valid_samples)

        self.epoch_size = self.dataset[Constants.OBS].shape[0] // self.batch_size

    def load_pos_amb_labels_(self):

        dataset = ArrayDataset(None)
        dataset.load_hdf5(self.pos_amb_labels_load_path)

        self.dataset[Constants.QS] = dataset[Constants.POSITIVE_LABELS]
        self.dataset[Constants.AMB_QS] = dataset[Constants.AMBIGUOUS_LABELS]
        # don't mask out optimal actions
        self.dataset[Constants.AMB_QS][self.dataset[Constants.QS] == True] = False

    def build_model_(self):

        heightmap_size = 90
        diag_length = float(heightmap_size) * np.sqrt(2)
        diag_length = int(np.ceil(diag_length / 32) * 32)

        patch_size = 24
        patch_channel = 1
        patch_shape = (patch_channel, patch_size, patch_size)

        self.encoder = ResUCatEncoder(
            1, 2, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape
        )

        self.model = AAF(self.encoder, self.model_config, self.logger)
        self.model.to(self.device)

    def plot_dataset_examples_(self):

        for i in range(100):

            obs = self.dataset[Constants.OBS][i]
            hand_obs = self.dataset[Constants.HAND_OBS][i]
            hand_bit = self.dataset[Constants.HAND_BITS][i]
            action = self.dataset[Constants.ACTIONS][i]
            reward = self.dataset[Constants.REWARDS][i][0]
            qs = self.dataset[Constants.QS][i].astype(np.float32)
            label = self.dataset[Constants.ABSTRACT_ACTIONS][i]

            print("hand bit: {:s}, action: {:s}, reward: {:s}, label: {:s}".format(
                str(hand_bit), str(action), str(reward), str(label)
            ))

            if Constants.ALLOWED_TASKS in self.dataset:
                print("allowed tasks: {:s}".format(str(self.dataset[Constants.ALLOWED_TASKS][i])))

            plt.figure(figsize=(16, 8))

            plt.subplot(1, 3, 1)
            plt.imshow(obs)
            plt.subplot(1, 3, 2)
            plt.imshow(hand_obs)
            plt.subplot(1, 3, 3)
            plt.imshow(qs.reshape((90, 90)))

            plt.tight_layout()
            plt.show()

    def plot_prediction_examples_(self):

        self.logger.info("first row: obs and hand obs")
        self.logger.info("second row: opt actions map with cutoff 0.5, 0.4, 0.3, 0.2 and 0.1 from left to right")

        from skimage.io import imsave

        for i in range(100):

            obs = self.dataset[Constants.OBS][i]
            hand_obs = self.dataset[Constants.HAND_OBS][i]
            hand_bits = self.dataset[Constants.HAND_BITS][i: i+1]

            imsave("obs_{:d}.png".format(i), obs * 25)
            imsave("hand_obs_{:d}.png".format(i), hand_obs * 25)

            with torch.no_grad():

                pred_qs = self.model.forward(
                    [runner_utils.other_to_torch(obs[None, None, :, :], self.device),
                     runner_utils.other_to_torch(hand_obs[None, None, :, :], self.device),
                     runner_utils.other_to_torch(hand_bits, self.device).long()]
                )
                pred_qs = pred_qs.cpu().numpy()[0]
                pred_qs = expit(pred_qs)

            map_0_5 = (pred_qs >= 0.5).astype(np.float32)
            map_0_4 = (pred_qs >= 0.4).astype(np.float32)
            map_0_3 = (pred_qs >= 0.3).astype(np.float32)
            map_0_2 = (pred_qs >= 0.2).astype(np.float32)
            map_0_1 = (pred_qs >= 0.1).astype(np.float32)

            size = int(np.sqrt(map_0_5.shape[0]))

            map_0_5 = map_0_5.reshape((size, size))
            map_0_4 = map_0_4.reshape((size, size))
            map_0_3 = map_0_3.reshape((size, size))
            map_0_2 = map_0_2.reshape((size, size))
            map_0_1 = map_0_1.reshape((size, size))

            imsave("map_0_5_{:d}.png".format(i), map_0_5)
            imsave("map_0_4_{:d}.png".format(i), map_0_4)
            imsave("map_0_3_{:d}.png".format(i), map_0_3)
            imsave("map_0_2_{:d}.png".format(i), map_0_2)
            imsave("map_0_1_{:d}.png".format(i), map_0_1)

            plt.figure(figsize=(16, 8))

            plt.subplot(2, 5, 1)
            plt.imshow(obs)

            plt.subplot(2, 5, 2)
            plt.imshow(hand_obs)

            plt.subplot(2, 5, 6)
            plt.imshow(map_0_5)

            plt.subplot(2, 5, 7)
            plt.imshow(map_0_4)

            plt.subplot(2, 5, 8)
            plt.imshow(map_0_3)

            plt.subplot(2, 5, 9)
            plt.imshow(map_0_2)

            plt.subplot(2, 5, 10)
            plt.imshow(map_0_1)

            plt.show()
