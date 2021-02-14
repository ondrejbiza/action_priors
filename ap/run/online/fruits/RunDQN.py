from datetime import datetime
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from ....modules.ConvEncoder import ConvEncoder
from ....modules.FCEncoder import FCEncoder
from ....modules.FCEncoderColumnSimple import FCEncoderColumnSimple
from ....modules.FCEncoderColumnLast import FCEncoderColumnLast
from ....models.dqn.DQN import DQN
from ....envs.fruits import Fruits
from ....constants import Constants
from ....utils.result import Result
from ....utils.dataset import ListDataset
from ....utils import runner as runner_utils
from ....utils.policy.EpsGreedyPolicy import EpsGreedyPolicy
from ....utils.policy.SoftmaxPolicy import SoftmaxPolicy


class RunDQN:

    NUM_EVAL_EPISODES = 10
    EVAL_FREQUENCY = 500

    def __init__(self, runner_config, model_config, logger):

        self.model_config = model_config
        self.logger = logger

        rc = runner_config
        self.goal = rc[Constants.GOAL]
        self.learning_rate = rc[Constants.LEARNING_RATE]
        self.weight_decay = rc[Constants.WEIGHT_DECAY]
        self.batch_size = rc[Constants.BATCH_SIZE]
        self.double_learning = rc[Constants.DOUBLE_LEARNING]
        self.target_network = rc[Constants.TARGET_NETWORK]
        self.target_network_sync = rc[Constants.TARGET_NETWORK_SYNC]
        self.max_steps = rc[Constants.MAX_STEPS]
        self.max_episodes = rc[Constants.MAX_EPISODES]
        self.num_fruits = rc[Constants.NUM_FRUITS]
        self.device = rc[Constants.DEVICE]
        self.policy_type = rc[Constants.POLICY]
        self.side_transfer = rc[Constants.SIDE_TRANSFER]
        self.side_transfer_last = rc[Constants.SIDE_TRANSFER_LAST]
        self.freeze_encoder = rc[Constants.FREEZE_ENCODER]

        assert self.max_steps is None or self.max_episodes is None
        assert self.max_steps is not None or self.max_episodes is not None

        self.build_env_()
        self.build_models_()

    def train_model(self):

        opt = self.get_opt()

        step = 0
        episode = 0

        result = Result()
        result.register(Constants.TOTAL_REWARDS)
        result.register(Constants.DISCOUNTED_REWARDS)
        result.register(Constants.TMP_REWARDS)
        result.register(Constants.EVAL_TOTAL_REWARDS)
        result.register(Constants.EVAL_DISCOUNTED_TOTAL_REWARDS)
        result.register(Constants.EVAL_NUM_STEPS)

        state = self.env.reset()

        while True:

            # maybe evaluate
            if step > 0 and step % self.EVAL_FREQUENCY == 0:
                eval_total, eval_discounted, eval_steps = self.evaluate_()
                result.add(Constants.EVAL_TOTAL_REWARDS, eval_total)
                result.add(Constants.EVAL_DISCOUNTED_TOTAL_REWARDS, eval_discounted)
                result.add(Constants.EVAL_NUM_STEPS, eval_steps)

            # maybe stop
            if self.should_terminate_(step, episode):
                break

            # maybe log
            if step > 0 and step % 100 == 0:
                self.logger.info("step {:d}: {:.2f}r 100ep, {:.2f}dr 100 ep, {:.2f}exp".format(
                    step, result.get_mean_window(Constants.TOTAL_REWARDS, 100),
                    result.get_mean_window(Constants.DISCOUNTED_REWARDS, 100),
                    self.dqn.policy.exploration_schedule.value(step)
                ))

            # act
            # TODO: I'm not using batch norm right now but DQN should be in an eval mode here
            # TODO: check if eval mode deleted momenta
            action = self.dqn.act(runner_utils.states_to_torch(state[np.newaxis, :, :, :], self.device), step)

            next_state, reward, done, metadata = self.env.step(action)
            result.add(Constants.TMP_REWARDS, reward)

            self.dqn.remember(state, action, reward, next_state, metadata[Constants.REACHED_GOAL])

            state = next_state

            if done:
                state = self.env.reset()

                result.add(Constants.TOTAL_REWARDS, result.sum(Constants.TMP_REWARDS))
                result.add(
                    Constants.DISCOUNTED_REWARDS, result.discounted_sum(Constants.TMP_REWARDS, self.dqn.discount)
                )
                result.reset(Constants.TMP_REWARDS)

                episode += 1

            step += 1

            # learn
            if step > 50:
                self.learn_step_(opt, step)

            # maybe sync nets
            if self.target_network and self.target_network_sync is not None and step % self.target_network_sync == 0:
                self.logger.info("target net sync")
                self.target_dqn.sync_weights(self.dqn)

        self.training_result = result

    def generate_demonstrations(self, num_expert, num_random):

        env = cp.deepcopy(self.env)
        env.reset()

        for i in range(num_expert + num_random):

            state = env.get_state()

            if i < num_expert:
                # TODO: add this as param or sth
                if 0.5 < np.random.uniform(0, 1):
                    action = int(np.random.randint(0, env.size ** 2))
                else:
                    action = env.get_optimal_action_()
            else:
                action = int(np.random.randint(0, env.size ** 2))

            next_state, reward, done, metadata = env.step(action)

            self.dqn.remember(state, action, reward, next_state, metadata[Constants.REACHED_GOAL])

            if done:
                env.reset()

    def collect_data(self, num_exp, eps, save_path):

        # create a dict of lists
        dataset = ListDataset()

        # set DQN exploration to a constant value
        self.dqn.policy.init_explore = eps
        self.dqn.policy.final_explore = eps
        self.dqn.policy.exploration_steps = 100
        self.dqn.policy.reset()
        self.dqn.eval()

        # collect data
        state = self.env.reset()

        for i in range(num_exp):

            # TODO: inference twice because lazy
            action = self.dqn.act(
                runner_utils.states_to_torch(state[np.newaxis, :, :, :], self.device), 0
            )
            with torch.no_grad():
                qs = self.dqn(
                    runner_utils.states_to_torch(state[np.newaxis, :, :, :], self.device)
                ).cpu().numpy()[0, :]

            abstract_action = self.env.get_abstract_action(action)
            next_state, reward, done, _ = self.env.step(action)

            dataset.add(Constants.STATES, state)
            dataset.add(Constants.ACTIONS, action)
            dataset.add(Constants.ABSTRACT_ACTIONS, abstract_action)
            dataset.add(Constants.QS, qs)
            dataset.add(Constants.REWARDS, reward)
            dataset.add(Constants.NEXT_STATES, next_state)
            dataset.add(Constants.DONES, done)

            state = next_state

            if done:
                state = self.env.reset()

        # turn lists into arrays and save the dataset as HDF5
        dataset = dataset.to_array_dataset({
            Constants.STATES: np.float32, Constants.ACTIONS: np.int32, Constants.ABSTRACT_ACTIONS: np.int32,
            Constants.QS: np.float32, Constants.REWARDS: np.float32, Constants.NEXT_STATES: np.float32,
            Constants.DONES: np.bool
        })
        dataset.metadata = {
            Constants.NUM_EXP: num_exp, Constants.EPS: eps, Constants.TIMESTAMP: str(datetime.today())
        }
        dataset.save_hdf5(save_path)

    def label_data(self, states, actions=None):

        num = int(np.ceil(len(states) / self.batch_size))
        qs_list = []

        for i in range(num):

            batch_states = states[i * self.batch_size: (i + 1) * self.batch_size]
            batch_states = runner_utils.states_to_torch(batch_states, self.device)

            with torch.no_grad():
                qs = self.dqn(batch_states).cpu().numpy()

            if actions is not None:
                batch_actions = actions[i * self.batch_size: (i + 1) * self.batch_size]
                qs = qs[list(range(len(batch_actions))), batch_actions]

            qs_list.append(qs)

        return np.concatenate(qs_list, axis=0)

    def demonstrate_dqn(self):

        state = self.env.reset()

        while True:

            action = self.dqn.act(
                runner_utils.states_to_torch(state[np.newaxis, :, :, :], self.device), 0, evaluation=True
            )
            with torch.no_grad():
                qs = self.dqn(
                    runner_utils.states_to_torch(state[np.newaxis, :, :, :], self.device)
                ).cpu().numpy()[0, :]

            print(qs.reshape((5, 5)))
            print(action)
            plt.imshow(Fruits.state_to_image(state))
            plt.show()

            next_state, reward, done, _ = self.env.step(action)

            state = next_state

            if done:
                state = self.env.reset()

    def learn_step_(self, opt, step):

        states, actions, rewards, next_states, dones, weights, batch_indexes = \
            self.dqn.sample_buffer(self.batch_size, step)

        states = runner_utils.states_to_torch(np.array(states, dtype=np.float32), self.device)
        actions = runner_utils.other_to_torch(np.array(actions, dtype=np.int32), self.device)
        rewards = runner_utils.other_to_torch(np.array(rewards, dtype=np.float32), self.device)
        next_states = runner_utils.states_to_torch(np.array(next_states, dtype=np.float32), self.device)
        dones = runner_utils.other_to_torch(np.array(dones, dtype=np.bool), self.device)
        weights = runner_utils.other_to_torch(weights, self.device)

        with torch.no_grad():
            target_next_qs = self.target_dqn(next_states)

        if self.double_learning:
            with torch.no_grad():
                policy_next_qs = self.dqn(next_states)
            next_actions = policy_next_qs.argmax(1)
        else:
            next_actions = target_next_qs.argmax(1)

        next_actions = next_actions.type(torch.long)
        tmp = torch.arange(next_actions.size()[0], dtype=torch.long, device=self.device)
        next_qs = target_next_qs[tmp, next_actions]

        opt.zero_grad()
        td_error, loss = self.dqn.calculate_td_error_and_loss(states, actions, rewards, dones, next_qs, weights=weights)
        loss.backward()
        opt.step()

        # doesn't do anything if not using prioritized replay
        td_error = td_error.detach().cpu().numpy()
        self.dqn.update_priorities(td_error, batch_indexes)

    def should_terminate_(self, step, episode):

        if self.max_steps is not None and step >= self.max_steps:
            return True

        if self.max_episodes is not None and episode >= self.max_episodes:
            return True

        return False

    def evaluate_(self):

        env = cp.deepcopy(self.env)
        total_rewards = []
        discounted_rewards = []
        num_steps = []

        for i in range(self.NUM_EVAL_EPISODES):

            state = env.reset()
            tmp_rewards = []

            while True:

                action = self.dqn.act(
                    runner_utils.states_to_torch(state[np.newaxis, :, :, :], self.device), 0, evaluation=True
                )
                next_state, reward, done, _ = env.step(action)
                tmp_rewards.append(reward)

                if done:
                    break

                state = next_state

            total_rewards.append(np.sum(tmp_rewards))
            discounted_rewards.append(np.sum([(self.dqn.discount ** i) * r for i, r in enumerate(tmp_rewards)]))
            num_steps.append(len(tmp_rewards))

        return np.mean(total_rewards), np.mean(discounted_rewards), np.mean(num_steps)

    def build_env_(self):

        self.env = Fruits(num_fruits=self.num_fruits, no_start=True, max_steps=30, no_wrong_pick=True)

        if self.goal is not None:
            self.env.goal = self.goal

    def build_models_(self):

        if self.side_transfer:
            build_dqn_fc = self.build_dqn_conv_fcn_side_transfer_
        else:
            build_dqn_fc = self.build_dqn_conv_fc_

        if not self.target_network:
            # only one network
            self.dqn, self.encoder = build_dqn_fc()
            self.dqn = self.dqn.to(self.device)
            self.target_dqn = self.dqn
            self.target_encoder = self.encoder
        else:
            # build policy and target nets
            self.dqn, self.encoder = build_dqn_fc()
            self.target_dqn, self.target_encoder = build_dqn_fc()

            # move to device
            self.dqn.to(self.device)
            self.target_dqn.to(self.device)

            # synchronize target net with policy net and set it to evaluation mode
            self.target_dqn.sync_weights(self.dqn)
            self.target_dqn.eval()

        # log layers
        self.logger.info("layers: " + str(self.dqn.state_dict().keys()))
        self.logger.info("target layers: " + str(self.target_dqn.state_dict().keys()))

        # setup for training
        self.dqn.setup_for_training()

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
            policy = EpsGreedyPolicy(self.model_config)
        elif self.policy_type == Constants.SOFTMAX:
            policy = SoftmaxPolicy(self.model_config)
        else:
            raise ValueError("Wrong policy type.")

        dqn = DQN(encoder, policy, self.model_config, self.logger)

        return dqn, encoder

    def build_dqn_conv_fcn_side_transfer_(self):

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

        side_fc_encoder = FCEncoder(fc_config, self.logger)

        if self.side_transfer_last:
            fc_encoder = FCEncoderColumnLast(fc_config, self.logger, side_fc_encoder)
        else:
            fc_encoder = FCEncoderColumnSimple(fc_config, self.logger, side_fc_encoder)

        encoder = nn.Sequential(conv_encoder, fc_encoder)
        encoder.input_size = conv_encoder.input_size
        encoder.output_size = fc_encoder.output_size

        if self.policy_type == Constants.EPS:
            policy = EpsGreedyPolicy(self.model_config)
        elif self.policy_type == Constants.SOFTMAX:
            policy = SoftmaxPolicy(self.model_config)
        else:
            raise ValueError("Wrong policy type.")

        dqn = DQN(encoder, policy, self.model_config, self.logger)

        return dqn, encoder

    def get_opt(self):

        if self.freeze_encoder:
            # only train the head, which is either a single Q layer, or an A and a V layer
            q_params = self.dqn.head_parameters()
        else:
            # in side transfer, I set require_grad=False for the side encoder, I also block gradients in forward
            q_params = self.dqn.parameters()

        return optim.Adam(q_params, self.learning_rate, weight_decay=self.weight_decay)

    def load_encoder(self, path):

        self.dqn.encoder.load_state_dict(torch.load(path, map_location=self.device))
        self.target_dqn.sync_weights(self.dqn)

    def load_side_encoder(self, path):

        state_dict = torch.load(path, map_location=self.device)

        # I need to remove the key prefix because of implementation mess
        # the encoder object is nn.Sequential, but the side encoder object is just FCEncoder
        # (contained inside nn.Sequential)
        # I need to use nn.Sequential._modules["1"] to access the actual encoder
        # but then I need to remove the "1." prefix from the weight names

        new_state_dict = {}
        for key in state_dict.keys():
            assert key[:2] == "1."
            new_key = key[2:]
            new_state_dict[new_key] = state_dict[key]

        self.dqn.encoder._modules["1"].column.load_state_dict(new_state_dict)
        self.target_dqn.sync_weights(self.dqn)
