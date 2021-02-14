import os
from datetime import datetime
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
import torch
from .... import constants
from ....constants import Constants
from ....constants import ExpertTransition
from ....helping_hands_rl_envs import env_factory
from ....modules.hand_obs.ResUCat import ResUCat
from ....models.dqn.expert.DQNXRotInHand import DQNXRotInHand
from ....models.dqn.expert.DQNXRotInHandMargin import DQNXRotInHandMargin
from ....models.dqn.expert.utils.QLearningBuffer import QLearningBuffer, QLearningBufferExpert
from ....models.dqn.expert.utils.PrioritizedQLearningBuffer import PrioritizedQLearningBuffer
from ....models.dqn.expert.utils.PrioritizedQLearningBuffer import NORMAL as PR_NORMAL
from ....models.dqn.expert.utils.PrioritizedQLearningBuffer import EXPERT as PR_EXPERT
from ....run.online.blocks.RunDeconstruct import RunDeconstruct
from ....utils.schedule import LinearSchedule
from ....utils.dataset import ArrayDataset, ListDataset
from ....utils.result import Result
from ....utils import runner as runner_utils


class RunDQN(RunDeconstruct):

    PATCH_CHANNEL = 1
    NUM_PRIMITIVES = 2
    # doesn't really matter since we always use the "0" rotation
    HALF_ROTATION = True
    SUPERVISED_LEARNING = False

    def __init__(self, runner_config, env_config, planner_config, logger, exp_policy=None):

        self.exp_policy = exp_policy
        rc = runner_config
        self.device = rc[Constants.DEVICE]
        self.patch_size = rc[Constants.PATCH_SIZE]
        self.learning_rate = rc[Constants.LEARNING_RATE]
        self.discount = rc[Constants.DISCOUNT] # TODO: default is 0.5, no wonder log 0.9 didn't work
        self.buffer_type = rc[Constants.BUFFER_TYPE]
        self.alg = rc[Constants.ALG]
        self.margin = rc[Constants.MARGIN]
        self.margin_l = rc[Constants.MARGIN_L]
        self.margin_weight = rc[Constants.MARGIN_WEIGHT]
        self.margin_beta = rc[Constants.MARGIN_BETA]
        self.divide_factor = rc[Constants.DIVIDE_FACTOR]
        self.buffer_size = rc[Constants.BUFFER_SIZE]
        self.per_alpha = rc[Constants.PER_ALPHA]
        self.exploration_steps = rc[Constants.EXPLORATION_STEPS]
        self.init_eps = rc[Constants.INIT_EPS]
        self.final_eps = rc[Constants.FINAL_EPS]
        self.per_expert_eps = rc[Constants.PER_EXPERT_EPS]
        self.per_eps = rc[Constants.PER_EPS]
        self.per_beta = rc[Constants.PER_BETA]
        self.max_episodes = rc[Constants.MAX_EPISODES]
        self.init_coef = rc[Constants.INIT_COEF]
        self.final_coef = rc[Constants.FINAL_COEF]
        self.batch_size = rc[Constants.BATCH_SIZE]
        self.target_update_freq = rc[Constants.TARGET_UPDATE_FREQ]
        self.fixed_eps = rc[Constants.FIXED_EPS]
        self.true_random = rc[Constants.TRUE_RANDOM]
        self.training_offset = rc[Constants.TRAINING_OFFSET]
        self.training_iters = rc[Constants.TRAINING_ITERS]
        self.get_custom_labels = rc[Constants.GET_CUSTOM_LABELS]
        self.fake_expert = rc[Constants.FAKE_EXPERT]

        assert self.buffer_type in [
            Constants.PRIORITIZED_BUFFER, Constants.PRIORITIZED_BUFFER_EXPERT, Constants.EXPERT_BUFFER,
            Constants.BUFFER
        ]

        # planner config doesn't do anything here
        # but the environment constructor still requires it
        super(RunDQN, self).__init__(runner_config, env_config, planner_config, logger)

        self.action_space = [0, self.heightmap_size]

        self.fcn, self.agent = None, None
        self.create_agent_()

        self.replay_buffer = None
        self.create_replay_buffer_()

        self.exploration, self.coefficient_schedule, self.p_beta_schedule = None, None, None
        self.create_schedules_()

    def pretraining(self, steps):

        self.logger.info("pretraining for {:d} steps".format(steps))
        result = self.init_result_()

        if self.buffer_type in [Constants.PRIORITIZED_BUFFER, Constants.PRIORITIZED_BUFFER_EXPERT]:
            self.logger.warning("using prioritized buffer during expert pre-training")

        for step_idx in range(steps):

            if step_idx % 1000 == 0:
                self.logger.info("pretraining step {:d}".format(step_idx))

            self.train_step_(step_idx, 0, result)

        self.logger.info("pretraining done")
        return result

    def training(self):

        self.logger.info("training for {:d} episodes".format(self.max_episodes))
        result = self.init_result_()

        states, in_hands, obs = self.envs.reset()

        episode_idx = 0
        step_idx = 0
        training_step_idx = 0
        episode_rewards = np.zeros(self.num_processes)
        is_expert = [torch.tensor(0, dtype=torch.int32, device="cpu") for _ in range(self.num_processes)]
        tmp_buffers = [[] for _ in range(self.num_processes)]

        while episode_idx < self.max_episodes:

            if step_idx % 1000 == 0:
                self.logger.info("training step {:d}, episode {:d}".format(step_idx, episode_idx))
                self.logger.info("mean reward last 100 ep: {:.5f}".format(np.mean(result[Constants.REWARDS][-100:])))

            eps, coef = self.get_eps_and_coef_(episode_idx)

            if self.exp_policy is not None:

                q_value_maps, actions_star_idx, actions_star = self.agent.getEGreedyActionsWithMask(
                    states, in_hands.permute((0, 3, 1, 2)), obs.permute((0, 3, 1, 2)), eps, self.exp_policy, coef
                )

            else:

                q_value_maps, actions_star_idx, actions_star = self.agent.getEGreedyActions(
                    states, in_hands.permute((0, 3, 1, 2)), obs.permute((0, 3, 1, 2)), eps, coef,
                    random_actions=self.true_random
                )

            buffer_obs = self.agent.getCurrentObs(in_hands, obs)
            actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)

            self.envs.stepAsync(actions_star, auto_reset=False)

            if len(self.replay_buffer) >= self.training_offset:
                for training_iter in range(self.training_iters):
                    self.train_step_(training_step_idx, episode_idx, result)
                    training_step_idx += 1

            res = self.envs.stepWait()
            states_, in_hands_, obs_, rewards, dones = res[:5]
            # steps left is not used anymore, it should always return 100
            steps_lefts = self.envs.getStepLeft()
            episode_rewards += rewards.squeeze().cpu().numpy()

            done_idxes = torch.nonzero(dones).squeeze(1)
            if done_idxes.shape[0] != 0:
                reset_states_, reset_in_hands_, reset_obs_ = self.envs.reset_envs(done_idxes)
                for j, idx in enumerate(done_idxes):
                    states_[idx] = reset_states_[j]
                    in_hands_[idx] = reset_in_hands_[j]
                    obs_[idx] = reset_obs_[j]

                    # retroactively set expert to true if the episode was successful
                    if episode_rewards[idx] > 0.0:
                        is_expert[idx] += 1
                    # create a new memory entry for is_expert
                    is_expert[idx] = torch.tensor(0, dtype=torch.int32, device="cpu")

                    # I keep a buffer for the current episode in each process when I'm doing fake expert
                    # so that the expert buffer sorts the collected transitions properly
                    if self.fake_expert:
                        for t in tmp_buffers[idx]:
                            self.replay_buffer.add(t)
                        tmp_buffers[idx] = []

                    result.add(Constants.REWARDS, float(episode_rewards[idx]))
                    episode_rewards[idx] = 0.0

            buffer_obs_ = self.agent.getCurrentObs(in_hands_, obs_)

            for i in range(self.num_processes):
                if self.fake_expert:
                    tmp_is_expert = is_expert[i]
                else:
                    tmp_is_expert = torch.tensor(0)

                t = ExpertTransition(states[i], buffer_obs[i], actions_star_idx[i], rewards[i], states_[i],
                                     buffer_obs_[i], dones[i], steps_lefts[i], tmp_is_expert)

                # if fake expert is used, the experience will be added to the replay buffer later
                if not self.fake_expert:
                    self.replay_buffer.add(t)
                else:
                    tmp_buffers[i].append(t)

            states = cp.copy(states_)
            obs = cp.copy(obs_)
            in_hands = cp.copy(in_hands_)

            step_idx += self.num_processes
            episode_idx += int(np.sum(dones.cpu().numpy()))

        self.logger.info("training done")
        return result

    def evaluation(self, num_episodes):

        self.logger.info("evaluating for {:d} episodes".format(num_episodes))
        result = self.init_result_()

        states, in_hands, obs = self.envs.reset()

        episode_idx = 0
        step_idx = 0
        eps = 0.0
        coef = self.final_coef
        episode_rewards = np.zeros(self.num_processes)

        while episode_idx < num_episodes:

            if step_idx % 1000 == 0:
                self.logger.info("training step {:d}, episode {:d}".format(step_idx, episode_idx))

            q_value_maps, actions_star_idx, actions_star = self.agent.getEGreedyActions(
                states, in_hands.permute((0, 3, 1, 2)), obs.permute((0, 3, 1, 2)), eps, coef,
                random_actions=self.true_random
            )

            actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)

            self.envs.stepAsync(actions_star, auto_reset=False)

            res = self.envs.stepWait()
            states_, in_hands_, obs_, rewards, dones = res[:5]

            episode_rewards += rewards.squeeze().cpu().numpy()

            done_idxes = torch.nonzero(dones).squeeze(1)
            if done_idxes.shape[0] != 0:
                reset_states_, reset_in_hands_, reset_obs_ = self.envs.reset_envs(done_idxes)
                for j, idx in enumerate(done_idxes):
                    states_[idx] = reset_states_[j]
                    in_hands_[idx] = reset_in_hands_[j]
                    obs_[idx] = reset_obs_[j]

                    result.add(Constants.REWARDS, float(episode_rewards[idx]))
                    episode_rewards[idx] = 0.0

            states = cp.copy(states_)
            obs = cp.copy(obs_)
            in_hands = cp.copy(in_hands_)

            step_idx += self.num_processes
            episode_idx += int(np.sum(dones.cpu().numpy()))

        self.logger.info("evaluation done")
        return result

    def collect_data(self, num_samples, save_path, save_all_qs):

        dataset = ListDataset()

        self.logger.info("collecting {:d} transitions".format(num_samples))
        result = self.init_result_()

        states, in_hands, obs = self.envs.reset()

        episode_idx = 0
        step_idx = 0
        episode_rewards = np.zeros(self.num_processes)

        while dataset.get_size() < num_samples:

            if step_idx % 1000 == 0:
                self.logger.info("data collection step {:d}, episode {:d}".format(step_idx, episode_idx))

            eps, coef = self.get_eps_and_coef_(episode_idx)

            q_value_maps, actions_star_idx, actions_star = self.agent.getEGreedyActions(
                states, in_hands.permute((0, 3, 1, 2)), obs.permute((0, 3, 1, 2)), eps, coef,
                random_actions=self.true_random
            )

            actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)

            self.envs.stepAsync(actions_star, auto_reset=False)

            res = self.envs.stepWait()

            if self.get_custom_labels:
                states_, in_hands_, obs_, rewards, dones, metadata = res
                labels = np.stack([list(d["labels"]) for d in metadata], axis=0)
            else:
                states_, in_hands_, obs_, rewards, dones = res
                labels = None

            episode_rewards += rewards.squeeze().cpu().numpy()

            for i in range(self.num_processes):
                # I use deepcopy because of reset that comes after this block of code
                # I think copy would suffice
                dataset.add(Constants.HAND_BITS, self.to_numpy_(states[i]))
                dataset.add(Constants.OBS, self.to_numpy_(obs[i][:, :, 0]))
                dataset.add(Constants.HAND_OBS, self.to_numpy_(in_hands[i][:, :, 0]))
                dataset.add(Constants.ACTIONS, self.to_numpy_(actions_star_idx[i]))
                dataset.add(Constants.REWARDS, self.to_numpy_(rewards[i]))
                dataset.add(Constants.NEXT_HAND_BITS, cp.deepcopy(self.to_numpy_(states_[i])))
                dataset.add(Constants.NEXT_OBS, cp.deepcopy(self.to_numpy_(obs_[i][:, :, 0])))
                dataset.add(Constants.NEXT_HAND_OBS, cp.deepcopy(self.to_numpy_(in_hands_[i][:, :, 0])))
                dataset.add(Constants.DONES, dones[i])

                if self.get_custom_labels:
                    dataset.add(Constants.ABSTRACT_ACTIONS, labels[i])

                if save_all_qs:
                    dataset.add(Constants.QS, self.to_numpy_(q_value_maps[i][0, :, :]))
                else:
                    raise NotImplementedError("")

            done_idxes = torch.nonzero(dones).squeeze(1)
            if done_idxes.shape[0] != 0:
                reset_states_, reset_in_hands_, reset_obs_ = self.envs.reset_envs(done_idxes)
                for j, idx in enumerate(done_idxes):
                    states_[idx] = reset_states_[j]
                    in_hands_[idx] = reset_in_hands_[j]
                    obs_[idx] = reset_obs_[j]

                    result.add(Constants.REWARDS, float(episode_rewards[idx]))
                    episode_rewards[idx] = 0.0

            states = cp.copy(states_)
            obs = cp.copy(obs_)
            in_hands = cp.copy(in_hands_)

            step_idx += self.num_processes
            episode_idx += int(np.sum(dones.cpu().numpy()))

        self.logger.info("data collection done")

        dtypes = {
            Constants.HAND_BITS: np.int32, Constants.OBS: np.float32, Constants.HAND_OBS: np.float32,
            Constants.ACTIONS: np.int32, Constants.REWARDS: np.float32, Constants.NEXT_HAND_BITS: np.int32,
            Constants.NEXT_OBS: np.float32, Constants.NEXT_HAND_OBS: np.float32, Constants.DONES: np.bool,
            Constants.QS: np.float32
        }
        if self.get_custom_labels:
            dtypes[Constants.ABSTRACT_ACTIONS] = np.int32

        dataset = dataset.to_array_dataset(dtypes)
        dataset.metadata = {
            Constants.NUM_EXP: dataset.size, Constants.TIMESTAMP: str(datetime.today())
        }
        dataset.save_hdf5(save_path)

        return result

    def add_qs_to_dataset(self, dataset_path, full_map_only_qs=False):

        goal_string = self.env_config["goal_string"]

        dataset = ArrayDataset(None)
        dataset.load_hdf5(dataset_path)
        all_goals = dataset.metadata[Constants.TASK_LIST].split(",")
        num_tasks = len(all_goals)
        task_index = all_goals.index(goal_string)

        if full_map_only_qs:
            # save all qs for all actions
            self.add_all_actions_q_value_array_to_dataset_(dataset, num_tasks)
        else:
            # save qs, advantages, log advantages and is optimal flags for a single action
            self.add_single_action_value_arrays_to_dataset_(dataset, num_tasks)

        num = int(np.ceil(float(dataset.size) / self.batch_size))
        qs = []
        advs = []
        log_advs = []
        is_opt_l = []

        for i in range(num):
            b = np.index_exp[i * self.batch_size: (i + 1) * self.batch_size]

            obs = dataset[Constants.OBS][b][:, None, :, :]
            hand_obs = dataset[Constants.HAND_OBS][b][:, None, :, :]
            hand_bits = dataset[Constants.HAND_BITS][b]
            actions = dataset[Constants.ACTIONS][b]

            obs = runner_utils.other_to_torch(obs, self.device)
            hand_obs = runner_utils.other_to_torch(hand_obs, self.device)
            hand_bits = runner_utils.other_to_torch(hand_bits, self.device)

            with torch.no_grad():
                q_value_maps, actions_star_idx, actions_star = self.agent.getEGreedyActions(
                    hand_bits, hand_obs, obs, 0.0, 0.0
                )
                q_value_maps = q_value_maps.detach().cpu().numpy()

            q_value_maps = q_value_maps[:, 0, :, :]
            shape = q_value_maps.shape
            q_value_maps = np.reshape(q_value_maps, (shape[0], shape[1] * shape[2]))

            if full_map_only_qs:
                qs.append(q_value_maps)
            else:
                q_value_maps, advantages, log_advantages, is_opt = \
                    self.get_single_action_values_from_q_values_(q_value_maps, actions)

                qs.append(q_value_maps)
                advs.append(advantages)
                log_advs.append(log_advantages)
                is_opt_l.append(is_opt)

        qs = np.concatenate(qs, axis=0)
        dataset[Constants.QS][:, task_index] = qs

        if not full_map_only_qs:
            advs = np.concatenate(advs, axis=0)
            log_advs = np.concatenate(log_advs, axis=0)
            is_opt_l = np.concatenate(is_opt_l, axis=0)

            dataset[Constants.ADVS][:, task_index] = advs
            dataset[Constants.LOG_ADVS][:, task_index] = log_advs
            dataset[Constants.IS_OPT][:, task_index] = is_opt_l

        dataset.save_hdf5(dataset_path)

    def generate_positive_and_ambiguous_labels(self, dataset_path, save_path, int1_threshold, int2_threshold):

        dataset = ArrayDataset(None)
        dataset.load_hdf5(dataset_path)

        num = int(np.ceil(float(dataset.size) / self.batch_size))

        opt_map = np.zeros((dataset.size, 90 * 90), dtype=np.bool)
        amb_map = np.zeros((dataset.size, 90 * 90), dtype=np.bool)

        for i in range(num):

            b = np.index_exp[i * self.batch_size: (i + 1) * self.batch_size]

            obs = dataset[Constants.OBS][b][:, None, :, :]
            hand_obs = dataset[Constants.HAND_OBS][b][:, None, :, :]
            hand_bits = dataset[Constants.HAND_BITS][b]

            obs = runner_utils.other_to_torch(obs, self.device)
            hand_obs = runner_utils.other_to_torch(hand_obs, self.device)
            hand_bits = runner_utils.other_to_torch(hand_bits, self.device)

            with torch.no_grad():
                q_value_maps, actions_star_idx, actions_star = self.agent.getEGreedyActions(
                    hand_bits, hand_obs, obs, 0.0, 0.0
                )
                q_value_maps = q_value_maps.detach().cpu().numpy()

            q_value_maps = q_value_maps[:, 0, :, :]
            shape = q_value_maps.shape
            q_value_maps = np.reshape(q_value_maps, (shape[0], shape[1] * shape[2]))

            opt_values = np.max(q_value_maps, axis=1)
            int1_values = opt_values - int1_threshold
            int2_values = opt_values - int1_threshold - int2_threshold

            tmp_opt_map = np.zeros((len(obs), 90 * 90), dtype=np.bool)
            tmp_amb_map = np.zeros((len(obs), 90 * 90), dtype=np.bool)

            tmp_opt_map[q_value_maps >= int1_values[:, np.newaxis]] = True
            tmp_amb_map[np.bitwise_and(q_value_maps < int1_values[:, np.newaxis],
                                       q_value_maps >= int2_values[:, np.newaxis])] = True

            opt_map[b] = tmp_opt_map
            amb_map[b] = tmp_amb_map

        np.save(save_path + constants.OPT_SUFFIX, opt_map)
        np.save(save_path + constants.AMB_SUFFIX, amb_map)

    @torch.no_grad()
    def show_predictions_from_dataset(self, load_path):

        # clean replay buffer
        self.create_replay_buffer_()
        # not exactly what we want, but it's easy to reuse this code
        self.load_expert_transitions(load_path)

        while True:

            batch = self.replay_buffer.sample(1)

            states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = \
                self.agent._loadBatchToDevice(batch)
            q_map = self.agent.forwardFCN(states, obs)

            image = cp.deepcopy(obs[0][0, 0].cpu().numpy())
            hand_image = obs[1][0, 0].cpu().numpy()

            print(image.shape, action_idx)
            image[action_idx[0][0], action_idx[0][1]] = np.max(image) + 0.03

            q_map = q_map.cpu().numpy()[0, 0]

            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.subplot(1, 3, 2)
            plt.imshow(hand_image)
            plt.subplot(1, 3, 3)
            plt.imshow(q_map)
            plt.show()

    def add_single_action_value_arrays_to_dataset_(self, dataset, num_tasks):

        if Constants.QS not in dataset:
            dataset[Constants.QS] = np.zeros((dataset.size, num_tasks), dtype=np.float32)

        if Constants.ADVS not in dataset:
            dataset[Constants.ADVS] = np.zeros((dataset.size, num_tasks), dtype=np.float32)

        if Constants.LOG_ADVS not in dataset:
            dataset[Constants.LOG_ADVS] = np.zeros((dataset.size, num_tasks), dtype=np.float32)

        if Constants.IS_OPT not in dataset:
            dataset[Constants.IS_OPT] = np.zeros((dataset.size, num_tasks), dtype=np.bool)

        for key in [Constants.QS, Constants.ADVS, Constants.LOG_ADVS, Constants.IS_OPT]:
            assert dataset[key].shape == (dataset.size, num_tasks)

    def add_all_actions_q_value_array_to_dataset_(self, dataset, num_tasks):

        if Constants.QS not in dataset:
            dataset[Constants.QS] = np.zeros((dataset.size, num_tasks, 90 * 90), dtype=np.float32)
        else:
            assert dataset[Constants.QS].shape == (dataset.size, num_tasks, 90 * 90)

    def get_single_action_values_from_q_values_(self, q_value_maps, actions):

        opt_actions = np.argmax(q_value_maps, axis=1)
        is_opt = actions == opt_actions

        q_value_maps[q_value_maps <= 0] = 1e-4
        values = np.max(q_value_maps, axis=1)
        advantages = q_value_maps - values[:, np.newaxis]

        log_q_value_maps = np.log(q_value_maps) / np.log(self.discount)
        log_values = np.max(log_q_value_maps, axis=1)
        log_advantages = log_q_value_maps - log_values[:, np.newaxis]

        q_value_maps = q_value_maps[list(range(q_value_maps.shape[0])), actions]
        advantages = advantages[list(range(advantages.shape[0])), actions]
        log_advantages = log_advantages[list(range(log_advantages.shape[0])), actions]

        return q_value_maps, advantages, log_advantages, is_opt

    def get_eps_and_coef_(self, episode_idx):

        if self.fixed_eps:
            eps = self.final_eps
            coef = self.final_coef
        else:
            eps = self.exploration.value(episode_idx)
            coef = self.coefficient_schedule.value(episode_idx)

        return eps, coef

    def init_result_(self):

        result = Result()
        result.register(Constants.EXPERT_FRACTION)
        result.register(Constants.TOTAL_LOSS)
        result.register(Constants.TD_ERROR)
        result.register(Constants.REWARDS)
        return result

    def train_step_(self, step_idx, episode_idx, result):

        if self.buffer_type == Constants.PRIORITIZED_BUFFER or \
                self.buffer_type == Constants.PRIORITIZED_BUFFER_EXPERT:

            # sample from buffer
            beta = self.p_beta_schedule.value(episode_idx)
            batch, weights, batch_idxes = self.replay_buffer.sample(self.batch_size, beta)
            # training step
            loss, td_error = self.agent.update(batch)
            # update priorities
            new_priorities = np.abs(td_error.cpu()) + \
                torch.tensor(list(zip(*batch))[-1]).float() * self.per_expert_eps + self.per_eps
            self.replay_buffer.update_priorities(batch_idxes, new_priorities)

            # log fraction of expert samples vs on-policy samples
            result.add(
                Constants.EXPERT_FRACTION,
                torch.tensor(list(zip(*batch))[-1]).sum().float().item() / self.batch_size
            )
        else:
            # sample from buffer
            batch = self.replay_buffer.sample(self.batch_size)
            # training step
            loss, td_error = self.agent.update(batch)

        # log loss and td-error
        result.add(Constants.TOTAL_LOSS, loss)
        result.add(Constants.TD_ERROR, td_error.mean().item())

        if step_idx % self.target_update_freq == 0:
            self.agent.updateTarget()

    def create_agent_(self):

        diag_length = self.get_diag_length_()
        patch_shape = self.get_patch_shape_()

        self.fcn = ResUCat(
            1, self.NUM_PRIMITIVES, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape
        ).to(self.device)

        assert self.alg in ["dqn", "dqn_margin"]

        if self.alg == "dqn_margin":
            self.agent = DQNXRotInHandMargin(
                self.fcn, self.action_space, self.workspace, self.heightmap_resolution, self.device, self.learning_rate,
                self.discount, self.NUM_PRIMITIVES, self.SUPERVISED_LEARNING,
                self.buffer_type == Constants.PRIORITIZED_BUFFER, self.num_rotations,
                self.HALF_ROTATION, self.patch_size, self.margin, self.margin_l, self.margin_weight, self.margin_beta,
                self.divide_factor
            )
        else:
            self.agent = DQNXRotInHand(
                self.fcn, self.action_space, self.workspace, self.heightmap_resolution, self.device, self.learning_rate,
                self.discount, self.NUM_PRIMITIVES, self.SUPERVISED_LEARNING,
                self.buffer_type == Constants.PRIORITIZED_BUFFER, self.num_rotations,
                self.HALF_ROTATION, self.patch_size
            )

        self.agent.train()

    def create_replay_buffer_(self):

        if self.buffer_type == Constants.PRIORITIZED_BUFFER:
            self.replay_buffer = PrioritizedQLearningBuffer(self.buffer_size, self.per_alpha, PR_NORMAL)
        elif self.buffer_type == Constants.PRIORITIZED_BUFFER_EXPERT:
            self.replay_buffer = PrioritizedQLearningBuffer(self.buffer_size, self.per_alpha, PR_EXPERT)
        elif self.buffer_type == Constants.EXPERT_BUFFER:
            self.replay_buffer = QLearningBufferExpert(self.buffer_size)
        elif self.buffer_type == Constants.BUFFER:
            self.replay_buffer = QLearningBuffer(self.buffer_size)

    def create_schedules_(self):

        self.exploration = LinearSchedule(
            schedule_timesteps=self.exploration_steps, initial_p=self.init_eps, final_p=self.final_eps
        )
        self.coefficient_schedule = LinearSchedule(
            schedule_timesteps=self.exploration_steps, initial_p=self.init_coef, final_p=self.final_coef
        )
        self.p_beta_schedule = LinearSchedule(
            schedule_timesteps=self.max_episodes, initial_p=self.per_beta, final_p=1.0
        )

    def get_diag_length_(self):

        diag_length = float(self.heightmap_size) * np.sqrt(2)
        return int(np.ceil(diag_length / 32) * 32)

    def get_patch_shape_(self):

        return self.PATCH_CHANNEL, self.patch_size, self.patch_size

    def load_expert_transitions(self, load_path):

        dataset = ArrayDataset(None)
        dataset.load_hdf5(load_path)

        self.logger.info("load expert data path: {:s}".format(load_path))
        self.logger.info("{:d} transitions, timestamp: {:s}".format(
            dataset.size, dataset.metadata[Constants.TIMESTAMP]
        ))

        # custom assert, not applicable to all sizes etc.
        assert len(dataset[Constants.OBS].shape) == 3
        assert dataset[Constants.OBS].shape[1] == 90
        assert dataset[Constants.OBS].shape[1] == dataset[Constants.OBS].shape[2]

        assert len(dataset[Constants.HAND_OBS].shape) == 3
        assert dataset[Constants.HAND_OBS].shape[1] == 24
        assert dataset[Constants.HAND_OBS].shape[1] == dataset[Constants.HAND_OBS].shape[2]

        for i in range(dataset.size):
            # these tuples will be decomposed back into obs and hand_obs in the fcn
            # quite unnecessary, but I would need to make too many changes to fix that
            state = (torch.tensor(dataset[Constants.OBS][i], device="cpu"),
                     torch.tensor(dataset[Constants.HAND_OBS][i], device="cpu"))
            next_state = (torch.tensor(dataset[Constants.NEXT_OBS][i], device="cpu"),
                          torch.tensor(dataset[Constants.NEXT_HAND_OBS][i], device="cpu"))

            # I'm changing types here because this data must have the same form as the data that is collected
            # on-policy after expert pre-training
            if Constants.STEPS_LEFT in dataset:
                steps_left = torch.tensor(dataset[Constants.STEPS_LEFT][i], device="cpu")
            else:
                steps_left = torch.tensor(1, device="cpu")

            action = dataset[Constants.ACTIONS][i]
            if len(action.shape) == 0 or action.shape[0] <= 1:
                action = [action // 90, action % 90]

            t = ExpertTransition(
                torch.tensor(dataset[Constants.HAND_BITS][i].astype(np.float32), device="cpu"), state,
                torch.tensor(action, device="cpu").long(),
                torch.tensor([float(dataset[Constants.REWARDS][i])], device="cpu"),
                torch.tensor(dataset[Constants.NEXT_HAND_BITS][i].astype(np.float32), device="cpu"), next_state,
                torch.tensor(dataset[Constants.DONES][i].astype(np.float32), device="cpu"),
                steps_left, torch.tensor(1, device="cpu")
            )

            self.replay_buffer.add(t)

    def load_agent(self, load_path):

        self.agent.fcn.load_state_dict(torch.load(load_path, map_location=self.device))

    def save_agent(self, save_path):
        # TODO: only saves the policy net
        dir_path = os.path.dirname(save_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        torch.save(self.agent.fcn.state_dict(), save_path)

    def close_envs(self):
        # TODO: only saves the policy net
        self.envs.close()

    def create_envs_(self):
        self.envs = env_factory.createEnvs(
            self.num_processes, "rl", self.simulator, "house_building_x", self.env_config,
            self.planner_config
        )
