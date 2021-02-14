from datetime import datetime
import numpy as np
import copy as cp
import torch
from ....constants import Constants
from ....helping_hands_rl_envs import env_factory
from ....utils.dataset import ListDataset
from ....utils import pb as pb_utils


class RunDeconstruct:

    def __init__(self, runner_config, env_config, planner_config, logger):

        rc = runner_config
        self.simulator = rc[Constants.SIMULATOR]
        self.robot = rc[Constants.ROBOT]
        self.workspace = rc[Constants.WORKSPACE]
        self.heightmap_size = rc[Constants.HEIGHTMAP_SIZE]
        self.num_processes = rc[Constants.NUM_PROCESSES]
        self.num_objects = rc[Constants.NUM_OBJECTS]
        self.num_rotations = rc[Constants.NUM_ROTATIONS]
        self.action_sequence = rc[Constants.ACTION_SEQUENCE]
        self.num_samples = rc[Constants.NUM_SAMPLES]
        self.save_path = rc[Constants.SAVE_PATH]

        self.workspace_size = pb_utils.get_workspace_size(self.workspace)
        self.heightmap_resolution = pb_utils.get_heightmap_resolution(self.workspace_size, self.heightmap_size)
        self.rotations = self.create_rotations_()

        self.env_config = env_config
        self.planner_config = planner_config
        self.logger = logger

        self.create_envs_()

    def collect(self):

        dataset = ListDataset()

        num_attempts = 0
        num_episodes = 0

        steps = [0 for i in range(self.num_processes)]
        local_hand_bit = [[] for i in range(self.num_processes)]
        local_obs = [[] for i in range(self.num_processes)]
        local_hand_obs = [[] for i in range(self.num_processes)]
        local_action = [[] for i in range(self.num_processes)]
        local_reward = [[] for i in range(self.num_processes)]

        states, in_hands, obs = self.envs.reset()

        while dataset.get_size() < self.num_samples:

            if num_episodes % 100 == 0:
                self.logger.info("{:d} transitions collected".format(dataset.get_size()))

            plan_actions = self.envs.getNextAction()

            actions_star_idx, actions_star = self.get_action_from_plan_(plan_actions)
            actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)

            states_, in_hands_, obs_, rewards, dones = self.envs.step(actions_star, auto_reset=False)

            state_id = self.action_sequence.find('p')
            dones[actions_star[:, state_id] + states_ != 1] = 1

            for i in range(self.num_processes):

                local_hand_bit[i].append(states[i])
                local_obs[i].append(obs[i])
                # we append the NEXT in hand image since we are doing deconstructions
                # so at the end we reverse the experience so that we are constructing stuff
                local_hand_obs[i].append(in_hands_[i])
                local_action[i].append(actions_star_idx[i])
                local_reward[i].append(rewards[i])

            steps = list(map(lambda x: x + 1, steps))

            done_idxes = torch.nonzero(dones).squeeze(1)
            if done_idxes.shape[0] != 0:

                empty_in_hands = self.envs.getEmptyInHand()

                reset_states_, reset_in_hands_, reset_obs_ = self.envs.reset_envs(done_idxes)

                for i, idx in enumerate(done_idxes):

                    local_hand_bit[idx].append(cp.deepcopy(states_[idx]))
                    local_obs[idx].append(cp.deepcopy(obs_[idx]))
                    local_hand_obs[idx].append(empty_in_hands[idx])

                    num_attempts += 1

                    if self.steps_valid_(steps[idx]) and self.states_valid_(local_hand_bit[idx]) and \
                            self.rewards_valid_(local_reward[idx]):

                        num_episodes += 1

                        for j in range(len(local_reward[idx])):

                            dataset.add(Constants.HAND_BITS, self.to_numpy_(local_hand_bit[idx][j + 1]))
                            dataset.add(Constants.OBS, self.to_numpy_(local_obs[idx][j + 1])[:, :, 0])
                            dataset.add(Constants.HAND_OBS, self.to_numpy_(local_hand_obs[idx][j + 1])[:, :, 0])
                            dataset.add(Constants.ACTIONS, self.to_numpy_(local_action[idx][j]))
                            dataset.add(Constants.REWARDS, self.to_numpy_(local_reward[idx][j]))
                            dataset.add(Constants.NEXT_HAND_BITS, self.to_numpy_(local_hand_bit[idx][j]))
                            dataset.add(Constants.NEXT_OBS, self.to_numpy_(local_obs[idx][j])[:, :, 0])
                            dataset.add(Constants.NEXT_HAND_OBS, self.to_numpy_(local_hand_obs[idx][j])[:, :, 0])
                            dataset.add(Constants.DONES, j == 0)
                            dataset.add(Constants.STEPS_LEFT, float(j))

                    states_[idx] = reset_states_[i]
                    obs_[idx] = reset_obs_[i]

                    steps[idx] = 0
                    local_hand_bit[idx] = []
                    local_obs[idx] = []
                    local_hand_obs[idx] = []
                    local_action[idx] = []
                    local_reward[idx] = []

            states = cp.deepcopy(states_)
            obs = cp.deepcopy(obs_)

        dataset = dataset.to_array_dataset({
            Constants.HAND_BITS: np.int32, Constants.OBS: np.float32, Constants.HAND_OBS: np.float32,
            Constants.ACTIONS: np.int32, Constants.REWARDS: np.float32, Constants.NEXT_HAND_BITS: np.int32,
            Constants.NEXT_OBS: np.float32, Constants.NEXT_HAND_OBS: np.float32, Constants.DONES: np.bool,
            Constants.STEPS_LEFT: np.float32
        })
        dataset.metadata = {
            Constants.NUM_EXP: dataset.size, Constants.TIMESTAMP: str(datetime.today())
        }
        dataset.save_hdf5(self.save_path)

        return num_attempts, num_episodes

    def create_envs_(self):

        self.envs = env_factory.createEnvs(
            self.num_processes, "rl", self.simulator, "house_building_x_deconstruct", self.env_config,
            self.planner_config
        )

    def get_action_from_plan_(self, plan):

        x = plan[:, 0:1]
        y = plan[:, 1:2]
        rot = plan[:, 2:3]
        states = plan[:, 3:4]
        pixel_x = ((x - self.workspace[0][0]) / self.heightmap_resolution).long()
        pixel_y = ((y - self.workspace[1][0]) / self.heightmap_resolution).long()
        pixel_x = torch.clamp(pixel_x, 0, self.heightmap_size - 1)
        pixel_y = torch.clamp(pixel_y, 0, self.heightmap_size - 1)
        rot_id = (rot.expand(-1, self.num_rotations) - self.rotations).abs().argmin(1).unsqueeze(1)

        x = (pixel_x.float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixel_y.float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)
        rot = self.rotations[rot_id]
        actions = torch.cat((x, y, rot), dim=1)
        action_idx = torch.cat((pixel_x, pixel_y, rot_id), dim=1)
        return action_idx, actions

    @staticmethod
    def states_valid_(states_list):

        if len(states_list) < 2:
            return False
        for i in range(1, len(states_list)):
            if states_list[i] != 1 - states_list[i-1]:
                return False
        return True

    @staticmethod
    def rewards_valid_(reward_list):

        if reward_list[0] != 1:
            return False
        for i in range(1, len(reward_list)):
            if reward_list[i] != 0:
                return False
        return True

    def create_rotations_(self):

        return torch.tensor([np.pi / self.num_rotations * i for i in range(self.num_rotations)])

    def steps_valid_(self, steps):

        return (self.num_objects - 2) * 2 <= steps <= self.num_objects * 2

    @staticmethod
    def to_numpy_(x):

        return x.detach().cpu().numpy()
