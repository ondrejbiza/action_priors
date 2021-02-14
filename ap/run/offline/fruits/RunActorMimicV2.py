import copy as cp
import numpy as np
import torch
from torch import nn
from torch import optim
from ....constants import Constants
from ....envs.fruits import Fruits
from ....modules.ConvEncoder import ConvEncoder
from ....modules.FCEncoder import FCEncoder
from ....models.dqn.DQN import DQN
from ....models.homo.MultiHeadPolicy import MultiHeadPolicy
from ....models.dqn.utils.replay_buffer import ReplayBuffer
from ....utils import runner as runner_utils
from ....utils.result import Result


class RunActorMimicV2:

    NUM_EVAL_EPISODES = 5
    EVAL_FREQUENCY = 1000

    def __init__(self, runner_config, student_config, teacher_config, logger):

        self.student_config = student_config
        self.teacher_config = teacher_config
        self.logger = logger

        rc = runner_config
        self.teacher_load_paths = rc[Constants.TEACHER_LOAD_PATHS]
        self.goals = rc[Constants.GOALS]
        self.num_fruits = rc[Constants.NUM_FRUITS]
        self.tau = rc[Constants.TAU]
        self.learning_rate = rc[Constants.LEARNING_RATE]
        self.weight_decay = rc[Constants.WEIGHT_DECAY]
        self.epsilon = rc[Constants.EPSILON]
        self.buffer_size = rc[Constants.BUFFER_SIZE]
        self.batch_size = rc[Constants.BATCH_SIZE]
        self.max_steps = rc[Constants.MAX_STEPS]
        self.max_episodes = rc[Constants.MAX_EPISODES]
        self.device = rc[Constants.DEVICE]

        self.training_result = None

        assert len(self.teacher_load_paths) == len(self.goals)

        self.envs = None
        self.build_envs_()

        self.student = self.build_student_()

        self.teachers = []
        for load_path in self.teacher_load_paths:
            self.teachers.append(self.build_teacher_(load_path))

        self.replay_buffers = None
        self.setup_buffers_()

    def train_model(self):

        opt = self.get_opt()

        step = 0
        episode = 0

        result = Result()
        result.register(Constants.TOTAL_REWARDS)
        result.register(Constants.DISCOUNTED_REWARDS)
        for goal_idx in range(len(self.goals)):
            result.register("tmp_rewards_{:d}".format(goal_idx))
        result.register(Constants.EVAL_TOTAL_REWARDS)
        result.register(Constants.EVAL_DISCOUNTED_TOTAL_REWARDS)
        result.register(Constants.EVAL_NUM_STEPS)
        result.register(Constants.LOSS)

        states = [env.reset() for env in self.envs]

        while True:

            # maybe evaluate
            if step > 0 and step % self.EVAL_FREQUENCY == 0:
                eval_total, eval_discounted, eval_steps = self.evaluate_()
                result.add(Constants.EVAL_TOTAL_REWARDS, eval_total)
                result.add(Constants.EVAL_DISCOUNTED_TOTAL_REWARDS, eval_discounted)
                result.add(Constants.EVAL_NUM_STEPS, eval_steps)

                print("eval total:".format(str(eval_total)))
                for task_idx in range(len(self.goals)):
                    print("    Task {:d}: {:.1f}".format(task_idx, eval_total[task_idx] * 100))

            # maybe stop
            if self.should_terminate_(step, episode):
                break

            # maybe log
            if step > 0 and step % 100 == 0:
                self.logger.info("step {:d}: {:.3f} loss, {:.2f}r 100ep, {:.2f}dr 100 ep, {:.2f}exp".format(
                    step, np.mean(result[Constants.LOSS][-100:]),
                    result.get_mean_window(Constants.TOTAL_REWARDS, 100),
                    result.get_mean_window(Constants.DISCOUNTED_REWARDS, 100),
                    self.epsilon
                ))

            actions = [self.get_student_action_(state, goal_idx, self.epsilon) for goal_idx, state in enumerate(states)]
            teacher_policies = [self.get_teacher_policy_(state, self.teachers[goal_idx]) for goal_idx, state in enumerate(states)]

            # viz
            # if step > 0 and step % self.EVAL_FREQUENCY == 0:
            #     for f in self.env.fruits:
            #         print(f)
            #     print("task: {:d}, {:s}".format(goal_idx, str(self.goals[goal_idx])))
            #     print("action: {:d}".format(action))
            #     print("teacher policy: {:s}".format(str(teacher_policy)))
            #     plt.imshow(Fruits.state_to_image(state))
            #     plt.show()

            next_states = []
            for goal_idx, action in enumerate(actions):

                next_state, reward, done, _ = self.envs[goal_idx].step(action)
                result.add("tmp_rewards_{:d}".format(goal_idx), reward)

                # I don't need reward, next state and done ...
                self.replay_buffers[goal_idx].add(states[goal_idx], teacher_policies[goal_idx], reward, next_state, done)

                if done:
                    next_states.append(self.envs[goal_idx].reset())

                    result.add(Constants.TOTAL_REWARDS, result.sum("tmp_rewards_{:d}".format(goal_idx)))
                    result.add(Constants.DISCOUNTED_REWARDS, result.discounted_sum("tmp_rewards_{:d}".format(goal_idx), 0.9))
                    result.reset("tmp_rewards_{:d}".format(goal_idx))

                    if goal_idx == 0:
                        # I want the average number of episodes over all envs
                        # simple hack is to long only at the first one
                        episode += 1
                else:
                    next_states.append(next_state)

            states = next_states

            step += 1

            # learn, samples from buffers for all tasks
            if step > 10:
                loss = self.learn_step_(opt)
                result.add(Constants.LOSS, loss)

        self.training_result = result

    def learn_step_(self, opt):

        all_states = []
        all_policies = []
        all_task_ids = []

        for goal_idx in range(len(self.goals)):
            states, policies, _, _, _ = self.replay_buffers[goal_idx].sample(self.batch_size)
            all_states.append(states)
            all_policies.append(policies)
            all_task_ids.append(np.array([goal_idx] * self.batch_size, dtype=np.long))

        all_states = runner_utils.states_to_torch(np.concatenate(all_states, axis=0), self.device)
        all_policies = runner_utils.other_to_torch(np.concatenate(all_policies, axis=0), self.device)
        all_task_ids = runner_utils.other_to_torch(np.concatenate(all_task_ids, axis=0), self.device)

        opt.zero_grad()
        loss = self.student.compute_loss(all_states, all_task_ids, all_policies)
        loss.backward()
        opt.step()

        return loss.detach().cpu().numpy()

    def should_terminate_(self, step, episode):

        if self.max_steps is not None and step >= self.max_steps:
            return True

        if self.max_episodes is not None and episode >= self.max_episodes:
            return True

        return False

    def evaluate_(self):

        # instead of initializing a new env I just deepcopy the training one
        env = cp.deepcopy(self.envs[0])

        task_rewards = []
        task_returns = []
        task_num_steps = []

        for goal_idx in range(len(self.goals)):

            env.goal = self.goals[goal_idx]

            total_rewards = []
            discounted_rewards = []
            num_steps = []

            for i in range(self.NUM_EVAL_EPISODES):

                state = env.reset()
                tmp_rewards = []

                while True:

                    action = self.get_student_action_(state, goal_idx, 0.0)
                    next_state, reward, done, _ = env.step(action)
                    tmp_rewards.append(reward)

                    if done:
                        break

                    state = next_state

                total_rewards.append(np.sum(tmp_rewards))
                discounted_rewards.append(np.sum([(0.9 ** i) * r for i, r in enumerate(tmp_rewards)]))
                num_steps.append(len(tmp_rewards))

            task_rewards.append(np.mean(total_rewards))
            task_returns.append(np.mean(discounted_rewards))
            task_num_steps.append(np.mean(num_steps))

        return np.array(task_rewards), np.array(task_returns), np.array(task_num_steps)

    def build_envs_(self):

        self.envs = []
        for goal in self.goals:
            env = Fruits(num_fruits=self.num_fruits, no_start=True, max_steps=30, no_wrong_pick=True)
            env.goal = goal
            self.envs.append(env)

    @torch.no_grad()
    def get_student_action_(self, state, task_idx, epsilon):

        state = state[None]
        state = runner_utils.states_to_torch(state, self.device)

        policy = self.student.forward(state, task_idx)[0].cpu().numpy()

        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(len(policy))
        else:
            action = np.argmax(policy)

        return action

    @torch.no_grad()
    def get_teacher_policy_(self, state, teacher):

        state = state[None]
        state = runner_utils.states_to_torch(state, self.device)

        qs = teacher(state)
        qs = qs / self.tau
        return nn.functional.softmax(qs, dim=1)[0].cpu().numpy()

    def setup_buffers_(self):

        self.replay_buffers = []

        for _ in range(len(self.goals)):
            self.replay_buffers.append(ReplayBuffer(self.buffer_size))

    def build_student_(self):

        encoder = self.build_encoder_(student=True)
        return MultiHeadPolicy(encoder, self.student_config, self.logger).to(self.device)

    def build_teacher_(self, load_path):

        encoder = self.build_encoder_()
        # I don't specify a policy because it is used to sample a single actions, not to return a distribution
        teacher = DQN(encoder, None, self.teacher_config, self.logger).to(self.device)
        teacher.load(load_path)
        return teacher

    def build_encoder_(self, student=False):

        conv_config = {
            Constants.INPUT_SIZE: self.envs[0].get_state().shape,
            Constants.FILTER_SIZES: [],
            Constants.FILTER_COUNTS: [],
            Constants.STRIDES: [],
            Constants.USE_BATCH_NORM: False,
            Constants.ACTIVATION_LAST: True,
            Constants.FLAT_OUTPUT: True
        }

        conv_encoder = ConvEncoder(conv_config, self.logger)

        if student:
            neurons = self.student_config[Constants.NEURONS]
        else:
            neurons = [256, 256]

        fc_config = {
            Constants.INPUT_SIZE: conv_encoder.output_size,
            Constants.NEURONS: neurons,
            Constants.USE_BATCH_NORM: False,
            Constants.USE_LAYER_NORM: False,
            Constants.ACTIVATION_LAST: True
        }

        fc_encoder = FCEncoder(fc_config, self.logger)

        encoder = nn.Sequential(conv_encoder, fc_encoder)
        encoder.input_size = conv_encoder.input_size
        encoder.output_size = fc_encoder.output_size

        return encoder

    def get_opt(self):

        params = self.student.parameters()
        return optim.Adam(params, self.learning_rate, weight_decay=self.weight_decay)

    def save_student_encoder(self, path):

        torch.save(self.student.encoder.state_dict(), path)
