import numpy as np
import torch
from torch import nn
from ....constants import Constants
from ....envs.fruits_seq import FruitsSeq
from ....modules.FCEncoder import FCEncoder
from ....modules.FlatStatesMultipleConcant import FlatStatesMultipleConcat
from ....run.offline.fruits.RunActorMimicV2 import RunActorMimicV2
from ....utils import runner as runner_utils


class RunActorMimicV2Seq(RunActorMimicV2):

    def learn_step_(self, opt):

        all_states = []
        all_policies = []
        all_task_ids = []

        for goal_idx in range(len(self.goals)):

            states, policies, _, _, _ = self.replay_buffers[goal_idx].sample(self.batch_size)
            states = [np.array([s[i] for s in states], dtype=np.float32) for i in [0, 1]]

            all_states.append(states)
            all_policies.append(policies)
            all_task_ids.append(np.array([goal_idx] * self.batch_size, dtype=np.long))

        all_states = [runner_utils.other_to_torch(
            np.concatenate([s[i] for s in all_states], axis=0), self.device
        ) for i in [0, 1]]
        all_policies = runner_utils.other_to_torch(np.concatenate(all_policies, axis=0), self.device)
        all_task_ids = runner_utils.other_to_torch(np.concatenate(all_task_ids, axis=0), self.device)

        opt.zero_grad()
        loss = self.student.compute_loss(all_states, all_task_ids, all_policies)
        loss.backward()
        opt.step()

        return loss.detach().cpu().numpy()

    def build_envs_(self):

        self.envs = []
        for goal in self.goals:
            env = FruitsSeq(num_fruits=self.num_fruits, max_steps=30)
            env.goal = goal
            self.envs.append(env)

    @torch.no_grad()
    def get_student_action_(self, state, task_idx, epsilon):

        state = [runner_utils.other_to_torch(s[None], self.device) for s in state]
        policy = self.student.forward(state, task_idx)[0].cpu().numpy()

        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(len(policy))
        else:
            action = np.argmax(policy)

        return action

    @torch.no_grad()
    def get_teacher_policy_(self, state, teacher):

        state = [runner_utils.other_to_torch(s[None], self.device) for s in state]

        qs = teacher(state)
        qs = qs / self.tau
        return nn.functional.softmax(qs, dim=1)[0].cpu().numpy()

    def build_encoder_(self, student=False):

        state_shape = [x.shape for x in self.envs[0].get_state()]

        flatten_states = FlatStatesMultipleConcat()
        flatten_states.output_size = int(np.prod(state_shape[0]) + np.prod(state_shape[1]))

        if student:
            neurons = self.student_config[Constants.NEURONS]
        else:
            neurons = [256, 256]

        fc_config = {
            Constants.INPUT_SIZE: flatten_states.output_size,
            Constants.NEURONS: neurons,
            Constants.USE_BATCH_NORM: False,
            Constants.USE_LAYER_NORM: False,
            Constants.ACTIVATION_LAST: True
        }

        fc_encoder = FCEncoder(fc_config, self.logger)

        encoder = nn.Sequential(flatten_states, fc_encoder)
        encoder.output_size = fc_encoder.output_size

        return encoder
