import numpy as np
from scipy.special import softmax
from ...models.dqn.utils.schedules import LinearSchedule
from ...constants import Constants


class SoftmaxPolicy:

    def __init__(self, config):

        self.init_tau = config[Constants.INIT_TAU]
        self.final_tau = config[Constants.FINAL_TAU]
        self.exploration_steps = config[Constants.EXPLORATION_STEPS]
        self.num_actions = config[Constants.NUM_ACTIONS]

        self.exploration_schedule = None
        self.reset()

    def act(self, state, qs, timestep, evaluation=False):

        if evaluation:
            return np.argmax(qs)

        s = softmax(qs / self.exploration_schedule.value(timestep))
        return np.random.choice(list(range(self.num_actions)), p=s)

    def reset(self):

        if self.exploration_steps > 0:
            self.exploration_schedule = LinearSchedule(
                schedule_timesteps=self.exploration_steps, initial_p=self.init_tau, final_p=self.final_tau
            )
