import numpy as np
from ...models.dqn.utils.schedules import LinearSchedule
from ...constants import Constants


class EpsGreedyPolicy:

    def __init__(self, config):

        self.exploration_steps = config[Constants.EXPLORATION_STEPS]
        self.num_actions = config[Constants.NUM_ACTIONS]

        self.init_explore = 1.0
        self.final_explore = 0.1

        self.exploration_schedule = None
        self.reset()

    def act(self, state, qs, timestep, evaluation=False):

        if self.exploration_schedule is not None and np.random.rand() < \
                self.exploration_schedule.value(timestep) and not evaluation:

            action = np.random.randint(self.num_actions)

        else:

            action = np.argmax(qs)

        return action

    def reset(self):

        if self.exploration_steps > 0:
            self.exploration_schedule = LinearSchedule(
                schedule_timesteps=self.exploration_steps, initial_p=self.init_explore, final_p=self.final_explore
            )
