import numpy as np
from .env import Env
from ..constants import Constants


class Fruits(Env):

    WRONG_PICK_REWARD = - 0.1
    STEP_REWARD = 0.0
    POSITIVE_REWARD = 1.0

    class Fruit:

        def __init__(self, index, position, env_size):
            # fruit has an index, a position and an indicator if it was picked (active)
            self.index = index
            self.position = position
            self.active = False

            self.x = self.position // env_size
            self.y = self.position % env_size

        def same_position(self, fruit):

            return self.position == fruit.position

        def __str__(self):

            s = "Fruit {:d}: position ({:d},{:d})/{:d}, ".format(self.index, self.x, self.y, self.position)

            if self.active:
                s += "active"
            else:
                s += "inactive"

            return s

    def __init__(self, num_fruits=5, size=5, max_steps=30, no_start=True, no_wrong_pick=True):
        # no_start should be True
        # it's from an old version where the agent needed to execute action 0 in order to start picking
        super(Fruits, self).__init__()

        self.num_fruits = num_fruits
        self.size = size
        self.max_steps = max_steps
        self.no_start = no_start
        self.no_wrong_pick = no_wrong_pick

        self.goal = None
        self.fruits = []
        self.started = None
        self.current_step = None

        self.reset_goal()
        self.reset()

    def reset(self):
        # reset fruits
        self.fruits = []
        self.started = False
        self.current_step = 0

        if self.no_start:
            self.started = True

        for i in range(self.num_fruits):

            while True:

                fruit = self.create_fruit_(i)

                done = True
                for fruit2 in self.fruits:
                    if fruit.same_position(fruit2):
                        done = False

                if done:
                    break

            self.fruits.append(fruit)

        return self.get_state()

    def reset_goal(self):
        # sample a random goal (a set of fruits to pick up)
        # self.goal is a list of indices of the goal fruits
        num_fruits = np.random.randint(1, self.num_fruits + 1)
        fruits = list(np.random.choice(list(range(self.num_fruits)), size=num_fruits, replace=False))

        self.goal = fruits

    def step(self, action):
        # execute a step in the env given an action
        num_actions = self.size ** 2
        assert 0 <= action < num_actions
        reached_goal = False
        done = False
        negative_reward = False

        if action == 0 and not self.no_start:
            # old version of the env had action 0 = start picking
            # now self.no_start should be always true
            self.started = True
        elif action == num_actions - 1:
            # action 24 (last action) is for saying that you are done picking
            reached_goal = self.check_goal_()
        else:
            # started should be always true
            if self.started:
                for fruit in self.fruits:
                    if fruit.position == action:
                        if self.no_wrong_pick:
                            # no wrong pick means that the agent gets penalized for picking the wrong fruit
                            # I use this in my paper
                            if fruit.index in self.goal:
                                fruit.active = True
                            else:
                                negative_reward = True
                        else:
                            fruit.active = not fruit.active

        self.current_step += 1

        if self.current_step >= self.max_steps or reached_goal:
            # we are done if we reached step limit or we picked up the right fruits and executed the end action
            done = True

        reward = self.STEP_REWARD

        if negative_reward:
            reward = self.WRONG_PICK_REWARD
        if reached_goal:
            reward = self.POSITIVE_REWARD

        return self.get_state(), reward, done, {Constants.REACHED_GOAL: reached_goal}

    def get_state(self):
        # get state as a tensor of HxWx|F|
        # H: height, W: width, |F|: number of fruits
        # the last dimension is used to one-hot encode fruits
        # zero everywhere means no fruit at that position
        image = np.zeros((self.size, self.size, self.num_fruits + 1), dtype=np.float32)

        if self.started and not self.no_start:
            # agent pressed the start flag
            image[0, 0, self.num_fruits] = 1.0

        for fruit in self.fruits:
            # mark the fruit location with 1
            image[fruit.x, fruit.y, fruit.index] = 1.0
            if fruit.active:
                # make if a fruit has been picked
                image[fruit.x, fruit.y, self.num_fruits] = 1.0

        return image

    def get_abstract_action(self, action):
        # used for debugging
        num_actions = self.size ** 2
        offset = 0

        if not self.started:
            if action == 0:
                return offset
            else:
                return offset + 1
        else:
            offset += 2
            if action == 0:
                return offset
            elif action == num_actions - 1:
                offset += 1
                if self.check_goal_():
                    return offset
                else:
                    return offset + 1
            else:
                offset += 3
                for fruit in self.fruits:
                    if action == fruit.position:
                        if fruit.active:
                            if fruit.index in self.goal:
                                return offset
                            else:
                                return offset + 1
                        else:
                            if fruit.index in self.goal:
                                return offset + 2
                            else:
                                return offset + 3

                offset += 4
                return offset

    def get_abstract_action_name(self, abstract_action):
        # used for debugging
        names = [
            "start", "start not active", "start already active", "finish", "not yet done", "deactivate goal fruit",
            "deactivate distractor fruit", "activate goal fruit", "activate distractor fruit", "do nothing"
        ]

        return names[abstract_action]

    def check_goal_(self):
        # check if goal reached
        if not self.started:
            return False

        for fruit in self.fruits:
            if fruit.active and fruit.index not in self.goal:
                return False
            if not fruit.active and fruit.index in self.goal:
                return False

        return True

    def create_fruit_(self, index):
        # create a fruit at a random position
        # fruit index is also its position in self.fruits, bit redundant
        num_cells = self.size ** 2

        if self.no_start:
            start_idx = 0
        else:
            start_idx = 1

        position = int(np.random.randint(start_idx, num_cells - 1))

        return self.Fruit(index, position, self.size)

    def get_next_fruit_to_pick_(self):
        # hand-crafted optimal policy, used to generate expert demonstrations
        for fruit_idx in self.goal:

            fruit = self.fruits[fruit_idx]

            if not fruit.active:
                # goal fruit not picked
                return fruit_idx

        # already picked all fruits
        return None

    def get_optimal_action_(self):
        # hand-crafted optimal policy, used to generate expert demonstrations
        assert self.no_start

        if self.check_goal_():
            # "end" action
            return int(self.size ** 2) - 1
        else:
            next_fruit_idx = self.get_next_fruit_to_pick_()
            return self.fruits[next_fruit_idx].position

    @staticmethod
    def state_to_image(state):
        # turn state into an RGB image, doesn't show the fruit indices
        r = np.sum(state[:, :, :-1], axis=2)
        g = state[:, :, -1]
        b = np.zeros_like(g)

        return np.stack([r, g, b], axis=2)
