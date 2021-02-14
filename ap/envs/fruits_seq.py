import numpy as np
from .env import Env
from ..constants import Constants


class FruitsSeq(Env):
    # this class is the same as envs/fruits.py, except for checking for a sequence of picked fruits
    # (order matters)
    # the agent is given the HxWx|F| tensor as well as information about what it is holding
    WRONG_PICK_REWARD = - 0.1
    STEP_REWARD = 0.0
    POSITIVE_REWARD = 1.0

    class Fruit:

        def __init__(self, index, position, env_size):

            self.index = index
            self.position = position

            self.x = self.position // env_size
            self.y = self.position % env_size

        def same_position(self, fruit):

            return self.position == fruit.position

        def __str__(self):

            s = "Fruit {:d}: position ({:d},{:d})/{:d}, ".format(self.index, self.x, self.y, self.position)

            return s

    def __init__(self, num_fruits=5, size=5, max_steps=30):

        super(FruitsSeq, self).__init__()

        self.num_fruits = num_fruits
        self.size = size
        self.max_steps = max_steps

        self.goal = None
        self.hand = None
        self.fruits = None
        self.current_step = None

        self.reset_goal()
        self.reset()

    def reset(self):

        self.fruits = []
        self.hand = []
        self.current_step = 0

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

        num_fruits = np.random.randint(1, self.num_fruits + 1)
        fruits = list(np.random.choice(list(range(self.num_fruits)), size=num_fruits, replace=False))

        self.goal = fruits

    def step(self, action):

        num_actions = self.size ** 2
        assert 0 <= action < num_actions
        reached_goal = False
        done = False
        negative_reward = False

        if action == num_actions - 1:
            reached_goal = self.check_goal_()
        else:
            next_fruit_to_pick = self.get_next_fruit_in_sequence_()

            for fruit in self.fruits:
                if fruit.position == action:

                    if fruit.index == next_fruit_to_pick:
                        self.hand.append(next_fruit_to_pick)
                    elif fruit.index not in self.hand:
                        negative_reward = True

        self.current_step += 1

        if self.current_step >= self.max_steps or reached_goal:
            done = True

        reward = self.STEP_REWARD

        if negative_reward:
            reward = self.WRONG_PICK_REWARD
        if reached_goal:
            reward = self.POSITIVE_REWARD

        return self.get_state(), reward, done, {Constants.REACHED_GOAL: reached_goal}

    def get_state(self):

        image = np.zeros((self.size, self.size, self.num_fruits), dtype=np.float32)
        hand = np.zeros((self.num_fruits, self.num_fruits), dtype=np.float32)

        for fruit in self.fruits:
            # mark the fruit location with 1
            image[fruit.x, fruit.y, fruit.index] = 1.0

        for idx, fruit_idx in enumerate(self.hand):
            hand[idx, fruit_idx] = 1.0

        return image, hand

    def get_abstract_action(self, action):

        num_actions = self.size ** 2

        if action == num_actions - 1:
            if self.check_goal_():
                return 0
            else:
                return 1
        else:
            offset = 2

            for fruit in self.fruits:

                if action == fruit.position:
                    if fruit.index == self.get_next_fruit_in_sequence_():
                        return offset
                    elif fruit.index in self.hand:
                        return offset + 1
                    else:
                        return offset + 2

        return offset + 3

    def get_abstract_action_name(self, abstract_action):

        names = [
            "done", "not yet done", "pick fruit in sequence", "pick fruit already in hand",
            "pick fruit not in sequence", "empty pick"
        ]

        return names[abstract_action]

    def get_next_fruit_in_sequence_(self):

        index = len(self.hand)

        if index >= len(self.goal):
            return None

        return self.goal[index]

    def check_goal_(self):

        if len(self.hand) == len(self.goal):
            # we shouldn't be allowed to pick a fruit that's not next in the sequence
            for i in range(len(self.hand)):
                assert self.hand[i] == self.goal[i]

            return True

        return False

    def create_fruit_(self, index):

        num_cells = self.size ** 2
        position = int(np.random.randint(0, num_cells - 1))

        return self.Fruit(index, position, self.size)

    def get_optimal_action_(self):

        if self.check_goal_():
            return int(self.size ** 2) - 1
        else:
            next_fruit = self.get_next_fruit_in_sequence_()
            for fruit in self.fruits:
                if fruit.index == next_fruit:
                    return fruit.position

    @staticmethod
    def state_to_image(state):

        r = np.sum(state, axis=2)
        g = np.zeros_like(r)
        b = np.zeros_like(r)

        return np.stack([r, g, b], axis=2)
