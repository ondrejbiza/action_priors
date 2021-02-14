import numpy as np


class Result:

    def __init__(self):

        self.d = {}

    def get_mean_window(self, key, size):

        return np.mean(self.d[key][-size:])

    def register(self, key):

        self.d[key] = []

    def add(self, key, value):

        self.d[key].append(value)

    def add_pytorch(self, key, value):

        self.add(key, value.detach().cpu().numpy())

    def reset(self, key):

        self.register(key)

    def mean(self, key):

        return np.mean(self.d[key])

    def sum(self, key):

        return np.sum(self.d[key])

    def discounted_sum(self, key, discount):

        return np.sum([(discount ** i) * r for i, r in enumerate(self.d[key])])

    def count_steps(self, key):

        return len(self.d[key])

    def __getitem__(self, key):

        return self.d[key]

    def __setitem__(self, key, value):

        self.d[key] = value

    def __contains__(self, key):

        return key in self.d
