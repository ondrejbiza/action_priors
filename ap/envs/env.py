from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Env(ABC):

    def reset(self) -> np.ndarray:
        # reset env and return current state
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # take a step in the env and return next state, reward, done and additional information in a dict
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        # get the current state
        pass
