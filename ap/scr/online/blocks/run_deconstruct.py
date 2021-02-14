import sys
sys.path.insert(0, "ap")

import numpy as np
from sacred import Experiment
from ....run.online.blocks.RunDeconstruct import RunDeconstruct
from ....constants import Constants
from ....utils.logger import Logger
from .... import paths

ex = Experiment("blocks_deconstruct")
ex.add_config(paths.CFG_BLOCKS_DEFAULT_ENV)
ex.add_config(paths.CFG_BLOCKS_DEFAULT_DECONSTRUCTION_PLANNER)
ex.add_config(paths.CFG_BLOCKS_DEFAULT_DECONSTRUCTION)


@ex.automain
def main(env_config, planner_config, simulator, robot, num_rotations, num_processes, num_samples, save_path):

    env_config = dict(env_config)
    env_config["workspace"] = np.array(env_config["workspace"])

    runner_config = {
        Constants.SIMULATOR: simulator,
        Constants.ROBOT: robot,
        Constants.WORKSPACE: env_config["workspace"],
        Constants.HEIGHTMAP_SIZE: env_config["obs_size"],
        Constants.NUM_OBJECTS: env_config["num_objects"],
        Constants.ACTION_SEQUENCE: env_config["action_sequence"],
        Constants.NUM_ROTATIONS: num_rotations,
        Constants.NUM_PROCESSES: num_processes,
        Constants.NUM_SAMPLES: num_samples,
        Constants.SAVE_PATH: save_path
    }

    logger = Logger(save_file=None, print_logs=True)

    # pass planner config as a dict because dian's code modifies it and sacred does not like that
    runner = RunDeconstruct(runner_config, env_config, dict(planner_config), logger)
    num_attempts, num_episodes = runner.collect()

    ex.log_scalar("num_attempts", num_attempts)
    ex.log_scalar("num_episodes", num_episodes)
