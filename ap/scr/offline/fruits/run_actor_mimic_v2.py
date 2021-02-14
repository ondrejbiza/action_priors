import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver
from .... import constants
from ....constants import Constants
from .... import paths
from ....run.offline.fruits.RunActorMimicV2 import RunActorMimicV2
from ....utils.logger import Logger
from ....utils import sacred_utils

ex = Experiment("fruits_actor_mimic_v2")
if constants.MONGO_URI is not None and constants.DB_NAME is not None:
    ex.observers.append(MongoObserver(url=constants.MONGO_URI, db_name=constants.DB_NAME))
else:
    print("WARNING: results are not being saved. See 'Setup MongoDB' in README.")
ex.add_config(paths.CFG_OFFLINE_FRUITS_ACTOR_MIMIC_V2)


@ex.capture(prefix="student_config")
def get_student_config(num_actions, tau, neurons):

    return {
        Constants.NUM_ACTIONS: num_actions,
        Constants.TAU: tau,
        Constants.NEURONS: neurons
    }


@ex.capture(prefix="teacher_config")
def get_teacher_config(num_actions, dueling, discount, prioritized_replay, prioritized_replay_max_steps,
                       exploration_steps, buffer_size):

    return {
        Constants.NUM_ACTIONS: num_actions,
        Constants.DUELING: dueling,
        Constants.DISCOUNT: discount,
        Constants.PRIORITIZED_REPLAY: prioritized_replay,
        Constants.PRIORITIZED_REPLAY_MAX_STEPS: prioritized_replay_max_steps,
        Constants.EXPLORATION_STEPS: exploration_steps,
        Constants.BUFFER_SIZE: buffer_size
    }


@ex.automain
def main(teacher_load_paths, goals, num_fruits, tau, learning_rate, weight_decay, epsilon, buffer_size,
         batch_size, max_steps, max_episodes, device, student_save_path):

    logger = Logger(save_file=None, print_logs=True)

    runner_config = {
        Constants.TEACHER_LOAD_PATHS: teacher_load_paths,
        # I go from ReadOnlyList to List, not sure if it matters
        Constants.GOALS: [list(g) for g in goals],
        Constants.NUM_FRUITS: num_fruits,
        Constants.TAU: tau,
        Constants.LEARNING_RATE: learning_rate,
        Constants.WEIGHT_DECAY: weight_decay,
        Constants.EPSILON: epsilon,
        Constants.BUFFER_SIZE: buffer_size,
        Constants.BATCH_SIZE: batch_size,
        Constants.MAX_STEPS: max_steps,
        Constants.MAX_EPISODES: max_episodes,
        Constants.DEVICE: device
    }

    student_config = get_student_config()
    student_config[Constants.NUM_HEADS] = len(goals)

    teacher_config = get_teacher_config()

    runner = RunActorMimicV2(runner_config, student_config, teacher_config, logger)
    runner.train_model()

    if student_save_path is not None:
        runner.save_student_encoder(student_save_path)

    sacred_utils.log_list("total_rewards", runner.training_result[Constants.TOTAL_REWARDS], ex)
    sacred_utils.log_list("discounted_total_rewards", runner.training_result[Constants.DISCOUNTED_REWARDS], ex)

    # need to average over all tasks
    sacred_utils.log_list(
        "eval_total_rewards", np.mean(runner.training_result[Constants.EVAL_TOTAL_REWARDS], axis=1), ex
    )
    sacred_utils.log_list(
        "eval_discounted_total_rewards",
        np.mean(runner.training_result[Constants.EVAL_DISCOUNTED_TOTAL_REWARDS], axis=1), ex
    )
    sacred_utils.log_list("eval_num_steps", np.mean(runner.training_result[Constants.EVAL_NUM_STEPS], axis=1), ex)
