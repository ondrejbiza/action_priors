from sacred import Experiment
from sacred.observers import MongoObserver
import numpy as np
from torch import nn
from ....run.online.fruits_seq.RunDQNAAF import RunDQNAAF
from ....constants import Constants
from .... import constants
from ....utils.logger import Logger
from ....utils import sacred_utils
from .... import paths
from ....modules.FlatStatesMultipleConcatOneHotActions import FlatStatesMultipleConcatOneHotActions
from ....modules.FCEncoder import FCEncoder
from ....models.homo.AAFFruits import AAFFruits


ex = Experiment("fruits_seq_DQNAAF")
if constants.MONGO_URI is not None and constants.DB_NAME is not None:
    ex.observers.append(MongoObserver(url=constants.MONGO_URI, db_name=constants.DB_NAME))
else:
    print("WARNING: results are not being saved. See 'Setup MongoDB' in README.")
ex.add_config(paths.CFG_ONLINE_FRUITS_DQNAAF)


@ex.capture(prefix="dqn_config")
def get_dqn_config(dueling, prioritized_replay, discount, exploration_steps, prioritized_replay_max_steps, buffer_size,
                   init_tau, final_tau):

    return {
        Constants.NUM_ACTIONS: 25,
        Constants.DUELING: dueling,
        Constants.PRIORITIZED_REPLAY: prioritized_replay,
        Constants.DISCOUNT: discount,
        Constants.EXPLORATION_STEPS: exploration_steps,
        Constants.PRIORITIZED_REPLAY_MAX_STEPS: prioritized_replay_max_steps,
        Constants.BUFFER_SIZE: buffer_size,
        Constants.INIT_TAU: init_tau,
        Constants.FINAL_TAU: final_tau
    }


def build_aaf_model(model_config, logger, device):

    input_size = ((5, 5, 5), (5, 5))
    num_actions = 25

    reshape_encoder = FlatStatesMultipleConcatOneHotActions(num_actions)

    fc_config = {
        Constants.INPUT_SIZE: int(np.prod(input_size[0])) + int(np.prod(input_size[1])) + num_actions,
        Constants.NEURONS: [256, 256, 32],
        Constants.USE_BATCH_NORM: False,
        Constants.USE_LAYER_NORM: False,
        Constants.ACTIVATION_LAST: True
    }

    fc_encoder = FCEncoder(fc_config, logger)

    encoder = nn.Sequential(reshape_encoder, fc_encoder)
    encoder.output_size = fc_encoder.output_size

    model = AAFFruits(encoder, model_config, logger)
    return model.to(device)


@ex.automain
def main(double_learning, learning_rate, weight_decay, batch_size, goal, max_steps, max_episodes,
         target_network, target_network_sync, num_fruits, aaf_path, load_model_path, save_model_path,
         demonstrate_dqn, device, policy):

    logger = Logger(save_file=None, print_logs=True)

    dqn_model_config = get_dqn_config()
    model = build_aaf_model({}, logger, device)
    model.load(aaf_path)

    runner_config = {
        Constants.DOUBLE_LEARNING: double_learning,
        Constants.LEARNING_RATE: learning_rate,
        Constants.WEIGHT_DECAY: weight_decay,
        Constants.BATCH_SIZE: batch_size,
        Constants.GOAL: goal,
        Constants.DEVICE: device,
        Constants.MAX_STEPS: max_steps,
        Constants.MAX_EPISODES: max_episodes,
        Constants.TARGET_NETWORK: target_network,
        Constants.TARGET_NETWORK_SYNC: target_network_sync,
        Constants.NUM_FRUITS: num_fruits,
        Constants.NUM_ACTIONS: 25,
        Constants.POLICY: Constants(policy),
        Constants.SIDE_TRANSFER: False,
        Constants.FREEZE_ENCODER: False,
        Constants.SIDE_TRANSFER_LAST: False
    }

    runner = RunDQNAAF(runner_config, dqn_model_config, model, logger)

    if load_model_path is not None:
        runner.dqn.load(load_model_path)
    else:
        runner.train_model()

    if save_model_path is not None:
        runner.dqn.save(save_model_path)

    if demonstrate_dqn:
        runner.demonstrate_dqn()

    sacred_utils.log_list("total_rewards", runner.training_result[Constants.TOTAL_REWARDS], ex)
    sacred_utils.log_list("discounted_total_rewards", runner.training_result[Constants.DISCOUNTED_REWARDS], ex)

    sacred_utils.log_list("eval_total_rewards", runner.training_result[Constants.EVAL_TOTAL_REWARDS], ex)
    sacred_utils.log_list(
        "eval_discounted_total_rewards", runner.training_result[Constants.EVAL_DISCOUNTED_TOTAL_REWARDS], ex
    )
    sacred_utils.log_list("eval_num_steps", runner.training_result[Constants.EVAL_NUM_STEPS], ex)
