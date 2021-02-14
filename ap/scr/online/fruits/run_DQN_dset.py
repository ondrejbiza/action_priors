from sacred import Experiment
from ....run.online.fruits.RunDQN import RunDQN
from ....constants import Constants
from ....utils.logger import Logger
from .... import paths

ex = Experiment("fruits_DQN_collect_dset")
ex.add_config(paths.CFG_ONLINE_FRUITS_DQN)


@ex.config
def config():
    # additional config entries
    num_exp = 1000
    eps = 0.1
    dset_save_path = None


@ex.automain
def main(dueling, double_learning, prioritized_replay, learning_rate, weight_decay, discount, goal, batch_size, max_steps,
         max_episodes, exploration_steps, prioritized_replay_max_steps, buffer_size, target_network,
         target_network_sync, num_fruits, load_model_path, num_exp, eps, dset_save_path, device,
         policy, init_tau, final_tau, side_transfer, freeze_encoder, side_transfer_last):

    model_config = {
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

    runner_config = {
        Constants.DOUBLE_LEARNING: double_learning,
        Constants.LEARNING_RATE: learning_rate,
        Constants.BATCH_SIZE: batch_size,
        Constants.WEIGHT_DECAY: weight_decay,
        Constants.GOAL: goal,
        Constants.DEVICE: device,
        Constants.MAX_STEPS: max_steps,
        Constants.MAX_EPISODES: max_episodes,
        Constants.TARGET_NETWORK: target_network,
        Constants.TARGET_NETWORK_SYNC: target_network_sync,
        Constants.NUM_FRUITS: num_fruits,
        Constants.POLICY: Constants(policy),
        Constants.SIDE_TRANSFER: side_transfer,
        Constants.FREEZE_ENCODER: freeze_encoder,
        Constants.SIDE_TRANSFER_LAST: side_transfer_last
    }

    logger = Logger(save_file=None, print_logs=True)

    runner = RunDQN(runner_config, model_config, logger)
    runner.dqn.load(load_model_path)
    runner.collect_data(num_exp, eps, dset_save_path)
