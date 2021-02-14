from datetime import datetime
import numpy as np
from sacred import Experiment
from ....run.online.fruits_seq.RunDQN import RunDQN
from ....constants import Constants
from ....utils.logger import Logger
from ....utils.dataset import ArrayDataset
from .... import paths

ex = Experiment("fruits_seq_DQN_join_dset")
ex.add_config(paths.CFG_ONLINE_FRUITS_DQN)


@ex.config
def config():

    datasets_list = None
    models_list = None
    exp_per_dataset = None
    dset_save_path = None
    c_string = None
    device = "cuda:1"


@ex.automain
def main(dueling, double_learning, prioritized_replay, learning_rate, weight_decay, discount, goal, batch_size,
         max_steps, max_episodes, exploration_steps, prioritized_replay_max_steps, buffer_size, target_network,
         target_network_sync, num_fruits, dset_save_path, device, datasets_list, models_list, exp_per_dataset,
         c_string):

    new_dset = None
    print(c_string)
    for i, dset_path in enumerate(datasets_list):

        dset = ArrayDataset(None)
        dset.load_hdf5(dset_path)
        dset[Constants.TASK_INDEX] = np.zeros(len(dset[Constants.STATES]), dtype=np.int32) + i

        if exp_per_dataset is not None:
            dset.shuffle()
            dset.limit(exp_per_dataset)

        if new_dset is None:
            new_dset = dset
        else:
            new_dset.concatenate_dset(dset)

    new_dset.metadata = None
    new_dset[Constants.QS] = np.zeros((len(new_dset[Constants.QS]), len(models_list), 25), dtype=np.float32)
    new_dset[Constants.ABSTRACT_ACTIONS][:] = 0

    model_config = {
        Constants.NUM_ACTIONS: 25,
        Constants.DUELING: dueling,
        Constants.PRIORITIZED_REPLAY: prioritized_replay,
        Constants.DISCOUNT: discount,
        Constants.EXPLORATION_STEPS: exploration_steps,
        Constants.PRIORITIZED_REPLAY_MAX_STEPS: prioritized_replay_max_steps,
        Constants.BUFFER_SIZE: buffer_size,
        Constants.INIT_TAU: None,
        Constants.FINAL_TAU: None
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
        Constants.POLICY: Constants.EPS,
        Constants.SIDE_TRANSFER: False,
        Constants.FREEZE_ENCODER: False,
        Constants.SIDE_TRANSFER_LAST: False
    }

    logger = Logger(save_file=None, print_logs=True)

    for i, model_path in enumerate(models_list):

        runner = RunDQN(runner_config, model_config, logger)
        runner.dqn.load(model_path)
        qs = runner.label_data(new_dset[Constants.STATES], new_dset[Constants.HAND_STATES])

        new_dset[Constants.QS][:, i, :] = qs

    new_dset.metadata = {
        Constants.NUM_EXP: new_dset.size, Constants.TIMESTAMP: str(datetime.today()),
        Constants.TASK_LIST: c_string
    }
    new_dset.save_hdf5(dset_save_path)
