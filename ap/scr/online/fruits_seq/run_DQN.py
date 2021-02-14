from sacred import Experiment
from sacred.observers import MongoObserver
from ....run.online.fruits_seq.RunDQN import RunDQN
from ....constants import Constants
from ....utils.logger import Logger
from ....utils import sacred_utils
from .... import constants
from .... import paths

ex = Experiment("fruits_seq_DQN")
if constants.MONGO_URI is not None and constants.DB_NAME is not None:
    ex.observers.append(MongoObserver(url=constants.MONGO_URI, db_name=constants.DB_NAME))
else:
    print("WARNING: results are not being saved. See 'Setup MongoDB' in README.")
ex.add_config(paths.CFG_ONLINE_FRUITS_DQN)


@ex.config
def config():

    num_expert_steps = 0
    num_random_steps = 0
    num_pretraining_steps = 0


@ex.automain
def main(dueling, double_learning, prioritized_replay, learning_rate, weight_decay, discount, goal, batch_size, max_steps,
         max_episodes, exploration_steps, prioritized_replay_max_steps, buffer_size, target_network,
         target_network_sync, num_fruits, save_model_path, load_model_path, demonstrate_dqn, device,
         policy, init_tau, final_tau, num_expert_steps, num_random_steps, num_pretraining_steps,
         encoder_load_path, side_transfer, side_encoder_load_path, freeze_encoder, side_transfer_last):

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
        Constants.WEIGHT_DECAY: weight_decay,
        Constants.BATCH_SIZE: batch_size,
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

    if num_expert_steps > 0 or num_random_steps > 0:
        logger.info("collecting {:d} expert and {:d} random transitions".format(num_expert_steps, num_random_steps))
        runner.generate_demonstrations(num_expert_steps, num_random_steps)

    if num_pretraining_steps > 0:
        assert num_expert_steps > 0 or num_random_steps > 0

        logger.info("pretraining DQN for {:d} steps".format(num_pretraining_steps))
        opt = runner.get_opt()
        for i in range(num_pretraining_steps):
            runner.learn_step_(opt, prioritized_replay_max_steps)
            if i % 1000 == 0:
                logger.info("pretraining step {:d}".format(i))
            if i % target_network_sync == 0:
                runner.target_dqn.sync_weights(runner.dqn)

    if encoder_load_path is not None:
        logger.info("loading encoder")
        runner.load_encoder(encoder_load_path)

    if side_encoder_load_path is not None:
        logger.info("loading side encoder")
        runner.load_side_encoder(side_encoder_load_path)

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
