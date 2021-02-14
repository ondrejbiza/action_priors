import sys
sys.path.insert(0, "ap")

import numpy as np
import torch
from torch.nn import functional
from sacred import Experiment
from sacred.observers import MongoObserver
from ....run.online.blocks.RunDQN import RunDQN
from .... import constants
from ....constants import Constants
from ....models.homo.AAF import AAF
from ....modules.hand_obs.ResUCatEncoder import ResUCatEncoder
from .... import paths
from ....utils.logger import Logger
from ....utils import sacred_utils
from ....utils.policy.DynaPartitionedEpsGreedyPolicy import DynaPartitionedEpsGreedyPolicy

ex = Experiment("blocks_DQN_AAF")
if constants.MONGO_URI is not None and constants.DB_NAME is not None:
    ex.observers.append(MongoObserver(url=constants.MONGO_URI, db_name=constants.DB_NAME))
else:
    print("WARNING: results are not being saved. See 'Setup MongoDB' in README.")
ex.add_config(paths.CFG_BLOCKS_DEFAULT_ENV)
ex.add_config(paths.CFG_BLOCKS_DEFAULT_DECONSTRUCTION_PLANNER)
ex.add_config(paths.CFG_BLOCKS_DEFAULT_DQN)
ex.add_config(paths.CFG_BLOCKS_AAF_MODEL)


@ex.config
def config():

    prob_threshold = 0.5
    positive_prob = 0.8
    aaf_load_path = None


@ex.capture(prefix="AAF_model_config")
def get_aaf_model_config(balance_loss):

    return {
        Constants.BALANCE_LOSS: balance_loss
    }


def get_aaf_model(model_config, logger, device, model_load_path):

    heightmap_size = 90
    diag_length = float(heightmap_size) * np.sqrt(2)
    diag_length = int(np.ceil(diag_length / 32) * 32)

    patch_size = 24
    patch_channel = 1
    patch_shape = (patch_channel, patch_size, patch_size)

    encoder = ResUCatEncoder(
        1, 2, domain_shape=(1, diag_length, diag_length), patch_shape=patch_shape
    )

    model = AAF(encoder, model_config, logger)
    model.to(device)
    model.load(model_load_path)
    model.eval()

    return model


def setup_aaf_mask(model, prob_threshold):

    def mask(state):

        with torch.no_grad():

            pred = model.forward(state)[0]
            pred = functional.sigmoid(pred)
            pred = (pred >= prob_threshold).int()
            pred = pred.cpu().numpy()

        return pred

    return mask


def setup_aaf_policy(mask_fc, positive_prob):

    probs = np.array([[1.0 - positive_prob, positive_prob], [1.0 - positive_prob, positive_prob]], dtype=np.float32)

    model_config = {
        Constants.EXPLORATION_STEPS: 100,
        Constants.NUM_ACTIONS: 90 * 90
    }

    abs_policy = DynaPartitionedEpsGreedyPolicy(model_config, mask_fc, probs=probs)
    abs_policy.init_explore = 1.0
    abs_policy.final_explore = 1.0
    abs_policy.reset()

    return abs_policy


@ex.automain
def main(env_config, planner_config, simulator, robot, num_rotations, num_processes, num_samples, save_path,
         device, patch_size, learning_rate, discount, buffer_type, alg, margin, margin_l, margin_weight, margin_beta,
         divide_factor, buffer_size, per_alpha, per_beta, exploration_steps, init_eps, final_eps, per_expert_eps, per_eps,
         max_episodes, init_coef, final_coef, batch_size, target_update_freq, fixed_eps, true_random,
         training_offset, training_iters, pretraining_steps, expert_buffer_load_path, save_model_path, load_model_path,
         training, evaluation_episodes, collect_data_num_samples, collect_data_save_path, save_all_qs,
         add_qs_path, full_map_only_qs, add_binary_labels_load_path, add_binary_labels_save_path,
         add_binary_labels_int1_threshold, add_binary_labels_int2_threshold, prob_threshold, positive_prob,
         aaf_load_path, fake_expert):

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
        Constants.SAVE_PATH: save_path,
        Constants.DEVICE: device,
        Constants.PATCH_SIZE: patch_size,
        Constants.LEARNING_RATE: learning_rate,
        Constants.DISCOUNT: discount,
        Constants.BUFFER_TYPE: Constants(buffer_type.upper()),
        Constants.ALG: alg,
        Constants.MARGIN: margin,
        Constants.MARGIN_L: margin_l,
        Constants.MARGIN_WEIGHT: margin_weight,
        Constants.MARGIN_BETA: margin_beta,
        Constants.DIVIDE_FACTOR: divide_factor,
        Constants.BUFFER_SIZE: buffer_size,
        Constants.PER_ALPHA: per_alpha,
        Constants.PER_BETA: per_beta,
        Constants.EXPLORATION_STEPS: exploration_steps,
        Constants.INIT_EPS: init_eps,
        Constants.FINAL_EPS: final_eps,
        Constants.PER_EXPERT_EPS: per_expert_eps,
        Constants.PER_EPS: per_eps,
        Constants.MAX_EPISODES: max_episodes,
        Constants.INIT_COEF: init_coef,
        Constants.FINAL_COEF: final_coef,
        Constants.BATCH_SIZE: batch_size,
        Constants.TARGET_UPDATE_FREQ: target_update_freq,
        Constants.FIXED_EPS: fixed_eps,
        Constants.TRUE_RANDOM: true_random,
        Constants.TRAINING_OFFSET: training_offset,
        Constants.TRAINING_ITERS: training_iters,
        Constants.GET_CUSTOM_LABELS: env_config['get_custom_labels'],
        Constants.FAKE_EXPERT: fake_expert
    }

    logger = Logger(save_file=None, print_logs=True)

    if aaf_load_path is not None:
        aaf_model_config = get_aaf_model_config()
        aaf_model = get_aaf_model(aaf_model_config, logger, device, aaf_load_path)

        mask_fc = setup_aaf_mask(aaf_model, prob_threshold)
        policy = setup_aaf_policy(mask_fc, positive_prob)
    else:
        policy = None

    # pass planner config as a dict because dian's code modifies it and sacred does not like that
    runner = RunDQN(runner_config, env_config, dict(planner_config), logger, exp_policy=policy)

    if load_model_path is not None:
        runner.load_agent(load_model_path)

    if pretraining_steps is not None and pretraining_steps > 0:
        assert expert_buffer_load_path is not None
        runner.load_expert_transitions(expert_buffer_load_path)
        runner.pretraining(pretraining_steps)

    if training:
        result = runner.training()
        logger.info("mean rewards: {:.5f}".format(np.mean(result[Constants.REWARDS])))
        logger.info("last 100 ep. mean rewards: {:.5f}".format(np.mean(result[Constants.REWARDS][-100:])))
        sacred_utils.log_list("rewards", result[Constants.REWARDS], ex)
        sacred_utils.log_list("td_errors", result[Constants.TD_ERROR], ex)
        sacred_utils.log_list("total_loss", result[Constants.TOTAL_LOSS], ex)

    if evaluation_episodes is not None:
        result = runner.evaluation(evaluation_episodes)
        print(result[Constants.REWARDS])
        logger.info("eval mean rewards: {:.5f}".format(np.mean(result[Constants.REWARDS])))
        sacred_utils.log_list("eval_rewards", result[Constants.REWARDS], ex)

    if save_model_path is not None:
        runner.save_agent(save_model_path)

    if collect_data_save_path is not None:
        assert collect_data_num_samples is not None
        runner.collect_data(collect_data_num_samples, collect_data_save_path, save_all_qs)

    if add_qs_path is not None:
        runner.add_qs_to_dataset(add_qs_path, full_map_only_qs=full_map_only_qs)

    if add_binary_labels_load_path is not None:
        assert add_binary_labels_save_path is not None
        # will save ${save_path}_opt.npy and ${save_path}_amb.npy
        runner.generate_positive_and_ambiguous_labels(
            add_binary_labels_load_path, add_binary_labels_save_path, add_binary_labels_int1_threshold,
            add_binary_labels_int2_threshold
        )
