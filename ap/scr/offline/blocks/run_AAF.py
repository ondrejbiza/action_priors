from sacred import Experiment
from sacred.observers import MongoObserver
from .... import constants
from ....constants import Constants
from .... import paths
from ....run.offline.blocks.RunAAF import RunAAF
from ....utils.logger import Logger

ex = Experiment("homo_AAF")
if constants.MONGO_URI is not None and constants.DB_NAME is not None:
    ex.observers.append(MongoObserver(url=constants.MONGO_URI, db_name=constants.DB_NAME))
else:
    print("WARNING: results are not being saved. See 'Setup MongoDB' in README.")
ex.add_config(paths.CFG_BLOCKS_AAF)


@ex.automain
def main(load_path, encoder_learning_rate, encoder_weight_decay, validation_fraction, batch_size,
         plot_dataset_examples, num_training_steps, device, model_save_path, model_load_path,
         plot_results, animate_latent, limit, plot_prediction_examples, ignore_list,
         balance_loss, pos_amb_labels_load_path, use_amb_labels):

    model_config = {
        Constants.BALANCE_LOSS: balance_loss
    }

    runner_config = {
        Constants.LOAD_PATH: load_path,
        Constants.ENCODER_LEARNING_RATE: encoder_learning_rate,
        Constants.ENCODER_WEIGHT_DECAY: encoder_weight_decay,
        Constants.VALIDATION_FRACTION: validation_fraction,
        Constants.BATCH_SIZE: batch_size,
        Constants.PLOT_DATASET_EXAMPLES: plot_dataset_examples,
        Constants.NUM_TRAINING_STEPS: num_training_steps,
        Constants.DEVICE: device,
        Constants.MODEL_SAVE_PATH: model_save_path,
        Constants.MODEL_LOAD_PATH: model_load_path,
        Constants.PLOT_RESULTS: plot_results,
        Constants.INPUT_SIZE: (5, 5, 6), # TODO: is this used anywhere
        Constants.NUM_ACTIONS: 90 * 90,
        Constants.ANIMATE_LATENT: animate_latent,
        Constants.LIMIT: limit,
        Constants.IGNORE_LIST: ignore_list,
        Constants.POS_AMB_LABELS_LOAD_PATH: pos_amb_labels_load_path,
        Constants.USE_AMB_LABELS: use_amb_labels
    }

    logger = Logger(save_file=None, print_logs=True)

    runner = RunAAF(runner_config, model_config, logger)
    runner.train_model()
    runner.evaluate()

    if plot_prediction_examples:
        runner.plot_prediction_examples_()

    ex.log_scalar("final_valid_loss", runner.final_valid_loss)
    ex.log_scalar("final_valid_accuracy", runner.final_valid_accuracy)
