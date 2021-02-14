import sys
sys.path.insert(0, "ap")

from sacred.observers import MongoObserver
from sacred import Experiment
from ....constants import Constants
from .... import constants
from ....run.online.blocks.RunTaskClassifier import RunTaskClassifier
from ....utils.logger import Logger
from .... import paths

ex = Experiment("blocks_task_classifier")
if constants.MONGO_URI is not None and constants.DB_NAME is not None:
    ex.observers.append(MongoObserver(url=constants.MONGO_URI, db_name=constants.DB_NAME))
else:
    print("WARNING: results are not being saved. See 'Setup MongoDB' in README.")
ex.add_config(paths.CFG_BLOCKS_DEFAULT_TASK_CLASSIFIER)


@ex.automain
def main(dataset_load_path, validation_fraction, batch_size, device, learning_rate, weight_decay,
         num_training_steps, plot_results, plot_dataset_examples, plot_prediction_examples,
         validation_freq, save_model_path, load_model_path, num_tasks):

    runner_config = {
        Constants.DATASET_LOAD_PATH: dataset_load_path,
        Constants.VALIDATION_FRACTION: validation_fraction,
        Constants.VALIDATION_FREQ: validation_freq,
        Constants.BATCH_SIZE: batch_size,
        Constants.DEVICE: device,
        Constants.LEARNING_RATE: learning_rate,
        Constants.WEIGHT_DECAY: weight_decay,
        Constants.NUM_TRAINING_STEPS: num_training_steps,
        Constants.PLOT_RESULTS: plot_results,
        Constants.NUM_TASKS: num_tasks
    }

    logger = Logger(save_file=None, print_logs=True)

    runner = RunTaskClassifier(runner_config, logger)

    if plot_dataset_examples:
        runner.plot_dataset_examples()

    if load_model_path is not None:
        runner.load_model(load_model_path)

    runner.train()

    if save_model_path is not None:
        runner.save_model(save_model_path)

    if plot_prediction_examples:
        runner.plot_prediction_examples()
