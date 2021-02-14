import sys
sys.path.insert(0, "ap")

import argparse
from datetime import datetime
import numpy as np
from ....run.online.blocks.RunTaskClassifier import RunTaskClassifier
from ....utils.logger import Logger
from ....utils.dataset import ArrayDataset
from .... import constants
from ....constants import Constants


def get_task_classifier_runner_(num_tasks, logger, classifier_load_path, device):

    # we won't load the dataset or setup the training of the model
    config = {
        Constants.DATASET_LOAD_PATH: None,
        Constants.VALIDATION_FRACTION: None,
        Constants.VALIDATION_FREQ: None,
        Constants.BATCH_SIZE: 100,
        Constants.DEVICE: device,
        Constants.LEARNING_RATE: 0.0,
        Constants.WEIGHT_DECAY: 0.0,
        Constants.NUM_TRAINING_STEPS: 0,
        Constants.PLOT_RESULTS: False,
        Constants.NUM_TASKS: num_tasks
    }
    # build the model and load its weights
    run_task_classifier = RunTaskClassifier(config, logger, build=False)
    run_task_classifier.build_model_()
    run_task_classifier.load_model(classifier_load_path)
    return run_task_classifier


def classify_tasks_(classifier_threshold, run_task_classifier, dataset, logger):

    # get a probability distribution over tasks for each sample
    # then deallocate the task classifier
    run_task_classifier.model.eval()
    probs = run_task_classifier.get_predictions(dataset)
    del run_task_classifier
    # if a sample has some minimum probability of belonging in a particular task
    # the Q-value prediction from a DQN for that task will be allowed for this sample
    allowed_tasks = probs >= classifier_threshold

    logger.info(
        "percentages of allowed tasks: " +
        str(100.0 * np.sum(allowed_tasks.astype(np.float32), axis=0) / allowed_tasks.shape[0])
    )

    return allowed_tasks


def run(num_tasks, logger, classifier_load_path, classifier_threshold, dataset, load_paths, save_path, ignore_list,
        device):

    run_task_classifier = get_task_classifier_runner_(num_tasks, logger, classifier_load_path, device)
    allowed_tasks = classify_tasks_(classifier_threshold, run_task_classifier, dataset, logger)

    positive_labels = np.zeros((dataset.size, 90 * 90), dtype=np.bool)
    ambiguous_labels = np.zeros((dataset.size, 90 * 90), dtype=np.bool)

    for idx, load_path in enumerate(load_paths):

        if idx in ignore_list:
            continue

        opt_load_path = load_path + constants.OPT_SUFFIX
        amb_load_path = load_path + constants.AMB_SUFFIX

        opt = np.load(opt_load_path)
        amb = np.load(amb_load_path)

        assert positive_labels.shape == opt.shape and positive_labels.dtype == opt.dtype
        assert ambiguous_labels.shape == amb.shape and ambiguous_labels.dtype == amb.dtype

        opt = np.bitwise_and(opt, allowed_tasks[:, idx][:, np.newaxis])
        amb = np.bitwise_and(amb, allowed_tasks[:, idx][:, np.newaxis])

        positive_labels = np.bitwise_or(positive_labels, opt)
        ambiguous_labels = np.bitwise_or(ambiguous_labels, amb)

    labels = ArrayDataset({
        Constants.POSITIVE_LABELS: positive_labels,
        Constants.AMBIGUOUS_LABELS: ambiguous_labels
    })
    labels.metadata = {
        Constants.NUM_EXP: dataset.size, Constants.TIMESTAMP: str(datetime.today())
    }
    labels.save_hdf5(save_path)


def main(args):

    logger = Logger(save_file=None, print_logs=True)
    dataset = ArrayDataset(None)
    dataset.load_hdf5(args.dataset_load_path)

    run(
        args.num_tasks, logger, args.classifier_load_path, args.classifier_threshold, dataset,
        args.label_load_paths, args.save_path, args.ignore_list, args.device
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("num_tasks", type=int)
    parser.add_argument("classifier_load_path")
    parser.add_argument("classifier_threshold", type=float)
    parser.add_argument("dataset_load_path")
    parser.add_argument("save_path")
    parser.add_argument("--ignore-list", nargs="+", default=[])
    parser.add_argument("--label-load-paths", nargs="+")
    parser.add_argument("--device", default="cuda:0")

    parsed = parser.parse_args()
    main(parsed)
