import argparse
import json
import os
import subprocess
from ... import paths


def main(args):

    with open(paths.TASKS_FRUITS_COMB, "r") as f:
        task_list = json.load(f)

    models_path = "data/fruits_DQN_models"
    dataset_path = "data/fruits_DQN_dsets"

    if not os.path.isdir(dataset_path):

        os.makedirs(dataset_path)

    for idx, c in enumerate(task_list):

        model_name = "model_{:s}.pt".format(str(c))
        model_path = os.path.join(models_path, model_name)

        dataset_name = "dset_eps_0_5_{:s}.h5".format(str(c))
        tmp_dataset_path = os.path.join(dataset_path, dataset_name)

        print("{:s} goal".format(str(c)))

        subprocess.call([
            "python", "-m", "ap.scr.online.fruits.run_DQN_dset",
            "with", "device={:s}".format(args.device), "goal={:s}".format(str(c)),
            "load_model_path={:s}".format(model_path), "dset_save_path={:s}".format(tmp_dataset_path),
            "eps=0.5", "num_exp=100000"
        ])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("device")

    parsed = parser.parse_args()
    main(parsed)
