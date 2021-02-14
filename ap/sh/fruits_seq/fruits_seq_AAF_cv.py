import argparse
import json
import os
import subprocess
from ... import paths


def main(args):

    dataset_path = "data/fruits_seq_DQN_dsets/dset_eps_0_5_all_20k.h5"
    model_save_dir = "data/fruits_seq_AAF_cv"

    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)

    with open(paths.TASKS_FRUITS_SEQ, "r") as f:
        tasks = json.load(f)

    print(tasks)

    for i in range(len(tasks)):

        model_save_path = os.path.join(model_save_dir, "model_{:d}.pt".format(i))

        subprocess.call([
            "python", "-m", "ap.scr.offline.fruits_seq.run_AAF", "--name", "fruits_seq_AAF_cv",
            "with", "load_path={:s}".format(dataset_path), "ignore_list=[{:d}]".format(i),
            "model_save_path={:s}".format(model_save_path), "device={:s}".format(args.device)
        ])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("device")

    parsed = parser.parse_args()
    main(parsed)
