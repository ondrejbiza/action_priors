import argparse
import json
import os
import subprocess
from ... import paths


def main(args):

    folder_path = "data/fruits_seq_DQN_models"

    with open(paths.TASKS_FRUITS_SEQ, "r") as f:
        tasks = json.load(f)

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    for c in tasks:

        model_name = "model_{:s}.pt".format(str(c))
        model_path = os.path.join(folder_path, model_name)

        print("{:d} fruits, {:s} goal".format(len(c), str(c)))

        subprocess.call([
            "python", "-m", "ap.scr.online.fruits_seq.run_DQN", "--name", "fruits_seq_DQN_models",
            "with", "device={:s}".format(args.device), "goal={:s}".format(str(c)),
            "max_steps=50000", "exploration_steps=1", "prioritized_replay=False",
            "prioritized_replay_max_steps=0", "save_model_path={:s}".format(model_path),
            "num_expert_steps=50000", "num_random_steps=0", "num_pretraining_steps=50000"
        ])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("device")

    parsed = parser.parse_args()
    main(parsed)
