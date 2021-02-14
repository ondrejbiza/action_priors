import argparse
import json
import subprocess
from ... import paths


def main(args):

    with open(paths.TASKS_FRUITS_SEQ, "r") as f:
        task_list = json.load(f)

    for _ in range(10):

        for model_idx, c in enumerate(task_list):

            print("{:s} goal, {:d} model idx".format(str(c), model_idx))

            subprocess.call([
                "python", "-m", "ap.scr.online.fruits_seq.run_DQNAAF", "--name", "fruits_seq_DQN_AAF",
                "with", "device={:s}".format(args.device), "goal={:s}".format(str(c)),
                "aaf_path=data/fruits_seq_AAF_cv/model_{:d}.pt".format(model_idx)
            ])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("device")

    parsed = parser.parse_args()
    main(parsed)
