import argparse
import json
import subprocess
from ... import paths


def main(args):

    with open(paths.TASKS_FRUITS_COMB, "r") as f:
        task_list = json.load(f)

    for _ in range(10):

        for idx, c in enumerate(task_list):

            print("{:s} goal, {:d} model idx".format(str(c), idx))

            subprocess.call([
                "python", "-m", "ap.scr.online.fruits.run_DQNAAF", "--name", "fruits_DQN_AAF",
                "with", "device={:s}".format(args.device), "goal={:s}".format(str(c)),
                "aaf_path=data/fruits_AAF_cv/model_{:d}.pt".format(idx)
            ])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("device")

    parsed = parser.parse_args()
    main(parsed)
