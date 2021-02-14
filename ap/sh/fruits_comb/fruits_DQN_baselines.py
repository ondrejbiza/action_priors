import argparse
import json
import subprocess
from ... import paths


def main(args):

    with open(paths.TASKS_FRUITS_COMB, "r") as f:
        task_list = json.load(f)

    for _ in range(10):

        for idx, c in enumerate(task_list):

            print("{:s} goal".format(str(c)))

            subprocess.call([
                "python", "-m", "ap.scr.online.fruits.run_DQN", "--name", "fruits_DQN_baselines",
                "with", "device={:s}".format(args.device), "goal={:s}".format(str(c))
            ])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("device")

    parsed = parser.parse_args()
    main(parsed)
