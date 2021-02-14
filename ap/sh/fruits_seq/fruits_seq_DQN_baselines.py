import argparse
import json
import subprocess
from ... import paths


def main(args):

    with open(paths.TASKS_FRUITS_SEQ, "r") as f:
        tasks = json.load(f)

    for _ in range(10):

        for c in tasks:

            print("{:d} fruits, {:s} goal".format(len(c), str(c)))

            subprocess.call([
                "python", "-m", "ap.scr.online.fruits_seq.run_DQN", "--name", "fruits_seq_DQN_baselines",
                "with", "device={:s}".format(args.device), "goal={:s}".format(str(c))
            ])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("device")

    parsed = parser.parse_args()
    main(parsed)
