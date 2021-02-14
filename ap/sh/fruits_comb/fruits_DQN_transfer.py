import argparse
import json
import subprocess
from ... import paths


def main(args):

    with open(paths.TASKS_FRUITS_COMB, "r") as f:
        task_list = json.load(f)

    for _ in range(10):

        for idx, c in enumerate(task_list):

            encoder_load_path = "data/fruits_DQN_students/model_ignore_{:d}.pt".format(idx)

            print("{:d} fruits, {:s} goal".format(idx, str(c)))

            subprocess.call([
                "python", "-m", "ap.scr.online.fruits.run_DQN", "--name", "fruits_DQN_transfer",
                "with", "device={:s}".format(args.device), "goal={:s}".format(str(c)),
                "encoder_load_path={:s}".format(encoder_load_path)
            ])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("device")

    parsed = parser.parse_args()
    main(parsed)
