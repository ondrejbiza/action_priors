import argparse
import json
import subprocess
from ... import paths


def main(args):

    with open(paths.TASKS_FRUITS_SEQ, "r") as f:
        tasks = json.load(f)

    for _ in range(10):

        for idx, c in enumerate(tasks):

            encoder_load_path = "data/fruits_seq_DQN_students/model_ignore_{:d}.pt".format(idx)

            print("{:d} fruits, {:s} goal".format(len(c), str(c)))

            subprocess.call([
                "python", "-m", "ap.scr.online.fruits_seq.run_DQN", "--name", "fruits_seq_DQN_side_transfer",
                "with", "device={:s}".format(args.device), "goal={:s}".format(str(c)),
                "side_transfer=True", "side_encoder_load_path={:s}".format(encoder_load_path)
            ])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("device")

    parsed = parser.parse_args()
    main(parsed)
