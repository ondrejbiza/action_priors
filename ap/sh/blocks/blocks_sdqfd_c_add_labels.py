import argparse
import os
import json
import subprocess
from ...envs.stacking_grammar import count_objects
from ...utils import discovery as discovery_utils
from ... import paths


def run_add_labels(string, num_objects, model_path, dataset_path, save_path, gpu=0):

    return subprocess.Popen(
        ["python", "-m", "ap.scr.online.blocks.run_DQN",
         "with", "env_config.goal_string={:s}".format(string),
         "env_config.num_objects={:d}".format(num_objects),
         "env_config.max_steps=20", "batch_size=32", "discount=0.9", "fixed_eps=True",
         "buffer=EXPERT_BUFFER", "max_episodes=20000", "margin=l", "margin_weight=0.1", "margin_l=0.1",
         "load_model_path={:s}".format(model_path), "device=cuda:{:d}".format(gpu),
         "num_processes=10", "training=False", "evaluation_episodes=None",
         "add_binary_labels_load_path={:s}".format(dataset_path),
         "add_binary_labels_save_path={:s}".format(save_path),
         "add_binary_labels_int1_threshold=0.0", "add_binary_labels_int2_threshold=1.0"]
    )


def main(args):

    executor = discovery_utils.setup_mock_executor(args.gpu_list, args.jobs_per_gpu)

    with open(paths.TASKS_BLOCK_STACKING, "r") as f:
        strings = json.load(f)

    load_dir = "data/blocks_sdqfd_c"
    save_dir = "data/blocks_sdqfd_collect_c"

    print("{:d} strings".format(len(strings)))

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # add qs to datasets
    dataset_path = os.path.join(save_dir, "joint.h5")

    jobs = []
    for idx, string in enumerate(strings):

        num_objects = count_objects(string)

        print("==========")
        print("{:s} goal, {:d} objects".format(string, num_objects))
        print("==========")

        model_path = os.path.join(load_dir, string + ".pt")
        save_path = os.path.join(save_dir, string + "_i1_0_i2_1")

        job = executor.submit(run_add_labels, string, num_objects, model_path, dataset_path, save_path)
        jobs.append(job)

    discovery_utils.check_jobs_done_mock(jobs, executor)
    print("done")
    executor.stop()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu-list", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--jobs-per-gpu", type=int, default=2)

    parsed = parser.parse_args()
    main(parsed)
