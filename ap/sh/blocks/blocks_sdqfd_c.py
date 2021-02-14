import argparse
import os
import json
import subprocess
from ...envs.stacking_grammar import  count_objects
from ...utils import discovery as discovery_utils
from ... import paths


def run(string, num_objects, load_path, save_path, gpu=0):

    return subprocess.Popen(
        ["python", "-m", "ap.scr.online.blocks.run_DQN", "--name", "blocks_sdqfd_c",
         "with", "env_config.goal_string={:s}".format(string), "env_config.num_objects={:d}".format(num_objects),
         "env_config.max_steps=20", "batch_size=32", "discount=0.9", "pretraining_steps=10000", "fixed_eps=True",
         "buffer=EXPERT_BUFFER", "max_episodes=40000", "margin=l", "margin_weight=0.1", "margin_l=0.1",
         "save_model_path={:s}".format(save_path), "device=cuda:{:d}".format(gpu),
         "expert_buffer_load_path={:s}".format(load_path), "num_processes=5", "env_config.check_roof_upright=False"]
    )


def main(args):

    save_dir = "data/blocks_sdqfd_c"
    load_dir = "data/blocks_dec_c"
    executor = discovery_utils.setup_mock_executor(args.gpu_list, args.jobs_per_gpu)

    with open(paths.TASKS_BLOCK_STACKING, "r") as f:
        strings = json.load(f)

    print("{:d} strings".format(len(strings)))

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    jobs = []
    for string in strings:

        num_objects = count_objects(string)

        print("==========")
        print("{:s} goal, {:d} objects".format(string, num_objects))
        print("==========")

        save_path = os.path.join(save_dir, string + ".pt")
        load_path = os.path.join(load_dir, string + ".h5")

        if not args.overwrite and os.path.isfile(save_path):
            print("checkpoint found, skip training")
            continue

        job = executor.submit(run, string, num_objects, load_path, save_path)
        jobs.append(job)

    discovery_utils.check_jobs_done_mock(jobs, executor)
    print("done")
    executor.stop()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--overwrite", default=False, action="store_true")
    parser.add_argument("--gpu-list", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--jobs-per-gpu", type=int, default=2)

    parsed = parser.parse_args()
    main(parsed)
