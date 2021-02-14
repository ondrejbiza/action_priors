import argparse
import os
import json
import subprocess
from ...envs.stacking_grammar import count_objects
from ...utils import discovery as discovery_utils
from ... import paths


def run(string, num_objects, save_path, gpu=0):

    return subprocess.Popen(
        ["python", "-m", "ap.scr.online.blocks.run_deconstruct", "with",
         "save_path={:s}".format(save_path), "env_config.goal_string={:s}".format(string),
         "env_config.num_objects={:d}".format(num_objects), "env_config.no_additional_objects=False",
         "env_config.check_roof_upright=False", "device=cuda:{:d}".format(gpu),
         "env_config.gen_blocks=4", "env_config.gen_bricks=2", "env_config.gen_roofs=1",
         "env_config.gen_triangles=1"]
    )


def main(args):

    save_dir = "data/blocks_dec_c"
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

        save_path = os.path.join(save_dir, string + ".h5")
        job = executor.submit(run, string, num_objects, save_path)
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
