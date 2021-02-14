import argparse
import os
import json
import subprocess
from ...envs.stacking_grammar import count_objects
from ...utils import discovery as discovery_utils
from ... import paths


def run(name, string, num_objects, aaf_load_path, prob_threshold, positive_prob, gpu=0):

    return subprocess.Popen(
        ["python", "-m", "ap.scr.online.blocks.run_DQN_AAF", "--name", name,
         "with", "env_config.goal_string={:s}".format(string), "env_config.num_objects={:d}".format(num_objects),
         "env_config.no_additional_objects=False", "env_config.check_roof_upright=True",
         "env_config.max_steps=30", "env_config.gen_blocks=4", "env_config.gen_bricks=2", "env_config.gen_roofs=1",
         "env_config.gen_triangles=1", "batch_size=32", "discount=0.9", "pretraining_steps=0", "fixed_eps=False",
         "buffer_type=PRIORITIZED_BUFFER", "max_episodes=100000", "alg=dqn", "init_eps=1.0", "final_eps=0.01",
         "exploration_steps=80000", "learning_rate=0.0001", "num_processes=5", "buffer_size=200000",
         "aaf_load_path={:s}".format(aaf_load_path), "prob_threshold={:f}".format(prob_threshold),
         "positive_prob={:f}".format(positive_prob), "device=cuda:{:d}".format(gpu)]
    )


def main(args):

    aaf_load_dir = "data/blocks_aaf_c_cv"
    aaf_prob_threshold = 0.1
    aaf_true_chance = 1.0
    name = "blocks_DQN_AAF"
    executor = discovery_utils.setup_mock_executor(args.gpu_list, args.jobs_per_gpu)

    with open(paths.TASKS_BLOCK_STACKING, "r") as f:
        strings = json.load(f)

    print("{:d} strings".format(len(strings)))

    jobs = []
    for idx, string in enumerate(strings):

        num_objects = count_objects(string)

        print("==========")
        print("{:s} goal, {:d} objects".format(string, num_objects))
        print("==========")

        tmp_aaf_load_path = os.path.join(aaf_load_dir, "ignore_{:d}_run_0.pt".format(idx))

        job = executor.submit(run, name, string, num_objects, tmp_aaf_load_path, aaf_prob_threshold, aaf_true_chance)
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
