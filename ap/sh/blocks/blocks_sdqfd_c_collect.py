import argparse
import os
import json
import subprocess
from ...envs.stacking_grammar import count_objects
from ...utils import discovery as discovery_utils
from ... import paths


def run_collect(string, num_objects, load_path, save_path, gpu=0):

    return subprocess.Popen(
        ["python", "-m", "ap.scr.online.blocks.run_DQN",
         "with", "env_config.goal_string={:s}".format(string),
         "env_config.num_objects={:d}".format(num_objects),
         "env_config.max_steps=20", "env_config.gen_blocks=4", "env_config.gen_bricks=2", "env_config.gen_roofs=1",
         "env_config.gen_triangles=1", "batch_size=32", "discount=0.9",
         "buffer=EXPERT_BUFFER", "max_episodes=20000", "margin=l", "margin_weight=0.1", "margin_l=0.1",
         "load_model_path={:s}".format(load_path), "device=cuda:{:d}".format(gpu),
         "num_processes=10", "collect_data_save_path={:s}".format(save_path),
         "collect_data_num_samples=20000", "training=False", "evaluation_episodes=None", "save_all_qs=True",
         "init_eps=0.0", "final_eps=0.0", "init_coef=0.01", "final_coef=0.01", "exploration_steps=1",
         "true_random=True", "fixed_eps=False", "env_config.check_roof_upright=False"]
    )


def run_join(base_path, strings, gpu=0):

    cmd = [
        "python", "-m", "ap.scr.online.blocks.join_datasets", "-s", "{:s}/joint.h5".format(base_path),
        "-l", "20000", "--no-fake-exp", "-d"
    ]

    for string in strings:
        cmd.append("{:s}/{:s}.h5".format(base_path, string))

    cmd.append("-t")

    for string in strings:
        cmd.append(string)

    return subprocess.Popen(cmd)


def run_task_classifier(dataset, save_path, gpu=0):

    return subprocess.Popen([
        "python", "-m", "ap.scr.online.blocks.run_task_classifier", "with",
        "device=cuda:{:d}".format(gpu), "num_training_steps=5001", "learning_rate=0.001",
        "dataset_load_path={:s}".format(dataset), "save_model_path={:s}".format(save_path),
        "plot_results=False", "plot_prediction_examples=False", "num_tasks=16"
    ])


def clean_up_intermediate_datasets(save_dir):

    subprocess.call("rm {:s}/1b*.h5".format(save_dir), shell=True)
    subprocess.call("rm {:s}/1l*.h5".format(save_dir), shell=True)
    subprocess.call("rm {:s}/2b*.h5".format(save_dir), shell=True)


def main(args):

    executor = discovery_utils.setup_mock_executor(args.gpu_list, args.jobs_per_gpu)

    with open(paths.TASKS_BLOCK_STACKING, "r") as f:
        strings = json.load(f)

    load_dir = "data/blocks_sdqfd_c"
    save_dir = "data/blocks_sdqfd_collect_c"

    print("{:d} strings".format(len(strings)))

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # collect data
    jobs = []
    for string in strings:

        num_objects = count_objects(string)

        print("==========")
        print("{:s} goal, {:d} objects".format(string, num_objects))
        print("==========")

        save_path = os.path.join(save_dir, string + ".h5")
        load_path = os.path.join(load_dir, string + ".pt")

        if not os.path.isfile(save_path):
            jobs.append(executor.submit(run_collect, string, num_objects, load_path, save_path))

    # check collection done
    discovery_utils.check_jobs_done_mock(jobs, executor)

    # join datasets
    job = executor.submit(run_join, save_dir, strings)
    discovery_utils.check_jobs_done_mock([job], executor)

    # clean up
    clean_up_intermediate_datasets(save_dir)

    # train classifier
    save_path = os.path.join(save_dir, "classifier.pt")
    dataset_path = os.path.join(save_dir, "joint.h5")
    job = executor.submit(run_task_classifier, dataset_path, save_path)
    discovery_utils.check_jobs_done_mock([job], executor)

    print("done")
    executor.stop()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu-list", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--jobs-per-gpu", type=int, default=2)

    parsed = parser.parse_args()
    main(parsed)
