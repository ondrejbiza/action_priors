import argparse
import os
import subprocess
from ...utils import discovery as discovery_utils


def run(name, dataset_path, save_path, labels_path, ignore_index, gpu=0):

    cmd = [
        "python", "-m", "ap.scr.offline.blocks.run_AAF", "with", "--name", name,
        "load_path={:s}".format(dataset_path), "device=cuda:{:d}".format(gpu),
        "pos_amb_labels_load_path={:s}".format(labels_path),
        "ignore_list=[{:d}]".format(ignore_index),
        "model_save_path={:s}".format(save_path)
    ]
    return subprocess.Popen(cmd)


def main(args):

    num_runs = 1

    base_dir = "data/blocks_sdqfd_collect_c"
    save_dir = "data/blocks_aaf_c_cv"

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    dataset_path = os.path.join(base_dir, "joint.h5")
    labels_path = os.path.join(base_dir, "labels_0_i1_0_i2_1_t_0_05.h5")

    name = "blocks_aaf_c_cv"
    executor = discovery_utils.setup_mock_executor(args.gpu_list, args.jobs_per_gpu)

    for run_idx in range(num_runs):

        jobs = []
        for task_idx in range(16):
            save_model_path = os.path.join(save_dir, "ignore_{:d}_run_{:d}.pt".format(task_idx, run_idx))
            jobs.append(
                executor.submit(run, name, dataset_path, save_model_path, labels_path, task_idx)
            )

        discovery_utils.check_jobs_done_mock(jobs, executor)

    print("done")
    executor.stop()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu-list", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--jobs-per-gpu", type=int, default=1)

    parsed = parser.parse_args()
    main(parsed)
