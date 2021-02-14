import argparse
import os
import json
import subprocess
from ...utils import discovery as discovery_utils
from ... import paths


def run_join_labels(classifier_load_path, classifier_threshold, dataset_load_path, label_load_paths, save_path,
                    ignore_list, gpu=0):

    return subprocess.Popen(
        ["python", "-m", "ap.scr.online.blocks.join_labels", "16", classifier_load_path, str(classifier_threshold),
         dataset_load_path, save_path, "--label-load-paths", *label_load_paths, "--ignore-list", *ignore_list]
    )


def clean_up_intermediate_labels(data_dir):

    subprocess.call("rm {:s}/*_amb.npy".format(data_dir), shell=True)
    subprocess.call("rm {:s}/*_opt.npy".format(data_dir), shell=True)


def main(args):

    executor = discovery_utils.setup_mock_executor(args.gpu_list, args.jobs_per_gpu)
    data_dir = "data/blocks_sdqfd_collect_c"

    with open(paths.TASKS_BLOCK_STACKING, "r") as f:
        strings = json.load(f)

    print("{:d} strings".format(len(strings)))

    classifier_load_path = os.path.join(data_dir, "classifier.pt")
    classifier_threshold = 0.05
    dataset_load_path = os.path.join(data_dir, "joint.h5")
    label_load_paths = [os.path.join(data_dir, s + "_i1_0_i2_1") for s in strings]

    jobs = []
    for i in range(len(strings)):

        save_path = os.path.join(data_dir, "labels_{:d}_i1_0_i2_1_t_0_05.h5".format(i))

        jobs.append(executor.submit(
            run_join_labels, classifier_load_path, classifier_threshold, dataset_load_path, label_load_paths, save_path,
            [str(i)]
        ))

    discovery_utils.check_jobs_done_mock(jobs, executor)

    # clean up
    # clean_up_intermediate_labels(data_dir)

    print("done")
    executor.stop()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu-list", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--jobs-per-gpu", type=int, default=1)

    parsed = parser.parse_args()
    main(parsed)
