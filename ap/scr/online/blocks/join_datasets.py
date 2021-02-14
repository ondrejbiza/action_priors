import sys
sys.path.insert(0, "ap")


import argparse
from datetime import datetime
import numpy as np
from ....constants import Constants
from ....utils.dataset import ArrayDataset


def get_args(args):

    datasets, tasks, limit, epsilon, save_path, no_fake_exp = \
        args.datasets, args.tasks, args.limit, args.epsilon, args.save_path, args.no_fake_exp
    assert datasets is not None
    assert tasks is not None
    assert len(datasets) == len(tasks)
    assert limit is not None
    assert epsilon is not None or no_fake_exp
    assert save_path is not None

    return datasets, tasks, limit, epsilon, save_path, no_fake_exp


def main(args):

    datasets, tasks, limit, epsilon, save_path, no_fake_exp = get_args(args)

    joint = None

    for idx in range(len(datasets)):

        dset_path = datasets[idx]

        dset = ArrayDataset(None)
        dset.load_hdf5(dset_path)
        dset.shuffle()
        dset.limit(args.limit)
        dset[Constants.TASK_INDEX] = np.zeros(len(dset[Constants.OBS]), dtype=np.int32) + idx
        dset[Constants.ORIG_ACTIONS] = dset[Constants.ACTIONS][:, 0] * 90 + dset[Constants.ACTIONS][:, 1]

        if not no_fake_exp:
            if Constants.QS in dset:
                print("Fake exploration.")
                shape = dset[Constants.QS].shape
                greedy_actions = np.argmax(dset[Constants.QS].reshape((shape[0], shape[1] * shape[2])), axis=1)
                exp_mask = np.random.choice([True, False], size=len(greedy_actions), p=[epsilon, 1 - epsilon])

                greedy_actions[exp_mask] = np.random.randint(0, 90 * 90, size=np.sum(exp_mask))
                dset[Constants.ACTIONS] = greedy_actions

                del dset.arrays[Constants.QS]
            else:
                print("No Q-values in dataset. Assuming test mode, eps-greedy actions not created.")
        else:
            print("Deleting Q-values and flattening actions.")
            del dset.arrays[Constants.QS]
            dset[Constants.ACTIONS] = dset[Constants.ACTIONS][:, 0] * 90 + dset[Constants.ACTIONS][:, 1]

        if joint is None:
            joint = dset
        else:
            joint.concatenate_dset(dset)

    c_string = ",".join(tasks)
    joint.metadata = {
        Constants.NUM_EXP: joint.size, Constants.TIMESTAMP: str(datetime.today()),
        Constants.TASK_LIST: c_string
    }
    joint.save_hdf5(save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--datasets", nargs="+", default=None)
    parser.add_argument("-t", "--tasks", nargs="+", default=None)
    parser.add_argument("-l", "--limit", type=int, default=None)
    parser.add_argument("-e", "--epsilon", type=float, default=None)
    parser.add_argument("-s", "--save-path", default=None)
    parser.add_argument("--no-fake-exp", default=False, action="store_true")

    parsed = parser.parse_args()
    main(parsed)
