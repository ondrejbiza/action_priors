import collections
import numpy as np
from ...utils import sacred_utils


def print_results(name):

    print()
    print("method: {:s}".format(name))

    loader = sacred_utils.get_experiment_loader()

    query = {
        "experiment.name": name
    }

    res = sacred_utils.execute_query(
        loader, query, ["goal"], ["eval_total_rewards"]
    )

    new_res = collections.defaultdict(list)

    for key, items in res.items():

        items = np.array(items)[:, 0]
        # this is string length instead of array length
        new_res[len(key[0])].append(items)

    for key, items in new_res.items():

        new_res[key] = np.concatenate(items, axis=0)

    for key, items in new_res.items():

        print("num fruits: {:d}".format(key // 3))

        num_evals = items.shape[1]
        mid_point_idx = num_evals // 2

        mid_point = items[:, mid_point_idx].mean()
        mid_point_window = items[:, mid_point_idx - 5: mid_point_idx + 1].mean()
        fin_point = items[:, -1].mean()
        fin_point_window = items[:, -10:].mean()

        print("mid point: {:.2f}, fin point: {:.2f}".format(mid_point * 100, fin_point * 100))
        print("min window: {:.2f}, fin window: {:.2f}".format(mid_point_window * 100, fin_point_window * 100))


print_results("fruits_seq_DQN_AAF")
print_results("fruits_seq_DQN_baselines")
print_results("fruits_seq_DQN_transfer")
print_results("fruits_seq_DQN_freeze_transfer")
print_results("fruits_seq_DQN_side_transfer")
