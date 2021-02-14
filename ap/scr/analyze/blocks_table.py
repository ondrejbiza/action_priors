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
        loader, query, ["env_config.goal_string"], ["rewards"]
    )

    for key, items in res.items():

        items = np.array(items)[:, 0]
        res[key].append(items)

    for key, items in res.items():

        print("task: {}".format(key))
        print("final success rate: {:.2f}".format(items[-100:].mean()))


print_results("blocks_DQN_AAF")
print_results("blocks_DQN_HS")
print_results("blocks_DQN_RS")
