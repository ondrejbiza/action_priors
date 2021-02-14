import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ...utils import sacred_utils


def load(name):

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
        items = items[:10]
        # I want exactly 10 runs per task, and 100k training steps (500 * 200)
        # print(items.shape)
        # assert items.shape[0] == 10 and items.shape[1] == 200
        # this is string length instead of array length
        new_res[len(key[0])].append(items)

    for key, items in new_res.items():
        new_res[key] = np.concatenate(items, axis=0)

    return new_res


names_comb = [
    "fruits_DQN_AAF",
    "fruits_DQN_baselines",
    "fruits_DQN_transfer",
    "fruits_DQN_freeze_transfer",
    "fruits_DQN_side_transfer",
]

names_seq = [
    "fruits_seq_DQN_AAF",
    "fruits_seq_DQN_baselines",
    "fruits_seq_DQN_transfer",
    "fruits_seq_DQN_freeze_transfer",
    "fruits_seq_DQN_side_transfer",
]

labels = [
    "AP (ours)",
    "DQN",
    "AM-share",
    "AM-freeze",
    "AM-prog"
]

for names in [names_comb, names_seq]:

    results = []
    for name in names:
        results.append(load(name))

    d = {
        "Training Step": [],
        "Model": [],
        "Reward": [],
        "Style": []
    }

    for r, l in zip(results, labels):
        r = np.array(r[12])
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                d["Training Step"].append((j + 1) * 500)
                d["Model"].append(l)
                d["Reward"].append(r[i, j])
                d["Style"].append(0)

    df = pd.DataFrame(data=d)

    sns.set(font_scale=2)
    pal = sns.color_palette("colorblind")
    sns.set_palette(pal)

    plt.figure(figsize=(20, 12))
    sns.lineplot(x="Training Step", y="Reward", hue="Model", data=df, ci=95)
    plt.show()
