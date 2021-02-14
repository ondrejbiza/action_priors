import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_losses(losses, names, validation=False, epoch_size=None):

    assert validation is False or epoch_size is not None

    x = np.array(list(range(len(losses[0]))), dtype=np.int32)

    if validation:
        x *= epoch_size

    for loss, name in zip(losses, names):
        plt.plot(x, loss, label=name, linestyle="--" if validation else "-")
    plt.legend()


def plot_log_losses(losses, names, validation=False, epoch_size=None):
    plot_losses(losses, names, validation=validation, epoch_size=epoch_size)
    plt.yscale("log")


def states_to_torch(states, device):

    # BxHxWxC to BxCxHxW
    states = np.transpose(states, axes=(0, 3, 1, 2))
    return other_to_torch(states, device)


def other_to_torch(x, device):

    return torch.from_numpy(x).to(device)


def parse_task_string(task_string, return_char_list=False):

    x = task_string.split(",")

    if not return_char_list:
        return x

    y = []

    for c in x:

        z = []

        for s in c:
            z.append(int(s))

        y.append(z)

    return y
