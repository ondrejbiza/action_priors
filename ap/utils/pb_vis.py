import numpy as np
import matplotlib.pyplot as plt


def plot_obs_and_hand(obs, hand, from_pt=True):

    if from_pt:
        obs = obs.detach().cpu().numpy()
        hand = hand.detach().cpu().numpy()

    obs = obs[0, :, :, 0]
    hand = hand[0, :, :, 0]

    fig = plt.figure()

    coords = []

    def onclick(event):
        x, y = event.xdata, event.ydata
        coords.append([x, y])
        plt.close()

    fig.canvas.mpl_connect("button_press_event", onclick)

    plt.subplot(1, 2, 1)
    plt.imshow(obs)
    plt.subplot(1, 2, 2)
    plt.imshow(hand)
    plt.show()

    if len(coords) == 0:
        return None
    else:
        return coords[-1]


def coords_to_action(coords, x_range, y_range, size):

    coords = np.array(coords, dtype=np.float32)
    coords /= float(size)

    x_len = x_range[1] - x_range[0]
    x_p = x_len * coords[0]
    x = x_range[0] + x_p

    y_len = y_range[1] - y_range[0]
    y_p = y_len * coords[1]
    y = y_range[0] + y_p

    return x, y


def check_hand_empty(hand):

    hand = hand.detach().cpu().numpy()
    return hand[0] == 0.0
