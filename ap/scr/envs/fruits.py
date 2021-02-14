import numpy as np
import matplotlib.pyplot as plt
from ...envs.fruits import Fruits


def show_image(image):

    plt.imshow(image)
    plt.show()


env = Fruits()
for f in env.fruits:
    print(f)
print(env.goal)

while True:

    state = env.get_state()
    print("optimal action: {:d}".format(env.get_optimal_action_()))
    show_image(Fruits.state_to_image(state))

    action = int(input("action: "))

    print("action: {:s}".format(env.get_abstract_action_name(env.get_abstract_action(action))))

    _, reward, done, _ = env.step(action)

    print("reward: {:.1f}, done: {}".format(reward, done))

    if done:
        env.reset()
        for f in env.fruits:
            print(f)