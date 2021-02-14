import time
import argparse
from scipy.misc import imsave
import torch
import pybullet as pb
import numpy as np
from helping_hands_rl_envs.env_factory import createEnvs
import utils.pb_vis as vis_utils
from helping_hands_rl_envs.envs.house_building_x_deconstruct_env import createHouseBuildingXDeconstructEnv
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv


def main(args):

    workspace = np.asarray([[0.35, 0.65], [-0.15, 0.15], [0, 0.50]])
    env_config = {
        "workspace": workspace,
        "max_steps": 10,
        "obs_size": 90,
        "fast_mode": False,
        "render": True,
        "goal_string": args.goal,
        "gen_blocks": 4,
        "gen_bricks": 2,
        "gen_triangles": 1,
        "gen_roofs": 1,
        "small_random_orientation": False,
        "seed": np.random.randint(100),
        "action_sequence": "xyrp",
        "simulate_grasp": True
    }

    env = createHouseBuildingXDeconstructEnv(PyBulletEnv, env_config)()
    env.reset()

    viewMatrix = pb.computeViewMatrix(
        cameraEyePosition=[1, 0, 0.2],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 0, 1])

    projectionMatrix = pb.computeProjectionMatrixFOV(
        fov=45.0,
        aspect=1.0,
        nearVal=0.1,
        farVal=3.1)

    width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
        width=224,
        height=224,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix)

    imsave("{:s}.png".format(args.goal), rgbImg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("goal")

    parsed = parser.parse_args()
    main(parsed)
