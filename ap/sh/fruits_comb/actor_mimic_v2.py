import argparse
import json
import copy as cp
import os
import subprocess
from ... import paths


def get_all_goals_and_load_paths():

    with open(paths.TASKS_FRUITS_COMB, "r") as f:
        task_list = json.load(f)

    load_paths = []

    for idx, c in enumerate(task_list):
        load_paths.append(os.path.join("data/fruits_DQN_models", "model_{:s}.pt".format(str(c))))

    return task_list, load_paths


def main(args):

    goals, load_paths = get_all_goals_and_load_paths()
    print(goals)

    save_path = "data/fruits_DQN_students"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for goal_idx in range(len(goals)):

        tmp_goals = cp.deepcopy(goals)
        tmp_load_paths = cp.deepcopy(load_paths)
        del tmp_goals[goal_idx]
        del tmp_load_paths[goal_idx]

        student_save_path = os.path.join(save_path, "model_ignore_{:d}.pt".format(goal_idx))

        subprocess.call([
            "python", "-m", "ap.scr.offline.fruits.run_actor_mimic_v2", "--name", "fruits_actor_mimic_v2",
            "with", "device={:s}".format(args.device), "teacher_load_paths={:s}".format(str(tmp_load_paths)),
            "goals={:s}".format(str(tmp_goals)), "student_save_path={:s}".format(student_save_path)
        ])


parser = argparse.ArgumentParser()

parser.add_argument("device")

parsed = parser.parse_args()
main(parsed)
