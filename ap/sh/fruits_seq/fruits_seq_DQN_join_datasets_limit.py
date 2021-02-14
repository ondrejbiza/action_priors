import json
import subprocess
from ... import paths


models_path = "data/fruits_seq_DQN_models"
dataset_path = "data/fruits_seq_DQN_dsets"
with open(paths.TASKS_FRUITS_SEQ, "r") as f:
    tasks = json.load(f)

d_paths = []
m_paths = []

save_path = "{:s}/dset_eps_0_5_all_20k.h5".format(dataset_path)
c_string = ",".join([str(c).replace("[", "").replace("]", "").replace(" ", "").replace(",", "") for c in tasks])
print(c_string)

for c in tasks:

    d_paths.append("{:s}/dset_eps_0_5_{:s}.h5".format(dataset_path, str(c)))
    m_paths.append("{:s}/model_{:s}.pt".format(models_path, str(c)))

subprocess.call([
    "python", "-m", "ap.scr.online.fruits_seq.join_dsets", "with", "datasets_list={:s}".format(str(d_paths)),
    "models_list={:s}".format(str(m_paths)), "dset_save_path={:s}".format(save_path), "exp_per_dataset=20000",
    "c_string={:s}".format(c_string)
])
