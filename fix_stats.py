import json
import torch
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = "/home/aboardman/lerobot/lerobot_ee_conversion/output"
ds = LeRobotDataset('local', root=root)

# We can just iterate over all data to compute the mean, min, max, std
all_actions = ds.hf_dataset["action"]
all_states = ds.hf_dataset["observation.state"]

actions = np.array(all_actions, dtype=np.float32)
states = np.array(all_states, dtype=np.float32)

print("actions shape", actions.shape)
print("states shape", states.shape)

with open(f"{root}/meta/stats.json", "r") as f:
    stats = json.load(f)

# Update action stats
stats["action"]["min"] = np.min(actions, axis=0).tolist()
stats["action"]["max"] = np.max(actions, axis=0).tolist()
stats["action"]["mean"] = np.mean(actions, axis=0).tolist()
stats["action"]["std"] = np.std(actions, axis=0).tolist()

# Update state stats
stats["observation.state"]["min"] = np.min(states, axis=0).tolist()
stats["observation.state"]["max"] = np.max(states, axis=0).tolist()
stats["observation.state"]["mean"] = np.mean(states, axis=0).tolist()
stats["observation.state"]["std"] = np.std(states, axis=0).tolist()

with open(f"{root}/meta/stats.json", "w") as f:
    json.dump(stats, f, indent=4)

print("Fixed stats.json")
