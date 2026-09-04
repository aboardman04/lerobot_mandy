import json

import draccus

from lerobot.configs.train import TrainPipelineConfig

checkpoint_dir = "outputs/train/my_smolvla14/checkpoints/000002"
with open(f"{checkpoint_dir}/pretrained_model/train_config.json") as f:
    cfg_dict = json.load(f)

if "checkpoint_path" in cfg_dict:
    del cfg_dict["checkpoint_path"]

cfg = draccus.decode(TrainPipelineConfig, cfg_dict)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained(f"{checkpoint_dir}/pretrained_model", device="cpu")
policy.push_model_to_hub(cfg)
