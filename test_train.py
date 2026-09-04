import json
import subprocess

with open("outputs/train/my_smolvla14/checkpoints/000002/pretrained_model/train_config.json") as f:
    config = json.load(f)

# Reconstruct the command line for lerobot-train
cmd = [
    "lerobot-train",
    "--policy.type=smolvla",
    "--policy.repo_id=aboardman/smolVLA_Instrament_training_2",
    "--dataset.repo_id=aboardman/combined_instrument_dataset_6-7",
    "--steps=2",
    "--batch_size=16",
    "--output_dir=outputs/train/test_crash",
    "--wandb.enable=false",
    "--policy.push_to_hub=true",
]

subprocess.run(cmd)
