import subprocess

cmd = [
    "lerobot-train",
    "--config",
    "outputs/train/my_smolvla14/checkpoints/000002/pretrained_model/train_config.json",
]

subprocess.run(cmd)
