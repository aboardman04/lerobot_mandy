from lerobot.configs.policies import PreTrainedConfig

config = PreTrainedConfig.from_pretrained("aboardman/smolVLA_Training9")
print("input_features keys:", config.input_features.keys())
print("input_features:", config.input_features)
print("image_features:", config.image_features)
