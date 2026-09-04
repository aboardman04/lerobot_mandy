import sys

sys.path.append("src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors


def main():
    repo_id = "aboardman/rollout_eval_test_2_20260611_102704"
    dataset = LeRobotDataset(repo_id)

    config = make_policy_config("smolvla")
    config.device = "cpu"  # Using CPU to avoid OOM for analysis

    policy = make_policy(config, ds_meta=dataset.meta)
    preprocessor, _ = make_smolvla_pre_post_processors(config, dataset_stats=dataset.meta.stats)

    item = dataset[0]
    batch = {
        "observation.images.arm": item["observation.images.arm"],
        "observation.images.overhead": item["observation.images.overhead"],
        "observation.state": item["observation.state"],
        "action": item["action"],
        "task": "pick and place object",  # Passed as string!
    }

    print("=== 1. Raw Dataset Item ===")
    print(
        f"observation.images.arm shape: {batch['observation.images.arm'].shape}, range: [{batch['observation.images.arm'].min():.3f}, {batch['observation.images.arm'].max():.3f}]"
    )
    print(
        f"observation.images.overhead shape: {batch['observation.images.overhead'].shape}, range: [{batch['observation.images.overhead'].min():.3f}, {batch['observation.images.overhead'].max():.3f}]"
    )
    print(
        f"observation.state shape: {batch['observation.state'].shape}, values: {batch['observation.state']}"
    )

    print("\n=== 2. Applying Full Preprocessor Pipeline ===")
    batch_proc = preprocessor(batch)

    print("\n=== 3. Processed Visual Data (Post-Preprocessor, Pre-Model) ===")
    print(f"observation.images.arm shape: {batch_proc['observation.images.arm'].shape}")
    print(
        f"observation.images.arm range: [{batch_proc['observation.images.arm'].min():.3f}, {batch_proc['observation.images.arm'].max():.3f}]"
    )
    print(
        "  * Notice that SmolVLA preprocessor does NOT resize or rescale the images! They are still 480x640 and in [0, 1]."
    )

    print("\n=== 4. Processed State Data ===")
    print(f"observation.state shape: {batch_proc['observation.state'].shape}")
    print(f"observation.state values: {batch_proc['observation.state'][0]}")
    print("  * The proprioceptive state has been normalized using dataset statistics (mean/std).")

    print("\n=== 5. Processed Language Data ===")
    print(f"observation.language.tokens shape: {batch_proc['observation.language.tokens'].shape}")
    print(f"observation.language.tokens IDs: {batch_proc['observation.language.tokens'][0].tolist()}")
    print("  * The string has been tokenized by the SmolVLM tokenizer.")

    print("\n=== 6. Inside SmolVLAPolicy (`prepare_images`) ===")
    images, img_masks = policy.prepare_images(batch_proc)

    print(f"\nImages array length: {len(images)} (one tensor per camera)")
    print(f"Image 0 shape (arm): {images[0].shape}")
    print(f"Image 0 range: [{images[0].min():.3f}, {images[0].max():.3f}]")
    print(
        "  * Ah ha! The model internally rescales the images from [0, 1] to [-1, 1] inside `prepare_images()`"
    )
    print("  * Since config.resize_imgs_with_padding is None by default, it does NOT resize them.")

    print(f"\nResize config value: {config.resize_imgs_with_padding}")


if __name__ == "__main__":
    main()
