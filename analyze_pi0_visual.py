import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors


def main():
    repo_id = "aboardman/rollout_eval_test_2_20260611_102704"
    print(f"Loading dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id)

    # Setup config based on dataset features
    config = make_policy_config("pi0")
    config.device = "cpu"  # Using CPU to avoid OOM for analysis

    print("Initializing policy...")
    # NOTE: This requires HF authentication to download the gated paligemma model
    policy = make_policy(config, ds_meta=dataset.meta)

    # Preprocessor to properly format inputs
    preprocessor, _ = make_pi0_pre_post_processors(config)

    captured = {}

    # We use a pre-hook on paligemma_with_expert because embed_prefix is a python method,
    # not an nn.Module. This intercepts the prefix embeddings right after they are created!
    def hook_pre_paligemma(module, args, kwargs):
        # inputs_embeds is a list: [prefix_embs, suffix_embs]
        prefix_embs = kwargs.get("inputs_embeds", [None])[0]
        if prefix_embs is not None:
            captured["prefix_tokens"] = prefix_embs.detach().cpu()
            print(f"Captured Token Sequence Shape: {prefix_embs.shape}")

    handle = policy.model.paligemma_with_expert.register_forward_pre_hook(
        hook_pre_paligemma, with_kwargs=True
    )

    policy.eval()

    item = dataset[0]

    # LeRobot datasets return unbatched items, so we add batch dim
    batch = {
        "observation.images.arm": item["observation.images.arm"],
        "observation.images.overhead": item["observation.images.overhead"],
        "observation.state": item["observation.state"],
        "action": item["action"],
        "task": ["pick and place"],
    }

    # Apply the exact preprocessing the policy expects
    batch_proc = preprocessor(batch)

    with torch.no_grad():
        print("Running forward pass...")
        # predict_action_chunk runs the visual processing path via _preprocess_images -> embed_prefix
        _ = policy.predict_action_chunk(batch_proc)

    handle.remove()

    tokens = captured["prefix_tokens"]
    print(f"\nFull prefix: {tokens.shape}")
    print(f"Token norm (mean L2): {tokens.norm(dim=-1).mean():.3f}")

    # For PI0, PaliGemma's resolution is 224, patch size 14 -> 256 tokens per camera
    # Shape should be [1, 512 + lang_tokens, hidden_dim]


if __name__ == "__main__":
    main()
