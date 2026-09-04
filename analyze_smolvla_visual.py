import sys

import torch

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
    preprocessor, _ = make_smolvla_pre_post_processors(config)

    captured = {}

    # 1. Capture Vision Encoder Outputs
    def hook_vision(module, input, output):
        if "vision_encoder_tokens" not in captured:
            captured["vision_encoder_tokens"] = []
        captured["vision_encoder_tokens"].append(output.last_hidden_state.detach().cpu())

    handle1 = policy.model.vlm_with_expert.get_vlm_model().vision_model.register_forward_hook(hook_vision)

    # 2. Capture Resampled Image Tokens
    original_embed_image = policy.model.vlm_with_expert.embed_image

    def hooked_embed_image(image):
        img_emb = original_embed_image(image)
        if "resampled_image_tokens" not in captured:
            captured["resampled_image_tokens"] = []
        captured["resampled_image_tokens"].append(img_emb.detach().cpu())
        return img_emb

    policy.model.vlm_with_expert.embed_image = hooked_embed_image

    # 3. Capture Full Prefix (to get final token sequence)
    original_embed_prefix = policy.model.embed_prefix

    def hooked_embed_prefix(*args, **kwargs):
        prefix_embs, prefix_pad_masks, prefix_att_masks = original_embed_prefix(*args, **kwargs)
        captured["prefix_tokens"] = prefix_embs.detach().cpu()
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    policy.model.embed_prefix = hooked_embed_prefix

    policy.eval()

    item = dataset[0]
    batch = {
        "observation.images.arm": item["observation.images.arm"],
        "observation.images.overhead": item["observation.images.overhead"],
        "observation.state": item["observation.state"],
        "action": item["action"],
        "task": "pick and place object",  # Passed as string!
    }

    batch_proc = preprocessor(batch)

    with torch.no_grad():
        _ = policy.predict_action_chunk(batch_proc)

    handle1.remove()

    print("\n=== SmolVLA Token Processing Analysis ===")

    # Analyze Vision Encoder
    print("\n1. Raw Vision Encoder (SigLIP):")
    for i, t in enumerate(captured["vision_encoder_tokens"]):
        print(f"   Camera {i}: {t.shape[1]} tokens (Shape: {t.shape})")

    # Analyze Resampling
    print("\n2. Resampled Image Tokens (Connector):")
    for i, t in enumerate(captured["resampled_image_tokens"]):
        print(f"   Camera {i}: {t.shape[1]} tokens (Shape: {t.shape})")

    # Analyze Final Prefix
    prefix = captured["prefix_tokens"]
    print("\n3. Full Prefix Sequence:")
    print(f"   Total Tokens: {prefix.shape[1]} (Shape: {prefix.shape})")

    # Do the math
    cam_tokens = sum(t.shape[1] for t in captured["resampled_image_tokens"])
    special_tokens = 2 * len(captured["resampled_image_tokens"])  # <image> and </image> per camera
    state_tokens = 1  # State projection sequence length is usually 1
    lang_tokens = prefix.shape[1] - cam_tokens - special_tokens - state_tokens

    print("\n=== Token Math ===")
    print(f"   {cam_tokens} (Visual: {captured['resampled_image_tokens'][0].shape[1]} per camera)")
    print(f" + {special_tokens} (Special <image> & </image> tokens)")
    print(f" + {lang_tokens} (Language tokens)")
    print(f" + {state_tokens} (State token)")
    print(f" = {prefix.shape[1]} Total Prefix Tokens")


if __name__ == "__main__":
    main()
