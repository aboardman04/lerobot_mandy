import cv2
import matplotlib.pyplot as plt
import torch

from lerobot.policies.act.modeling_act import ACTPolicy


def main():
    # 1. Define the policy to load
    policy_path = "aboardman/act_so101_test11_policy"
    print(f"Loading policy: {policy_path} ...")

    # ACTPolicy natively supports from_pretrained from Hugging Face hub or local dirs
    policy = ACTPolicy.from_pretrained(policy_path)
    policy.eval()
    device = next(policy.parameters()).device

    print("Policy loaded successfully.")

    # 2. Setup a "Hook" to eavesdrop on the Attention Layer
    attention_weights = []

    def hook_fn(module, input, output):
        # PyTorch's MultiheadAttention returns a tuple: (attn_output, attn_weights)
        # output[1] contains the attention weights: (Batch, Num_Queries, Total_Keys)
        attn = output[1].detach().cpu()
        attention_weights.append(attn)

    # In ACT, the cross-attention between Actions (Queries) and Images (Keys)
    # happens in the Decoder. We attach our hook to the last (and only) layer's multihead_attn.
    target_layer = policy.model.decoder.layers[-1].multihead_attn
    hook_handle = target_layer.register_forward_hook(hook_fn)

    # 3. Create dummy inputs (replace these with your actual camera frames later!)
    print("Loading specific frames from dataset videos...")
    import av

    frame_idx = 100

    def extract_frame(video_path, frame_idx):
        container = av.open(video_path)
        stream = container.streams.video[0]
        # Seek to slightly before the frame to handle keyframes
        # av uses time_base, so we approximate
        for i, frame in enumerate(container.decode(video=0)):
            if i == frame_idx:
                return frame.to_ndarray(format="rgb24")
        return None

    dummy_arm_img = extract_frame("videos/observation.images.arm/chunk-000/file-000.mp4", 100)
    dummy_overhead_img = extract_frame("videos/observation.images.overhead/chunk-000/file-000.mp4", 100)

    if dummy_arm_img is None or dummy_overhead_img is None:
        raise ValueError("Could not extract frames from video.")

    # Optionally load the exact robot state from the parquet file for maximum accuracy
    import pandas as pd

    df = pd.read_parquet("data/chunk-000/file-000.parquet")
    # State dimension for SO101 is typically 14
    state_dim = policy.config.robot_state_feature.shape[0]

    # State data format in HF dataset: typically list or array per row.
    # df['observation.state'][frame_idx] should have the 14-dim array
    state_arr = df["observation.state"].iloc[frame_idx]
    state_tensor = torch.tensor(state_arr, dtype=torch.float32)

    # LeRobot expects (C, H, W) normalized to [0, 1] as float32
    arm_tensor = torch.from_numpy(dummy_arm_img).permute(2, 0, 1).float() / 255.0
    overhead_tensor = torch.from_numpy(dummy_overhead_img).permute(2, 0, 1).float() / 255.0

    # Format the batch
    batch = {
        "observation.state": state_tensor.unsqueeze(0).to(device),  # Add batch dim
        "observation.images.arm": arm_tensor.unsqueeze(0).to(device),
        "observation.images.overhead": overhead_tensor.unsqueeze(0).to(device),
    }

    # Run the model
    with torch.no_grad():
        # select_action handles the chunk generation
        _ = policy.select_action(batch)

    # Clean up the hook
    hook_handle.remove()

    # 4. Extract and Reshape the Grids
    # Shape is [Batch=1, Chunk_Size=100, Total_Tokens=602]
    raw_attn = attention_weights[0][0]

    # Average the attention across all actions in the chunk to get the "overall" focus
    overall_attn = raw_attn.mean(dim=0)  # Shape: [602]

    # The 602 tokens map to:
    # Token 0: Latent
    # Token 1: Robot State
    # Tokens 2 to 301: Arm Camera Grid (300 tokens)
    # Tokens 302 to 601: Overhead Camera Grid (300 tokens)
    # ResNet18 downsamples 480x640 by a factor of 32, giving a 15x20 grid (15 * 20 = 300)
    grid_h, grid_w = 15, 20

    arm_attn_tokens = overall_attn[2:302]
    overhead_attn_tokens = overall_attn[302:602]

    arm_grid = arm_attn_tokens.reshape(grid_h, grid_w).numpy()
    overhead_grid = overhead_attn_tokens.reshape(grid_h, grid_w).numpy()

    # Normalize between 0 and 1 for plotting
    arm_grid = (arm_grid - arm_grid.min()) / (arm_grid.max() - arm_grid.min() + 1e-8)
    overhead_grid = (overhead_grid - overhead_grid.min()) / (overhead_grid.max() - overhead_grid.min() + 1e-8)

    # Resize the 15x20 grids back up to 480x640 for the overlay
    arm_resized = cv2.resize(arm_grid, (640, 480), interpolation=cv2.INTER_CUBIC)
    overhead_resized = cv2.resize(overhead_grid, (640, 480), interpolation=cv2.INTER_CUBIC)

    # 5. Plot and Save
    print("Generating attention plots...")
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    # Arm Camera
    axs[0, 0].imshow(dummy_arm_img)
    axs[0, 0].set_title("Arm Camera (Raw)")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(dummy_arm_img)
    axs[0, 1].imshow(arm_resized, cmap="jet", alpha=0.5)
    axs[0, 1].set_title("Arm Camera Attention")
    axs[0, 1].axis("off")

    # Overhead Camera
    axs[1, 0].imshow(dummy_overhead_img)
    axs[1, 0].set_title("Overhead Camera (Raw)")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(dummy_overhead_img)
    axs[1, 1].imshow(overhead_resized, cmap="jet", alpha=0.5)
    axs[1, 1].set_title("Overhead Camera Attention")
    axs[1, 1].axis("off")

    plt.tight_layout()
    output_file = "attention_heatmap.png"
    plt.savefig(output_file, dpi=150)
    print(f"Success! Attention heatmap saved to '{output_file}'")


if __name__ == "__main__":
    main()
