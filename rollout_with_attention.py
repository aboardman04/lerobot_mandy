import numpy as np

from lerobot.configs import parser
from lerobot.rollout.configs import RolloutConfig

# Import LeRobot rollout internals
from lerobot.scripts.lerobot_rollout import build_rollout_context, create_strategy, init_logging
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler

# Import both custom viewers
from live_attention_act import LiveAttentionViewerACT
from live_attention_smolvla_v2 import LiveAttentionViewerSmolVLA_V2


@parser.wrap()
def rollout_attention(cfg: RolloutConfig):
    init_logging()

    # 1. Force turn off rerun so it doesn't conflict with our OpenCV window
    print("Disabling standard display_data (rerun) in favor of Live Attention Viewer.")
    cfg.display_data = False

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    print("Building rollout context (this will connect to the robot and load the policy)...")
    ctx = build_rollout_context(cfg, shutdown_event)

    # 2. ---> INJECT OUR ATTENTION VIEWER <---
    policy = ctx.policy.policy

    # Check if policy is unwrapped PEFT or base
    base_policy = policy
    if hasattr(policy, "base_model"):
        base_policy = policy.base_model

    policy_name = getattr(base_policy, "name", None)
    viewer = None

    if policy_name == "act":
        print("\n=== ATTACHING LIVE ATTENTION VIEWER (ACT) ===\n")
        viewer = LiveAttentionViewerACT(base_policy)
    elif policy_name == "smolvla":
        print("\n=== ATTACHING LIVE ATTENTION VIEWER (SmolVLA V2: Cross-Attention) ===\n")
        viewer = LiveAttentionViewerSmolVLA_V2(base_policy)
    else:
        print(f"\n[!] WARNING: Policy is '{policy_name}', which is not supported. Viewer Disabled.\n")

    if viewer is not None:
        original_select_action = base_policy.select_action

        def select_action_with_viewer(batch, *args, **kwargs):
            # Run the normal policy calculation (this triggers the hooks under the hood)
            action = original_select_action(batch, *args, **kwargs)

            # Extract the raw images currently passing through the policy
            arm_tensor = batch["observation.images.arm"].squeeze(0).cpu()
            overhead_tensor = batch["observation.images.overhead"].squeeze(0).cpu()

            # Convert back to (H, W, 3) in [0, 255] RGB for OpenCV
            arm_rgb = (arm_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            overhead_rgb = (overhead_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Update the live UI
            viewer.update_and_show(arm_rgb, overhead_rgb)

            return action

        # Replace the policy's function with our augmented one
        base_policy.select_action = select_action_with_viewer

    # 3. Hand control back to the standard LeRobot rollout strategy
    strategy = create_strategy(cfg.strategy)

    try:
        strategy.setup(ctx)
        print("Rollout setup complete, starting rollout...")
        strategy.run(ctx)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        strategy.teardown(ctx)
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    register_third_party_plugins()
    rollout_attention()
