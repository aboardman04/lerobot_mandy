import os

import cv2
import numpy as np
import torch


class LiveAttentionViewerSmolVLA_V2:
    def __init__(self, policy):
        """
        A real-time visualizer for Vision Language Action models (SmolVLA).
        This version computes actual Cross-Attention maps from the Action tokens
        (in the denoising steps) back to the Image tokens (in the VLM prefix context).
        """
        self.policy = policy
        self.attention_maps = []
        self.image_token_lengths = []

        self.vlm_with_expert = self.policy.model.vlm_with_expert

        self.add_image_special_tokens = getattr(self.policy.model, "add_image_special_tokens", True)

        # 1. Hook embed_image to dynamically know exactly how many patches each image generates
        self.original_embed_image = self.vlm_with_expert.embed_image

        def patched_embed_image(img):
            out = self.original_embed_image(img)
            self.image_token_lengths.append(out.shape[1])
            return out

        self.vlm_with_expert.embed_image = patched_embed_image

        # 2. Monkey-patch eager_attention_forward to capture actual attention probabilities
        self.original_eager_attn = self.vlm_with_expert.eager_attention_forward

        def patched_eager_attn(attention_mask, batch_size, head_dim, query_states, key_states, value_states):
            out = self.original_eager_attn(
                attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )

            q_len = query_states.shape[1]
            k_len = key_states.shape[1]

            # The denoising step produces action steps as `suffix`.
            # Typically q_len == chunk_size (e.g. 64) and k_len includes images + language + suffix
            if q_len > 0 and q_len < k_len and k_len > 100:
                with torch.no_grad():
                    # Recompute only the attention weights locally to capture them
                    num_att_heads = self.vlm_with_expert.num_attention_heads
                    num_key_value_heads = getattr(self.vlm_with_expert, "num_key_value_heads", num_att_heads)
                    num_key_value_groups = num_att_heads // num_key_value_heads

                    seq_len = key_states.shape[1]

                    k_st = (
                        key_states[:, :, :, None, :]
                        .expand(batch_size, seq_len, num_key_value_heads, num_key_value_groups, head_dim)
                        .reshape(batch_size, seq_len, num_att_heads, head_dim)
                        .to(torch.float32)
                        .transpose(1, 2)
                    )

                    q_st = query_states.to(torch.float32).transpose(1, 2)

                    att_weights = torch.matmul(q_st, k_st.transpose(2, 3)) * (head_dim**-0.5)
                    big_neg = torch.finfo(att_weights.dtype).min
                    masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
                    probs = torch.nn.functional.softmax(masked_att_weights, dim=-1)

                    # Probs shape: [batch_size, num_heads, q_len, k_len]
                    # Average over all attention heads and all action queries (q_len)
                    avg_probs = probs.mean(dim=(1, 2)).detach().cpu()  # Shape: [batch_size, k_len]
                    self.attention_maps.append(avg_probs)

            return out

        self.vlm_with_expert.eager_attention_forward = patched_eager_attn

    def update_and_show(self, arm_img_rgb, overhead_img_rgb):
        arm_bgr = cv2.cvtColor(arm_img_rgb, cv2.COLOR_RGB2BGR)
        overhead_bgr = cv2.cvtColor(overhead_img_rgb, cv2.COLOR_RGB2BGR)
        cam_h, cam_w = arm_bgr.shape[:2]

        cv2.putText(
            arm_bgr, "Arm Camera (SmolVLA input)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        cv2.putText(
            overhead_bgr,
            "Overhead Camera (SmolVLA input)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        attn_heatmap_arm = np.zeros_like(arm_bgr)
        attn_heatmap_overhead = np.zeros_like(overhead_bgr)

        if len(self.attention_maps) > 0 and len(self.image_token_lengths) >= 2:
            if len(self.attention_maps) > 0:
                # Some maps might have different lengths due to dynamic sequence lengths
                # Find the maximum length and pad before stacking
                max_len = max([am.shape[1] for am in self.attention_maps])
                padded_maps = []
                for am in self.attention_maps:
                    if am.shape[1] < max_len:
                        pad_amount = max_len - am.shape[1]
                        am = torch.nn.functional.pad(am, (0, pad_amount), value=0)
                    padded_maps.append(am)
                total_attn = torch.stack(padded_maps).mean(dim=0).squeeze(0)  # Shape: [max_k_len]
            else:
                total_attn = None

            if total_attn is not None:
                # Extract the actual images from the sequence
                idx = 0

                img1_len = self.image_token_lengths[-2]
                if self.add_image_special_tokens:
                    idx += 1
                attn_img1 = total_attn[idx : idx + img1_len]
                idx += img1_len
                if self.add_image_special_tokens:
                    idx += 1  # End token for img1

                img2_len = self.image_token_lengths[-1]
                if self.add_image_special_tokens:
                    idx += 1  # Start token for img2
                attn_img2 = total_attn[idx : idx + img2_len]
                # ... and an end token follows but we don't need to advance idx anymore

                self.attention_maps.clear()
                self.image_token_lengths.clear()

                # Normalize and reshape to 2D
                def process_attn(attn_1d, target_h, target_w):
                    num_tokens = attn_1d.shape[0]
                    if num_tokens == 0:
                        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    side_len = int(np.sqrt(num_tokens))
                    h_feat, w_feat = side_len, side_len
                    if h_feat * w_feat != num_tokens:
                        w_feat = int(np.ceil(np.sqrt(num_tokens)))
                        h_feat = int(np.ceil(num_tokens / w_feat))

                    pad = (h_feat * w_feat) - num_tokens
                    if pad > 0:
                        attn_1d = torch.cat([attn_1d, torch.zeros(pad)])

                    attn_2d = attn_1d.reshape(h_feat, w_feat).numpy()
                    attn_2d = np.clip(attn_2d, 0, None)
                    if attn_2d.max() > 0:
                        attn_2d = attn_2d / attn_2d.max()

                    attn_2d = cv2.resize(attn_2d, (target_w, target_h))

                    # Apply colormap
                    heatmap = np.uint8(255 * attn_2d)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    return heatmap

                # Assuming the images are passed in the order defined by the dataset.
                # In LeRobot SmolVLA usually the cameras are sorted alphabetically or according to config.
                # Assuming [arm, overhead] or [overhead, arm]. Let's try [arm, overhead] mapping.
                # If colors look swapped, we swap them.

                heatmap1 = process_attn(attn_img1, cam_h, cam_w)
                heatmap2 = process_attn(attn_img2, cam_h, cam_w)

                # Blend with original image
                attn_heatmap_arm = cv2.addWeighted(arm_bgr, 0.5, heatmap1, 0.5, 0)
                attn_heatmap_overhead = cv2.addWeighted(overhead_bgr, 0.5, heatmap2, 0.5, 0)

                self.last_arm_hm = attn_heatmap_arm
                self.last_overhead_hm = attn_heatmap_overhead
        else:
            self.attention_maps.clear()
            self.image_token_lengths.clear()

        arm_out = getattr(self, "last_arm_hm", np.zeros((cam_h, cam_w, 3), dtype=np.uint8))
        overhead_out = getattr(self, "last_overhead_hm", np.zeros((cam_h, cam_w, 3), dtype=np.uint8))

        cv2.putText(
            arm_out, "Arm Cross-Attention Map", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        cv2.putText(
            overhead_out,
            "Overhead Cross-Attention Map",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        top_row = np.hstack((overhead_bgr, arm_bgr))
        bottom_row = np.hstack((overhead_out, arm_out))
        combined_display = np.vstack((top_row, bottom_row))

        if not hasattr(self, "frame_idx"):
            self.frame_idx = 0
            self.output_dir = "attention_frames"
            os.makedirs(self.output_dir, exist_ok=True)

        cv2.imwrite(f"{self.output_dir}/frame_{self.frame_idx:04d}.jpg", combined_display)
        self.frame_idx += 1

        try:
            cv2.imshow("SmolVLA Real-Time Viewer", combined_display)
            cv2.waitKey(1)
        except:
            pass

    def close(self):
        try:
            # Revert hooks if needed
            self.vlm_with_expert.embed_image = self.original_embed_image
            self.vlm_with_expert.eager_attention_forward = self.original_eager_attn
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except:
            pass
