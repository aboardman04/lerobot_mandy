import os

import cv2
import numpy as np
import torch


class LiveAttentionViewerACT:
    def __init__(self, policy, grid_size=(15, 20)):
        """
        A real-time attention visualizer specifically for ACT policies.
        """
        self.policy = policy
        self.grid_size = grid_size
        self.attention_weights = None
        self.backbone_features = []

        try:
            target_layer = self.policy.model.decoder.layers[-1].multihead_attn
            self.hook_handle = target_layer.register_forward_hook(self._hook_fn)
            self.backbone_hook_handle = self.policy.model.backbone.register_forward_hook(
                self._backbone_hook_fn
            )
            print("[✓] ACT Transformer & ResNet Hooks registered!")
        except Exception as e:
            print(f"[!] Could not hook ACT policy: {e}")

    def _hook_fn(self, module, input, output):
        self.attention_weights = output[1].detach().cpu()

    def _backbone_hook_fn(self, module, input, output):
        self.backbone_features.append(output["feature_map"].detach().cpu())

    def update_and_show(self, arm_img_rgb, overhead_img_rgb):
        arm_bgr = cv2.cvtColor(arm_img_rgb, cv2.COLOR_RGB2BGR)
        overhead_bgr = cv2.cvtColor(overhead_img_rgb, cv2.COLOR_RGB2BGR)
        cam_h, cam_w = arm_bgr.shape[:2]

        arm_overlay = arm_bgr.copy()
        overhead_overlay = overhead_bgr.copy()

        # 1. PROCESS ATTENTION MAPS
        if getattr(self, "attention_weights", None) is not None:
            raw_attn = self.attention_weights[0].mean(dim=0)
            h, w = self.grid_size
            grid_len = h * w

            arm_attn = raw_attn[2 : 2 + grid_len].reshape(h, w).numpy()
            overhead_attn = raw_attn[2 + grid_len : 2 + (grid_len * 2)].reshape(h, w).numpy()

            arm_attn = (arm_attn - arm_attn.min()) / (arm_attn.max() - arm_attn.min() + 1e-8)
            overhead_attn = (overhead_attn - overhead_attn.min()) / (
                overhead_attn.max() - overhead_attn.min() + 1e-8
            )

            arm_heatmap = cv2.applyColorMap(np.uint8(255 * arm_attn), cv2.COLORMAP_JET)
            overhead_heatmap = cv2.applyColorMap(np.uint8(255 * overhead_attn), cv2.COLORMAP_JET)

            arm_heatmap = cv2.resize(arm_heatmap, (cam_w, cam_h), interpolation=cv2.INTER_CUBIC)
            overhead_heatmap = cv2.resize(overhead_heatmap, (cam_w, cam_h), interpolation=cv2.INTER_CUBIC)

            arm_overlay = cv2.addWeighted(arm_bgr, 0.6, arm_heatmap, 0.4, 0)
            overhead_overlay = cv2.addWeighted(overhead_bgr, 0.6, overhead_heatmap, 0.4, 0)

        cv2.putText(
            arm_overlay, "Arm Attention (ACT)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        cv2.putText(
            overhead_overlay,
            "Overhead Attention (ACT)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # 2. PROCESS PCA TOKENS
        if len(self.backbone_features) >= 2:
            feat_arm = self.backbone_features[-2]
            feat_overhead = self.backbone_features[-1]
            self.backbone_features.clear()

            B, C, H, W = feat_arm.shape
            x_arm = feat_arm.permute(0, 2, 3, 1).reshape(-1, C)
            x_overhead = feat_overhead.permute(0, 2, 3, 1).reshape(-1, C)

            X = torch.cat([x_arm, x_overhead], dim=0)
            X_centered = X - X.mean(dim=0)

            try:
                U, S, V = torch.pca_lowrank(X_centered, q=3)
                X_proj = torch.matmul(X, V[:, :3])
                X_min = X_proj.min(dim=0, keepdim=True)[0]
                X_max = X_proj.max(dim=0, keepdim=True)[0]
                X_norm = (X_proj - X_min) / (X_max - X_min + 1e-8)
                X_rgb = (X_norm * 255).numpy().astype(np.uint8)

                rgb_arm = X_rgb[: H * W].reshape(H, W, 3)
                rgb_overhead = X_rgb[H * W :].reshape(H, W, 3)

                arm_pca = cv2.resize(rgb_arm, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)
                overhead_pca = cv2.resize(rgb_overhead, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)

                arm_pca_bgr = cv2.cvtColor(arm_pca, cv2.COLOR_RGB2BGR)
                overhead_pca_bgr = cv2.cvtColor(overhead_pca, cv2.COLOR_RGB2BGR)

                self.last_arm_pca = arm_pca_bgr
                self.last_overhead_pca = overhead_pca_bgr
            except Exception:
                pass
        else:
            self.backbone_features.clear()

        arm_pca_bgr = getattr(self, "last_arm_pca", np.zeros((cam_h, cam_w, 3), dtype=np.uint8))
        overhead_pca_bgr = getattr(self, "last_overhead_pca", np.zeros((cam_h, cam_w, 3), dtype=np.uint8))

        cv2.putText(
            arm_pca_bgr, "Arm PCA Tokens (ACT)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        cv2.putText(
            overhead_pca_bgr,
            "Overhead PCA Tokens (ACT)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Stitch
        top_row = np.hstack((overhead_overlay, arm_overlay))
        bottom_row = np.hstack((overhead_pca_bgr, arm_pca_bgr))
        combined_display = np.vstack((top_row, bottom_row))

        if not hasattr(self, "frame_idx"):
            self.frame_idx = 0
            self.output_dir = "attention_frames"
            os.makedirs(self.output_dir, exist_ok=True)

        cv2.imwrite(f"{self.output_dir}/frame_{self.frame_idx:04d}.jpg", combined_display)
        self.frame_idx += 1

        try:
            cv2.imshow("ACT Real-Time Viewer", combined_display)
            cv2.waitKey(1)
        except:
            pass

    def close(self):
        try:
            self.hook_handle.remove()
            self.backbone_hook_handle.remove()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except:
            pass
