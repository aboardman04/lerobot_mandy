import os

import cv2
import numpy as np
import torch


class LiveAttentionViewerSmolVLA:
    def __init__(self, policy):
        """
        A real-time visualizer for Vision Language Action models (SmolVLA).
        Hooks into the Vision Connector to map the Vision Transformer tokens via PCA.
        """
        self.policy = policy
        self.backbone_features = []

        try:
            target_layer = self.policy.model.vlm_with_expert.vlm.model.connector
            self.backbone_hook_handle = target_layer.register_forward_hook(self._backbone_hook_fn)
            print("[✓] SmolVLA ViT Hook registered!")
        except Exception as e:
            print(f"[!] Could not hook SmolVLA: {e}")
            self.backbone_hook_handle = None

    def _backbone_hook_fn(self, module, input, output):
        self.backbone_features.append(output.detach().cpu())

    def update_and_show(self, arm_img_rgb, overhead_img_rgb):
        arm_bgr = cv2.cvtColor(arm_img_rgb, cv2.COLOR_RGB2BGR)
        overhead_bgr = cv2.cvtColor(overhead_img_rgb, cv2.COLOR_RGB2BGR)
        cam_h, cam_w = arm_bgr.shape[:2]

        cv2.putText(
            arm_bgr,
            "Arm Camera (SmolVLA: 1D LLM, No Attn Map)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overhead_bgr,
            "Overhead Camera (SmolVLA)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # PROCESS PCA TOKENS
        if len(self.backbone_features) >= 2:
            feat_arm = self.backbone_features[-2]
            feat_overhead = self.backbone_features[-1]
            self.backbone_features.clear()

            while feat_arm.ndim > 2 and feat_arm.shape[0] == 1:
                feat_arm = feat_arm.squeeze(0)
                feat_overhead = feat_overhead.squeeze(0)

            num_tokens = feat_arm.shape[0]
            side_len = int(np.sqrt(num_tokens))
            h, w = side_len, side_len
            if h * w != num_tokens:
                w = int(np.ceil(np.sqrt(num_tokens)))
                h = int(np.ceil(num_tokens / w))

            X = torch.cat([feat_arm, feat_overhead], dim=0)
            X_centered = X - X.mean(dim=0)

            try:
                U, S, V = torch.pca_lowrank(X_centered, q=3)
                X_proj = torch.matmul(X, V[:, :3])

                X_min = X_proj.min(dim=0, keepdim=True)[0]
                X_max = X_proj.max(dim=0, keepdim=True)[0]
                X_norm = (X_proj - X_min) / (X_max - X_min + 1e-8)
                X_rgb = (X_norm * 255).numpy().astype(np.uint8)

                pad_needed = (h * w) - num_tokens
                if pad_needed > 0:
                    padding = np.zeros((pad_needed, 3), dtype=np.uint8)
                    x_rgb_arm = np.vstack([X_rgb[:num_tokens], padding])
                    x_rgb_overhead = np.vstack([X_rgb[num_tokens:], padding])
                else:
                    x_rgb_arm = X_rgb[:num_tokens]
                    x_rgb_overhead = X_rgb[num_tokens:]

                rgb_arm = x_rgb_arm.reshape(h, w, 3)
                rgb_overhead = x_rgb_overhead.reshape(h, w, 3)

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
            arm_pca_bgr, "Arm ViT Tokens (SmolVLA)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        cv2.putText(
            overhead_pca_bgr,
            "Overhead ViT Tokens (SmolVLA)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        top_row = np.hstack((overhead_bgr, arm_bgr))
        bottom_row = np.hstack((overhead_pca_bgr, arm_pca_bgr))
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
            if self.backbone_hook_handle:
                self.backbone_hook_handle.remove()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except:
            pass
