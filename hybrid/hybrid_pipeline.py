import os
import torch
import numpy as np
import cv2

from models.unet import GuidedDepthUNet
from classical.classical_methods import (
    bicubic_upsample,
    fill_depth_holes,
    guided_filter_refinement,
    joint_bilateral_filter,
    detect_depth_discontinuities,
    edge_gated_boundary_refinement
)


class HybridDepthPipeline:
    """
    Hybrid Edge-Preserving Depth Reconstruction Pipeline:
    Combines:
      - Classical Preprocessing (hole filling & bicubic upsampling)
      - Deep Learning Guided U-Net (global geometry & metric structure recovery)
      - Edge-Gated Boundary Refinement (prevents texture-copying and sharpens object edges)
    """
    def __init__(self, model_weights_path=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = GuidedDepthUNet(in_channels=5, out_channels=1, base_filters=32, use_residual=True)
        self.weights_loaded = False
        self.loaded_weights_path = None

        # Resolve weights path if not directly found
        candidates = []
        if model_weights_path:
            candidates.append(model_weights_path)
            candidates.append(os.path.abspath(model_weights_path))

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates.extend([
            os.path.join(base_dir, "models", "unet_depth_best.pth"),
            os.path.join(base_dir, "unet_depth_best.pth"),
            os.path.join(base_dir, "models", "unet_depth_final.pth"),
            os.path.join(base_dir, "models", "unet_depth.pth")
        ])

        valid_path = None
        for p in candidates:
            if p and os.path.exists(p) and os.path.isfile(p):
                valid_path = p
                break

        if valid_path:
            try:
                state_dict = torch.load(valid_path, map_location=self.device)
                # Strip 'module.' prefix if saved with DataParallel
                cleaned_sd = {}
                for k, v in state_dict.items():
                    key = k[7:] if k.startswith("module.") else k
                    cleaned_sd[key] = v

                load_res = self.model.load_state_dict(cleaned_sd, strict=False)
                self.model.to(self.device)
                self.model.eval()
                self.weights_loaded = True
                self.loaded_weights_path = valid_path
                print(f"[HybridPipeline] Successfully loaded weights from {valid_path}")
                if len(load_res.unexpected_keys) > 0:
                    print(f"[HybridPipeline] Note unexpected keys: {load_res.unexpected_keys[:3]}")
            except Exception as e:
                print(f"[HybridPipeline] Error loading weights from {valid_path}: {e}")
                self.model.to(self.device)
                self.model.eval()
        else:
            self.model.to(self.device)
            self.model.eval()
            print("[HybridPipeline] Warning: No trained weights file found on disk.")

    def preprocess_inputs(self, rgb_img, lr_depth, target_shape=(256, 320)):
        target_h, target_w = target_shape

        if rgb_img.shape[:2] != target_shape:
            rgb_resized = cv2.resize(rgb_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            rgb_resized = rgb_img.copy()

        if rgb_resized.dtype == np.uint8 or rgb_resized.max() > 1.0:
            rgb_norm = rgb_resized.astype(np.float32) / 255.0
        else:
            rgb_norm = rgb_resized.astype(np.float32)

        # Sanitize depth input
        lr_depth_clean = lr_depth.copy()
        if lr_depth_clean.ndim > 2:
            lr_depth_clean = lr_depth_clean[:, :, 0]
        if float(np.max(lr_depth_clean)) > 100.0:
            lr_depth_clean = lr_depth_clean.astype(np.float32) / 1000.0

        lr_filled = fill_depth_holes(lr_depth_clean)
        up_depth = bicubic_upsample(lr_filled, target_shape)
        validity_mask = (up_depth > 0.05).astype(np.float32)

        rgb_tensor = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float()
        depth_tensor = torch.from_numpy(up_depth).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(validity_mask).unsqueeze(0).float()

        input_tensor = torch.cat([rgb_tensor, depth_tensor, mask_tensor], dim=0).unsqueeze(0)
        return input_tensor, up_depth, rgb_norm

    def predict_unet(self, input_tensor):
        self.model.eval()
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            pred = self.model(input_tensor)
            pred_np = pred.squeeze().cpu().numpy()
        return np.clip(pred_np, 0.0, 10.0)

    def reconstruct(self, rgb_img, lr_depth, target_shape=(256, 320),
                    guided_filter_radius=5, guided_filter_eps=1e-3,
                    apply_hybrid_refinement=True, use_edge_gating=True):
        input_tensor, up_depth, rgb_norm = self.preprocess_inputs(rgb_img, lr_depth, target_shape)

        # 1. Deep Learning U-Net prediction
        unet_depth = self.predict_unet(input_tensor)

        # 2. Hybrid Refinement
        gate = None
        if apply_hybrid_refinement:
            if use_edge_gating:
                refined_depth, gate = edge_gated_boundary_refinement(
                    unet_depth=unet_depth,
                    rgb_guide=rgb_norm,
                    radius=guided_filter_radius,
                    eps=guided_filter_eps
                )
            else:
                refined_depth = guided_filter_refinement(
                    depth_map=unet_depth,
                    rgb_guide=rgb_norm,
                    radius=guided_filter_radius,
                    eps=guided_filter_eps
                )
        else:
            refined_depth = unet_depth.copy()

        # 3. Discontinuity detection
        discontinuities, _ = detect_depth_discontinuities(refined_depth)

        return {
            'bicubic_baseline': up_depth,
            'unet_depth': unet_depth,
            'hybrid_refined': refined_depth,
            'discontinuities': discontinuities,
            'edge_gate': gate,
            'rgb_norm': rgb_norm
        }