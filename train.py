"""
Problem Statement 4 (ProbID - P4): Edge-Preserving Depth Map Upsampling and Refinement
End-to-End Training and Evaluation Script for NYU Depth V2 Dataset (Kaggle: soumikrakshit/nyu-depth-v2)
"""

import os
import sys
import time
import zipfile
import random
import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage.metrics import structural_similarity as ssim

# Import project modules
from models.unet import GuidedDepthUNet, DepthLoss
from classical.classical_methods import (
    nearest_upsample, bilinear_upsample, bicubic_upsample,
    guided_filter_refinement, fill_depth_holes, detect_depth_discontinuities
)
from evaluation.metrics import compute_depth_metrics


# ==========================================
# 1. Dataset Class with Synthetic Degradation
# ==========================================
class NYUDepthV2Dataset(Dataset):
    """
    Dataset loader for NYU Depth V2 (.mat format).
    Simulates multi-scale degradation (2x, 4x, 8x downsampling), sensor holes, and noise.
    Returns 5-channel input (RGB + Upsampled LR Depth + Validity Mask) and GT Depth.
    """
    def __init__(self, mat_file_path, indices, target_shape=(256, 320),
                 scales=[2, 4, 8], is_training=True):
        self.mat_file_path = mat_file_path
        self.indices = indices
        self.target_shape = target_shape
        self.scales = scales
        self.is_training = is_training
        self.h5_file = None

    def _open_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.mat_file_path, 'r')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._open_file()
        real_idx = self.indices[idx]
        
        # Load raw RGB and Depth from HDF5
        # NYU Depth V2 .mat dimensions:
        # 'images': (1449, 3, 640, 480) or (3, 640, 480, 1449)
        # 'depths': (1449, 640, 480) or (640, 480, 1449)
        images_ds = self.h5_file['images']
        depths_ds = self.h5_file['depths']

        if images_ds.shape[0] == 1449 or images_ds.shape[0] > images_ds.shape[-1]:
            raw_img = images_ds[real_idx] # (3, 640, 480)
            raw_depth = depths_ds[real_idx] # (640, 480)
        else:
            raw_img = images_ds[:, :, :, real_idx] # (3, 640, 480)
            raw_depth = depths_ds[:, :, real_idx] # (640, 480)

        # Transpose MATLAB dimensions to standard (H, W, C) -> (480, 640, 3)
        img_np = np.transpose(raw_img, (2, 1, 0)).astype(np.float32) / 255.0
        depth_np = raw_depth.T.astype(np.float32) # (480, 640)

        # Resize to target resolution (256, 320)
        target_h, target_w = self.target_shape
        img_resized = cv2.resize(img_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        depth_gt = cv2.resize(depth_np, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Multi-scale degradation factor
        if self.is_training:
            scale = random.choice(self.scales)
        else:
            scale = 4 # default validation scale factor

        # Downsample depth map to simulate low-resolution sensor
        lr_h, lr_w = target_h // scale, target_w // scale
        depth_lr = cv2.resize(depth_gt, (lr_w, lr_h), interpolation=cv2.INTER_NEAREST)

        # Simulate missing sensor dropouts / holes on 30% training samples
        if self.is_training and random.random() < 0.3:
            num_holes = random.randint(3, 7)
            for _ in range(num_holes):
                hh = random.randint(3, max(4, lr_h // 4))
                ww = random.randint(3, max(4, lr_w // 4))
                y0 = random.randint(0, lr_h - hh)
                x0 = random.randint(0, lr_w - ww)
                depth_lr[y0:y0+hh, x0:x0+ww] = 0.0

        # Simulate small depth sensor noise
        if self.is_training and random.random() < 0.5:
            noise = np.random.normal(0, 0.008, depth_lr.shape).astype(np.float32)
            depth_lr = np.maximum(depth_lr + noise, 0.0)

        # Upsample LR depth back to target shape using Bicubic interpolation
        lr_filled = fill_depth_holes(depth_lr)
        depth_up = cv2.resize(lr_filled, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        depth_up = np.maximum(depth_up, 0.0)

        # Validity mask (1 for valid depth pixels, 0 for missing)
        validity_mask = (depth_up > 0.05).astype(np.float32)

        # Convert to PyTorch tensors
        # RGB: (3, H, W)
        rgb_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float()
        # LR Depth: (1, H, W)
        depth_lr_tensor = torch.from_numpy(depth_up).unsqueeze(0).float()
        # Mask: (1, H, W)
        mask_tensor = torch.from_numpy(validity_mask).unsqueeze(0).float()
        # GT Depth: (1, H, W)
        gt_tensor = torch.from_numpy(depth_gt).unsqueeze(0).float()

        # 5-channel network input: [RGB (3), Upsampled LR Depth (1), Validity Mask (1)]
        input_tensor = torch.cat([rgb_tensor, depth_lr_tensor, mask_tensor], dim=0)

        return {
            'input': input_tensor,
            'gt': gt_tensor,
            'lr_up': depth_lr_tensor,
            'mask': mask_tensor,
            'rgb': rgb_tensor,
            'scale': scale
        }


# ==========================================
# 2. Training and Validation Functions
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_l1, total_grad = 0.0, 0.0, 0.0
    
    for batch in loader:
        inputs = batch['input'].to(device)
        gt = batch['gt'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()
        preds = model(inputs)
        loss, l1, grad = criterion(preds, gt, mask)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_l1 += l1.item()
        total_grad += grad.item()

    num_batches = len(loader)
    return total_loss / num_batches, total_l1 / num_batches, total_grad / num_batches


@torch.no_grad()
def evaluate_validation(model, loader, criterion, device):
    model.eval()
    total_loss, total_rmse, total_mae = 0.0, 0.0, 0.0
    
    for batch in loader:
        inputs = batch['input'].to(device)
        gt = batch['gt'].to(device)
        mask = batch['mask'].to(device)

        preds = model(inputs)
        loss, _, _ = criterion(preds, gt, mask)
        total_loss += loss.item()

        # Compute batch metrics on CPU
        preds_np = preds.squeeze(1).cpu().numpy()
        gts_np = gt.squeeze(1).cpu().numpy()

        for p, g in zip(preds_np, gts_np):
            valid = (g > 0.05)
            if np.sum(valid) > 0:
                diff = p[valid] - g[valid]
                total_rmse += np.sqrt(np.mean(diff ** 2))
                total_mae += np.mean(np.abs(diff))

    num_samples = len(loader.dataset)
    return total_loss / len(loader), total_rmse / num_samples, total_mae / num_samples


# ==========================================
# 3. Main Training Execution
# ==========================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Guided Depth U-Net on NYU Depth V2")
    parser.add_argument('--mat_file', type=str, default='nyu_depth_v2_labeled.mat', help="Path to nyu_depth_v2_labeled.mat")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--save_dir', type=str, default='models', help="Directory to save checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Training on device: {device}")

    # Inspect dataset
    if not os.path.exists(args.mat_file):
        print(f"[Error] Dataset file '{args.mat_file}' not found.")
        print("Please download it from Kaggle: soumikrakshit/nyu-depth-v2")
        return

    # Check total number of samples
    with h5py.File(args.mat_file, 'r') as f:
        total_samples = 1449
        print(f"[Info] Found {total_samples} aligned RGB-Depth pairs in {args.mat_file}")

    # 80/20 Train/Validation Split
    indices = list(range(total_samples))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.8 * total_samples)
    train_indices = indices[:split]
    val_indices = indices[split:]
    print(f"[Info] Train samples: {len(train_indices)}, Validation samples: {len(val_indices)}")

    # Data loaders
    train_dataset = NYUDepthV2Dataset(args.mat_file, train_indices, is_training=True)
    val_dataset = NYUDepthV2Dataset(args.mat_file, val_indices, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize model, loss and optimizer
    model = GuidedDepthUNet(in_channels=5, out_channels=1, base_filters=32, use_residual=True).to(device)
    criterion = DepthLoss(alpha_l1=1.0, alpha_grad=0.5).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    print(f"[Info] Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_val_rmse = float('inf')
    best_weights_path = os.path.join(args.save_dir, 'unet_depth_best.pth')

    train_losses, val_losses, val_rmses = [], [], []

    print("\n--- Starting Training ---")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        tr_loss, tr_l1, tr_grad = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmse, val_mae = evaluate_validation(model, val_loader, criterion, device)
        scheduler.step()

        ep_duration = time.time() - ep_start
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({ep_duration:.1f}s) | "
              f"Train Loss: {tr_loss:.4f} (L1: {tr_l1:.4f}, Grad: {tr_grad:.4f}) | "
              f"Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}m | Val MAE: {val_mae:.4f}m")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_weights_path)
            print(f"  >>> Checkpoint saved to {best_weights_path} (New Best RMSE: {best_val_rmse:.4f}m)")

    total_duration = (time.time() - start_time) / 60.0
    print(f"\n[Done] Training complete in {total_duration:.2f} minutes.")
    print(f"Best Validation RMSE: {best_val_rmse:.4f}m")

    # Save final model as well
    final_weights_path = os.path.join(args.save_dir, 'unet_depth_final.pth')
    torch.save(model.state_dict(), final_weights_path)


if __name__ == '__main__':
    main()
