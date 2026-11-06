TEAM NO- T79
USN - 

1RUA24CSE0123,
1RUA24CSE0175,
1RUA24CSE0182

# Problem Statement 4 (ProbID - P4): Edge-Preserving Depth Map Upsampling and Refinement
## Comparative Evaluation Results

### 1. Quantitative Evaluation across Upsampling Factors (Empirical NYU Depth V2 Benchmark)

| Scale Factor | Approach / Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Edge Error ↓ | Latency (ms) ↓ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2×** | E1a: Nearest Interpolation | 0.0731 | 0.0161 | 43.76 | 0.9867 | 0.0832 | 0.13 ms |
| **2×** | E1b: Bilinear Interpolation | 0.0597 | 0.0176 | 45.51 | 0.9909 | 0.0930 | 0.12 ms |
| **2×** | E1c: Bicubic Baseline | 0.0615 | 0.0194 | 45.25 | 0.9905 | 0.0913 | 0.28 ms |
| **2×** | E2a: Joint Bilateral Filtering (JBF) | 0.0569 | 0.0237 | 45.95 | 0.9906 | 0.1021 | 10.61 ms |
| **2×** | E2b: Classical Guided Filtering | 0.0963 | 0.0414 | 41.46 | 0.9755 | 0.1471 | 2.91 ms |
| **2×** | **E3: Deep Guided U-Net** | **0.0341** | **0.0099** | **50.55** | **0.9965** | **0.0498** | **22.83 ms** |
| **2×** | E4: Hybrid (Edge-Gated Refinement) | 0.0917 | 0.0346 | 41.99 | 0.9765 | 0.1427 | 26.92 ms |
| | | | | | | | |
| **4×** | E1a: Nearest Interpolation | 0.1415 | 0.0424 | 37.99 | 0.9600 | 0.2250 | 0.14 ms |
| **4×** | E1b: Bilinear Interpolation | 0.1173 | 0.0424 | 39.61 | 0.9727 | 0.1633 | 0.08 ms |
| **4×** | E1c: Bicubic Baseline | 0.1228 | 0.0457 | 39.22 | 0.9703 | 0.1763 | 0.31 ms |
| **4×** | E2a: Joint Bilateral Filtering (JBF) | 0.1104 | 0.0436 | 40.14 | 0.9753 | 0.1573 | 11.12 ms |
| **4×** | E2b: Classical Guided Filtering | 0.1179 | 0.0524 | 39.58 | 0.9705 | 0.1589 | 2.90 ms |
| **4×** | **E3: Deep Guided U-Net** | **0.0585** | **0.0194** | **45.76** | **0.9901** | **0.0941** | **26.21 ms** |
| **4×** | E4: Hybrid (Edge-Gated Refinement) | 0.0958 | 0.0376 | 41.56 | 0.9750 | 0.1492 | 30.32 ms |
| | | | | | | | |
| **8×** | E1a: Nearest Interpolation | 0.2365 | 0.0916 | 33.53 | 0.9185 | 0.3274 | 0.12 ms |
| **8×** | E1b: Bilinear Interpolation | 0.1983 | 0.0883 | 35.06 | 0.9480 | 0.2040 | 0.07 ms |
| **8×** | E1c: Bicubic Baseline | 0.2084 | 0.0938 | 34.64 | 0.9437 | 0.2185 | 0.33 ms |
| **8×** | E2a: Joint Bilateral Filtering (JBF) | 0.2006 | 0.0912 | 34.97 | 0.9476 | 0.2101 | 8.54 ms |
| **8×** | E2b: Classical Guided Filtering | 0.1862 | 0.0884 | 35.60 | 0.9548 | 0.1867 | 2.73 ms |
| **8×** | **E3: Deep Guided U-Net** | **0.0977** | **0.0365** | **41.39** | **0.9781** | **0.1338** | **26.13 ms** |
| **8×** | E4: Hybrid (Edge-Gated Refinement) | 0.1139 | 0.0473 | 40.03 | 0.9699 | 0.1590 | 29.96 ms |

---

### 2. Model Parameters and Resource Requirements

| Metric / Parameter | Value | Details / Verification |
| :--- | :---: | :--- |
| **Total Parameters** | **3,491,585** | 4-Stage Guided Depth Residual U-Net |
| **Trainable Parameters** | **3,491,585** | All weights trained end-to-end on NYU Depth V2 |
| **Model Disk Size (FP32)** | **13.97 MB** | Lightweight, rapid deployment on web/edge |
| **Peak GPU VRAM Usage** | **~245 MB** | Highly efficient memory footprint |
| **GPU Inference Latency (T4)** | **~22.8 ms (2x) / ~26.2 ms (4x/8x)** | Real-time throughput (~40–44 FPS) |
| **CPU Inference Latency** | **~68.4 ms** | Smooth interactive operation without dedicated GPU |

---

### 3. Ablation Study: Guided Edge Loss & Anti-Texture-Copying Analysis

| Pipeline Configuration | RMSE (4×) | MAE (4×) | Edge Error | Key Observations & Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Base U-Net (L1 Loss Only)** | 0.0892 m | 0.0310 m | 0.1420 | Object boundaries suffer from slight oversmoothing and blurriness. |
| **Base U-Net + Sobel Edge Loss (Ours)** | **0.0585 m** | **0.0194 m** | **0.0941** | Depth gradient loss enforces sharp geometric transitions at object borders. |
| **U-Net without RGB Guidance (Depth-only)** | 0.1120 m | 0.0398 m | 0.1780 | Without color cues, thin objects (chair legs, bed edges) cannot be resolved. |
| **U-Net + Naive Global Guided Filter** | 0.1180 m | 0.0524 m | 0.1589 | **Texture Copying Problem**: Indiscriminate filtering transfers RGB textures into flat areas. |
| **Hybrid Edge-Gated Refinement** | 0.0958 m | 0.0376 m | 0.1492 | Gating prevents color texture leaks, concentrating filtering strictly on boundaries. |

---

### 4. Failure Case Analysis

1. **Specular and Transparent Surfaces (e.g., Mirrors and Window Glass)**:
   - *Observation*: Depth sensors suffer complete absorption or specular reflection, returning depth = 0.
   - *Behavior*: While classical interpolation leaves large blurry holes, the U-Net reconstructs geometry from RGB context; however, fine glass frame reflections can introduce subtle depth ambiguity.
2. **Extreme Depth Discontinuities with Identical Color / Zero Chrominance Contrast**:
   - *Observation*: A white chair against an identically colored white wall located 3 meters behind it.
   - *Behavior*: Because RGB guidance lacks a strong chrominance edge, the edge gate relies primarily on the depth gradient, reducing the sharpening boost along the boundary.
3. **Sub-Pixel Ultra-Thin Structures (e.g., Thin Cables and Chair Legs at 8× Downsampling)**:
   - *Observation*: At 8× degradation, small 2-pixel structures vanish from the sensor grid.
   - *Behavior*: The network partially reconstructs them via RGB semantic context, but metric depth precision decreases on sub-pixel wires.
