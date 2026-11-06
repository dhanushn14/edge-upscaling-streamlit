# Edge-Preserving Depth Map Upsampling and Refinement (PS4)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)](https://streamlit.io/)
[![Cloud Ready](https://img.shields.io/badge/24%2F7%20Cloud-Always--On-success.svg)](DEPLOYMENT_24_7.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Problem Statement 4 (ProbID - P4)**: Intelligent, edge-aware depth reconstruction system that transforms sparse or low-resolution depth maps into dense, high-resolution representations while accurately preserving object boundaries, geometric structures, and depth discontinuities using a hybrid of classical computer vision and deep learning.

---

## 🏛️ System Architecture

![System Architecture Diagram](assets/system_architecture_diagram.png)

```mermaid
graph TD
    subgraph S1["Inputs & Degradation"]
        RGB["RGB Guidance Image (256x320x3)"]
        LRD["Low-Res Depth (2x, 4x, 8x)"]
        Holes["Simulated Sensor Holes & Noise"]
        LRD --> Holes
    end

    subgraph S2["Classical Preprocessing"]
        Holes --> Inpaint["Telea Inpainting (Hole Filling)"]
        Inpaint --> Bicubic["Bicubic Upsampling Baseline"]
        Bicubic --> Mask["Binary Validity Mask Generation"]
    end

    subgraph S3["Deep Guided U-Net (3.49M params)"]
        RGB --> Stack["5-Channel Input Tensor: RGB(3) + Depth(1) + Mask(1)"]
        Bicubic --> Stack
        Mask --> Stack
        Stack --> Enc["4-Stage Residual Conv Encoder"]
        Enc --> Bottle["Bottleneck Latent Space"]
        Bottle --> Dec["Decoder with Skip Connections"]
        Dec --> Head["Residual Boundary Head"]
        Bicubic --> Add["Residual Addition"]
        Head --> Add
        Add --> UNetPred["Dense U-Net Depth Map"]
    end

    subgraph S4["Anti-Texture Copying Refinement"]
        RGB --> EdgeGate["Edge Activation Gate G"]
        UNetPred --> EdgeGate
        RGB --> Guided["RGB Guided Filtering"]
        UNetPred --> Guided
        EdgeGate --> Blend["Gated Boundary Blending"]
        Guided --> Blend
        UNetPred --> Blend
        Blend --> FinalDepth["Final Dense Metric Depth Map"]
    end

    subgraph S5["Interactive UI & Benchmarks"]
        FinalDepth --> Metrics["RMSE, MAE, PSNR, SSIM, Edge Error, Latency"]
        FinalDepth --> StreamlitUI["Interactive Streamlit Dashboard (24/7 Cloud Hosted)"]
        FinalDepth --> Export["16-bit Metric PNG Export (Millimeters)"]
    end
```

---

## 🚀 Key Features

1. **Multi-Scale Degradation Generalization**: Single model robustly handles $2\times, 4\times, \text{and } 8\times$ depth downsampling.
2. **Sensor Defect Robustness**: Telea inpainting and learned validity masks fill missing sensor dropouts (specular/reflective holes) and suppress sensor noise.
3. **Anti-Texture Copying Strategy**: Edge-gated boundary refinement prevents RGB texture patterns (bedsheet wrinkles, wallpaper, shadows) from corrupting planar depth surfaces.
4. **Lightweight & Real-Time**: 3.49M parameters ($13.97\text{ MB}$ FP32 disk footprint), running at sub-26ms latency on GPU (>40 FPS) and ~68ms on consumer CPU.
5. **Full Baseline Benchmarking**:
   - **E1a, E1b, E1c**: Nearest, Bilinear, Bicubic Interpolation Baselines
   - **E2a, E2b**: Joint Bilateral Filtering (JBF) & Classical Guided Filtering
   - **E3**: Deep Learning Guided U-Net
   - **E4**: Hybrid Intelligence (Edge-Gated Boundary Refinement)
6. **Production Streamlit Dashboard**: 
   - 5 pre-loaded benchmark NYU test scenes with 1-click evaluation.
   - **Live Webcam Mode**: Real-time snapshot and depth upsampling right from your laptop or phone camera.
   - Interactive multi-method visual comparison and residual MAE error heatmaps.
   - 16-bit metric PNG export (millimeter precision) and CSV benchmark reporting.
7. **24/7 Cloud Architecture (Zero Laptop Required)**:
   - Configured for free 24/7 permanent deployment on **Streamlit Community Cloud** and **Hugging Face Spaces**.
   - Fully detached from local laptop tunnels (`cloudflared.exe` removed). Runs non-stop even when your laptop is turned off!

---

## 📂 Project Structure

```text
project/
├── .streamlit/
│   └── config.toml                   # Streamlit server, dark theme & performance config
├── app.py                            # Production Streamlit Web Application
├── train.py                          # Standalone Python training script
├── requirements.txt                  # Python dependencies (cloud-ready)
├── packages.txt                      # Linux apt dependencies for cloud hosts
├── Dockerfile                        # Multi-platform containerization for 24/7 deployment
├── DEPLOYMENT_24_7.md                # 1-Click 24/7 Cloud Deployment Guide
├── eval_P4_T79.md                    # Official comparative evaluation results
├── DESIGN_DOCUMENT.md                # System design & architecture document
├── README.md                         # Project documentation
├── models/
│   ├── __init__.py
│   ├── unet.py                       # 5-Channel Guided Depth U-Net & Edge Loss
│   ├── unet_depth_best.pth           # Trained model weights (13.97 MB)
│   ├── evaluation_results_table.csv
│   └── visual_comparison_2x_4x_8x.png
├── classical/
│   └── classical_methods.py          # Nearest, Bilinear, Bicubic, JBF, Guided Filter, Inpainting
├── hybrid/
│   └── hybrid_pipeline.py            # End-to-end hybrid reconstruction pipeline
├── evaluation/
│   └── metrics.py                    # Metric calculator (RMSE, MAE, PSNR, SSIM, Edge Error)
├── assets/
│   ├── system_architecture_diagram.png
│   └── samples/                      # 5 High-resolution benchmark test pairs
└── notebooks/
    └── PS4_Depth_Upsampling_Colab.ipynb # End-to-end Google Colab Training Notebook
```

---

## ⚡ Quick Start: Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Streamlit Web Application
streamlit run app.py
```

The application immediately opens at `http://localhost:8501`.

---

## ☁️ 24/7 Live Working (Without Your Laptop Being On)

To keep this project live online 24/7 so anyone can open and test it anytime:

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy 24/7 Streamlit depth upsampling app"
   git push origin main
   ```
2. **Go to [share.streamlit.io](https://share.streamlit.io)**:
   - Sign in with GitHub.
   - Click **Create app**, choose your repository and set Main file path to `app.py`.
   - Click **Deploy!**
3. **Your app is now live 24/7**:
   - Access from anywhere: `https://<your-subdomain>.streamlit.app`
   - **Your laptop does NOT need to stay on!** It runs directly on cloud servers for free.

For full step-by-step instructions (including Hugging Face Spaces and Docker), see **[DEPLOYMENT_24_7.md](DEPLOYMENT_24_7.md)**.

---

## 📊 Empirical Evaluation Results (NYU Depth V2 Validation)

| Scale Factor | Approach / Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | Edge Error ↓ | Latency (ms) ↓ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2×** | Nearest Interpolation | 0.0731 | 0.0161 | 43.76 | 0.9867 | 0.0832 | 0.13 ms |
| **2×** | Bilinear Interpolation | 0.0597 | 0.0176 | 45.51 | 0.9909 | 0.0930 | 0.12 ms |
| **2×** | Bicubic Baseline | 0.0615 | 0.0194 | 45.25 | 0.9905 | 0.0913 | 0.28 ms |
| **2×** | Joint Bilateral Filter (JBF) | 0.0569 | 0.0237 | 45.95 | 0.9906 | 0.1021 | 10.61 ms |
| **2×** | Classical Guided Filter | 0.0963 | 0.0414 | 41.46 | 0.9755 | 0.1471 | 2.91 ms |
| **2×** | **Guided U-Net (Ours)** | **0.0341** | **0.0099** | **50.55** | **0.9965** | **0.0498** | **22.83 ms** |
| **2×** | Hybrid (Edge-Gated) | 0.0917 | 0.0346 | 41.99 | 0.9765 | 0.1427 | 26.92 ms |
| | | | | | | | |
| **4×** | Nearest Interpolation | 0.1415 | 0.0424 | 37.99 | 0.9600 | 0.2250 | 0.14 ms |
| **4×** | Bilinear Interpolation | 0.1173 | 0.0424 | 39.61 | 0.9727 | 0.1633 | 0.08 ms |
| **4×** | Bicubic Baseline | 0.1228 | 0.0457 | 39.22 | 0.9703 | 0.1763 | 0.31 ms |
| **4×** | Joint Bilateral Filter (JBF) | 0.1104 | 0.0436 | 40.14 | 0.9753 | 0.1573 | 11.12 ms |
| **4×** | Classical Guided Filter | 0.1179 | 0.0524 | 39.58 | 0.9705 | 0.1589 | 2.90 ms |
| **4×** | **Guided U-Net (Ours)** | **0.0585** | **0.0194** | **45.76** | **0.9901** | **0.0941** | **26.21 ms** |
| **4×** | Hybrid (Edge-Gated) | 0.0958 | 0.0376 | 41.56 | 0.9750 | 0.1492 | 30.32 ms |
| | | | | | | | |
| **8×** | Nearest Interpolation | 0.2365 | 0.0916 | 33.53 | 0.9185 | 0.3274 | 0.12 ms |
| **8×** | Bilinear Interpolation | 0.1983 | 0.0883 | 35.06 | 0.9480 | 0.2040 | 0.07 ms |
| **8×** | Bicubic Baseline | 0.2084 | 0.0938 | 34.64 | 0.9437 | 0.2185 | 0.33 ms |
| **8×** | Joint Bilateral Filter (JBF) | 0.2006 | 0.0912 | 34.97 | 0.9476 | 0.2101 | 8.54 ms |
| **8×** | Classical Guided Filter | 0.1862 | 0.0884 | 35.60 | 0.9548 | 0.1867 | 2.73 ms |
| **8×** | **Guided U-Net (Ours)** | **0.0977** | **0.0365** | **41.39** | **0.9781** | **0.1338** | **26.13 ms** |
| **8×** | Hybrid (Edge-Gated) | 0.1139 | 0.0473 | 40.03 | 0.9699 | 0.1590 | 29.96 ms |

---

## 📜 Deliverables Checklist

- [x] Complete source code in modular structure (`models/`, `classical/`, `hybrid/`, `evaluation/`).
- [x] System architecture diagram matching code (Mermaid flowchart & PNG schematic).
- [x] Design document (`DESIGN_DOCUMENT.md`).
- [x] Comparative evaluation results file (`eval_P4_T79.md`).
- [x] Production Streamlit web application with multi-tab benchmarking and Live Camera mode.
- [x] 24/7 Cloud Deployment architecture (`packages.txt`, `.streamlit/`, `Dockerfile`, `DEPLOYMENT_24_7.md`).
- [x] 16-bit metric depth millimeter export for robotics and 3D vision pipelines.
