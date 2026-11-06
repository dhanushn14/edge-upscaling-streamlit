# ☁️ 24/7 Live Cloud Deployment Guide (Zero Laptop Required)

This guide explains how to host the **Edge-Preserving Depth Reconstruction Web Application** online **24/7/365 for FREE** without needing your laptop to stay awake, open, or powered on.

---

## 🌟 Why Cloud Hosting is Required for 24/7 Uptime

- **Previous Setup (Gradio + Cloudflare Quick Tunnel)**:
  - Ran a local executable (`cloudflared.exe`) on your personal laptop.
  - As soon as your laptop went to sleep, closed its lid, or shut down, the tunnel was severed and the link crashed.
- **New Architecture (Streamlit 24/7 Cloud Architecture)**:
  - The entire application and neural network run on **Google / AWS Cloud servers**.
  - 100% Free, SSL/HTTPS secured, auto-restarts on any failure, zero power drawn from your personal machine.
  - You can completely shut down your laptop, and the link will still be online for evaluators, professors, judges, or mobile users worldwide.

---

## 🚀 Method 1: Streamlit Community Cloud (Recommended — 2 Minutes)

Streamlit Community Cloud is 100% free and provides direct 1-click continuous deployment from your GitHub account.

### Step 1: Initialize Git and Push to GitHub
Open your terminal in this project directory:
```bash
git add .
git commit -m "Deploy 24/7 Streamlit depth reconstruction app"
git branch -M main
```

Create a new repository on [GitHub](https://github.com/new) (e.g. `depth-upsampling-p4`) and push:
```bash
git remote add origin https://github.com/<YOUR-USERNAME>/depth-upsampling-p4.git
git push -u origin main
```

*(Note: The model weight file `models/unet_depth_best.pth` is only 13.97 MB, so it pushes smoothly without needing Git LFS).*

### Step 2: Deploy on Streamlit Cloud
1. Navigate to **[share.streamlit.io](https://share.streamlit.io)** and click **Sign In with GitHub**.
2. Click **Create app** -> **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository**: `<YOUR-USERNAME>/depth-upsampling-p4`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom subdomain (e.g. `depth-p4.streamlit.app`)
4. Click **Deploy!**

That's it! Your application will build in ~90 seconds and remain **permanently online 24/7**.

---

## 🚀 Method 2: Hugging Face Spaces (Alternative Free 24/7 Cloud)

Hugging Face Spaces offers **16GB RAM cloud CPU instances** permanently for free.

1. Navigate to **[huggingface.co/spaces](https://huggingface.co/spaces)** and click **Create new Space**.
2. Set:
   - **Space Name**: `edge-depth-upsampling-p4`
   - **Space SDK**: **Streamlit**
   - **License**: MIT
   - **Visibility**: Public
3. Push your repository files to the Hugging Face Git remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/<YOUR-HF-USERNAME>/edge-depth-upsampling-p4
   git push -u hf main
   ```
4. Hugging Face automatically detects `requirements.txt` and `packages.txt`, installs dependencies, and boots your app 24/7 at:
   `https://huggingface.co/spaces/<YOUR-HF-USERNAME>/edge-depth-upsampling-p4`

---

## 🚀 Method 3: Docker Deployment (Render / Railway / Cloud Run)

If you wish to deploy as a containerized web service:
1. Connect your GitHub repository to [Render.com](https://render.com) or [Railway.app](https://railway.app).
2. Select **Web Service** -> **Docker**.
3. Render / Railway will build the included `Dockerfile` and expose port `8501`.
4. Provides a permanent HTTPS URL with 24/7 background uptime.

---

## 💻 Testing Locally on Your Laptop

To test or develop locally before deploying to the cloud:
```bash
# 1. Activate your Python environment
python -m pip install -r requirements.txt

# 2. Launch Streamlit
streamlit run app.py
```
This opens the dashboard at `http://localhost:8501` in your default browser.
