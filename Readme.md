# RedDimension: Custom 3D Training Pipeline (AMD ROCm Optimized) 🔴📐

RedDimension is a comprehensive toolkit designed to train custom neural networks for 3D model generation from 2D images. This system is specifically architected and optimized for **AMD Radeon GPUs (RDNA 3/4)** using the ROCm stack.

---

## 🛠 Prerequisites & Environment Setup

### 1. Folder Structure
Create the following directory structure in the project root:
`dataset/raw`, `dataset/processed`, `dataset/images`.

### 2. Basic Installation
Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

---

## 🖥 Windows System Requirements (AMD GPUs)

To run this pipeline on Windows with **AMD 7000/9000 series** (RDNA 3/4), the following components are mandatory:

*   **HIP SDK:** Install [AMD HIP SDK 7.1.1](https://amd.com).
*   **ROCm Libraries:** Download essential ROCm 7.1.1 components from the official [Radeon Repository](https://radeon.com).
*   **Environment Variables:** Ensure the HIP binary paths are added to your system `PATH`.

*Project tested on RDNA 4 architecture (**GFX 12.0.1**) using `HSA_OVERRIDE`.*

---

## ⚠️ CRITICAL: Installation Order

For ROCm to initialize correctly on Windows, you **MUST** follow this specific sequence:

1.  **Python Libraries First:** Install all dependencies via `pip install -r requirements.txt`.
2.  **ROCm / HIP SDK Second:** Install the AMD HIP SDK and ROCm components **ONLY AFTER** the Python packages are installed.

**Why this matters:** The ROCm installer must register system paths and libraries after the Python environment is set up to avoid version conflicts with `hipblas.dll` and other core kernels.

---

## 📈 Training Workflow (Step-by-Step)

### 1. SDF Conversion
Place your `.glb` or `.obj` models into `dataset/raw/` and calculate the signed distance fields:
```bash
python step2_convert_to_sdf.py
```
*Note: This stage is extremely CPU-intensive and optimized for high-core-count multithreading.*

### 2. Multi-View Dataset Generation
Generate 2D projections of your models to train the computer vision encoder:
```bash
python step5_render_dataset.py
```
*The script automatically cleans up "orphaned" SDF files if rendering fails for a specific model.*

### 3. Core Training
Start the neural network training (Transformer 12 layers, 1024 dim):
```bash
python train_image_to_3d.py
```
*Features: 8-bit Adam optimizer and Mixed Precision (AMP) to minimize VRAM usage on AMD hardware.*

---

## 🚀 Inference & Usage

Once training is complete (recommended **Loss < 0.001**), launch the Gradio-based web interface for testing:
```bash
python app.py
```

---

## ⚙️ AMD Specific Tweaks
Each script includes pre-configured environment variables for **ROCm 7.x** stability on **GFX 12.0.1**. If you are using a different AMD architecture, please adjust `HSA_OVERRIDE_GFX_VERSION` in the source code.

## 📄 License
MIT

