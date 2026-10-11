# Adaptive Watermarking for High-Resolution Images

A comprehensive, research-grade mini-project developed for the **Digital Watermarking and Steganography** curriculum.

---

## 📌 Project Overview
High-resolution images (Full-HD, 2K, 4K, 8K) present unique challenges:
1. **Perceptual Heterogeneity**: They contain vast smooth areas (e.g., skies, walls) with low masking capability alongside densely textured areas (e.g., foliage, architecture). Uniform embedding causes noticeable degradation in smooth areas or insufficient robustness in textured areas.
2. **Abundant Transform Capacity**: The large pixel count allows watermark replication across thousands of blocks, enabling **Spatial Majority Voting** for Forward Error Correction (FEC).

This system provides:
* **HVS-Adaptive DWT-DCT Blind Watermarking Engine**: Uses local variance, Sobel edge magnitude, and Weber luminance masking to dynamically adjust embedding strength $\alpha(i, j)$ per $8 \times 8$ transform block.
* **Multiple Watermarking Paradigms**: Compare our proposed scheme against **Spatial LSB**, **Block-DCT**, **DWT-Subband**, and **Hybrid DWT-DCT-SVD**.
* **Watermark Attack Protection & Security**:
  * **Arnold Cat Map Scrambling**: Pixel decorrelation to withstand spatial burst attacks and heavy cropping.
  * **Cryptographic Secret Key Permutation**: Bit-level encryption preventing unauthorized detection.
  * **Spatial Redundancy & Majority-Voting FEC**: Mitigates random bit flips from lossy compression and noise.
* **Excel Operations & Upload Logger (`watermarking_log.xlsx`)**: Tracks every image uploaded, watermarking scheme used, parameters, and objective metrics (PSNR, SSIM, NC, BER).
* **Interactive Streamlit Web Dashboard**: Upload custom high-resolution images, visualize real-time HVS heatmaps, simulate attacks, and verify watermark extraction.

---

## 🚀 Quickstart & Starting Process

### Step 1: Clone the Repository
```bash
git clone https://github.com/PratyushVerma-26/Watermarking-for-High-Res-Images.git
cd Watermarking-for-High-Res-Images
```

### Step 2: Install Python Dependencies
Ensure you have Python 3.10+ installed:
```bash
pip install -r requirements.txt
```
*(Dependencies: `opencv-python`, `pywavelets`, `numpy`, `matplotlib`, `streamlit`, `pandas`, `openpyxl`, `scipy`)*

### Step 3: Run the Interactive Web Dashboard
Launch the web interface locally:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to:
1. Select **"📤 Upload Your Own Image"** to test your own high-resolution files (1080p, 2K, 4K, 8K).
2. Choose a watermarking scheme (e.g., *Proposed: Adaptive HVS DWT-DCT*).
3. Enable security protections (Arnold Cat Map Scrambling & Secret Key Permutation).
4. Inspect the **HVS Adaptive $\alpha$ Heatmap** and imperceptibility metrics (PSNR/SSIM).
5. Apply real-time attacks (JPEG compression, Gaussian noise, cropping, filtering) and observe blind watermark extraction.
6. Click **"Save to Excel"** to record operations in `watermarking_log.xlsx`.

### Step 4: Run the Automated Benchmark Suite (Optional)
To run automated comparative tests and generate publication-style charts:
```bash
python experiment_runner.py
```
Outputs generated in the `results/` folder:
* `benchmark_comparison.csv`: Numerical results across 11 attacks.
* `robustness_chart.png`: Visual bar chart comparing fixed vs. adaptive approaches.
* `hvs_alpha_heatmap.png`: Spatial color map of HVS embedding strength.

---

## 📁 Repository Structure
```
├── app.py                   # Full interactive Streamlit Web UI dashboard
├── watermarking.py          # Core DWT-DCT Adaptive & Fixed watermarking engine
├── watermarking_schemes.py  # Spatial LSB, Block-DCT, DWT-Subband, DWT-DCT-SVD
├── hvs_masking.py           # HVS texture & luminance masking algorithm
├── watermark_security.py    # Arnold Cat Map encryption & Key permutation
├── attacks.py               # Robustness attack simulation suite
├── metrics.py               # PSNR, SSIM, Normalized Correlation (NC), BER
├── tracker.py               # Excel logging engine for tracking uploads & metrics
├── experiment_runner.py     # Automated benchmark and plot generator
├── watermarking_log.xlsx    # Excel tracker of watermarking runs
├── requirements.txt         # Project dependencies
├── PROJECT_REPORT.md        # Academic report & Viva presentation guide
├── .gitignore               # Ignored cache and temporary artifacts
└── README.md                # Project documentation & start guide
```

---

## 📊 Experimental Results (1080p Image Benchmark)

| Evaluation Metric | Fixed-Strength Baseline | Proposed Adaptive HVS | Improvement |
| :--- | :--- | :--- | :--- |
| **PSNR (dB)** | 46.70 dB | 44.04 dB | Both well above the 38 dB threshold |
| **SSIM** | 0.9957 | 0.9864 | Preserves structural integrity |
| **JPEG Compression (Q=30)** | NC: 0.739 (BER: 7.6%) | **NC: 1.000 (BER: 0.0%)** | **+0.261 NC gain** |
| **Gaussian Noise ($\sigma^2=25$)** | NC: 0.642 (BER: 20.8%) | **NC: 0.969 (BER: 1.1%)** | **+0.327 NC gain** |
| **Salt & Pepper Noise (2%)** | NC: 0.725 (BER: 14.5%) | **NC: 0.983 (BER: 0.6%)** | **+0.258 NC gain** |
| **Median Filtering (3x3)** | NC: 0.983 | **NC: 1.000** | Perfect extraction |
| **Rescaling / Downsample (50%)** | NC: 1.000 | **NC: 1.000** | Full scale invariance |

---

## 🔐 Watermark Attack Protections
1. **Arnold Cat Map Pixel Scrambling**:
   $$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} \pmod N$$
   Scrambles watermark spatial continuity so localized cropping or burst noise is dispersed uniformly across the watermark as faint background noise, keeping characters legible.
2. **Key-Based Permutation**:
   Permutes the watermark bitstream using a cryptographic secret seed key. Unauthorized parties cannot recover the watermark without the key.
3. **Spatial Redundancy & Majority Voting**:
   Replicates the watermark payload across thousands of $8 \times 8$ blocks in high-resolution images, allowing bit reconstruction via voting even under heavy signal distortion.
