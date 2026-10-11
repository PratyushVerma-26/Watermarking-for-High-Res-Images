"""
app.py
======
Clean, Main-Screen Interface for:
Adaptive Watermarking for High-Resolution Images

Key Design Principles:
1. No top tab panels (removed the tab bar).
2. No sidebar controls - all configuration widgets are situated on the main screen.
3. Completely clean, academic, professional look with zero emojis/emoticons.
4. Direct Before (Original) and After (Watermarked / Attacked) image comparison.
5. Simple execution workflow with Excel logging support.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

import tracker
from watermarking import AdaptiveWatermarker, generate_binary_logo
from watermarking_schemes import (
    SpatialLSBWatermarker,
    DCTBlockWatermarker,
    DWTSubbandWatermarker,
    DWT_DCT_SVD_Watermarker
)
from watermark_security import (
    arnold_cat_map_encrypt,
    arnold_cat_map_decrypt,
    key_permute_bits,
    key_unpermute_bits
)
from attacks import (
    attack_jpeg,
    attack_gaussian_noise,
    attack_salt_and_pepper,
    attack_median_filter,
    attack_gaussian_blur,
    attack_cropping,
    attack_resize,
    attack_brightness
)
from metrics import compute_psnr, compute_ssim, compute_nc, compute_ber

st.set_page_config(
    page_title="Adaptive Watermarking for High-Resolution Images",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Header
st.title("Adaptive Watermarking for High-Resolution Images")
st.caption("Digital Watermarking and Steganography Research System")
st.markdown("---")

# =========================================================================
# SECTION 1: CONFIGURATION & CONTROLS (MAIN SCREEN)
# =========================================================================
st.subheader("Configuration and Parameters")

cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

with cfg_col1:
    st.markdown("**1. Image Source**")
    image_source_mode = st.radio(
        "Cover Image Selection:",
        ["Upload Image", "Generate Synthetic 1080p Image"],
        label_visibility="collapsed"
    )
    
    if image_source_mode == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload High-Resolution Image",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"]
        )
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            active_image_name = uploaded_file.name
        else:
            sample_path = os.path.join("results", "highres_test_1080p.png")
            if os.path.exists(sample_path):
                img_bgr = cv2.imread(sample_path)
            else:
                from experiment_runner import generate_synthetic_highres_image
                img_bgr = generate_synthetic_highres_image(1920, 1080)
                os.makedirs("results", exist_ok=True)
                cv2.imwrite(sample_path, img_bgr)
            active_image_name = "Sample_1080p_Image.png"
    else:
        sample_path = os.path.join("results", "highres_test_1080p.png")
        if os.path.exists(sample_path):
            img_bgr = cv2.imread(sample_path)
        else:
            from experiment_runner import generate_synthetic_highres_image
            img_bgr = generate_synthetic_highres_image(1920, 1080)
            os.makedirs("results", exist_ok=True)
            cv2.imwrite(sample_path, img_bgr)
        active_image_name = "Synthetic_1080p_Cover.png"

    h, w = img_bgr.shape[:2]
    mp = (w * h) / 1e6
    st.text(f"Dimensions: {w} x {h} ({mp:.2f} Megapixels)")

with cfg_col2:
    st.markdown("**2. Watermarking Scheme**")
    scheme_choice = st.selectbox(
        "Select Technique:",
        [
            "Adaptive HVS (DWT-DCT) [Proposed]",
            "DWT-DCT (Fixed Baseline)",
            "Hybrid DWT-DCT-SVD",
            "Block-DCT Mid-Frequency",
            "DWT-Subband",
            "Spatial Domain LSB"
        ]
    )
    
    wm_payload_type = st.radio("Payload Type:", ["Text Signature", "Upload Watermark Image"], horizontal=True)
    wm_size = st.selectbox("Watermark Size:", [16, 32, 64], index=1)
    
    if wm_payload_type == "Text Signature":
        wm_text = st.text_input("Watermark Text:", value="COPYRIGHT")
        raw_logo = generate_binary_logo(wm_text, size=wm_size)
    else:
        wm_file = st.file_uploader("Upload Watermark Logo (PNG/JPG)", type=["png", "jpg", "jpeg", "bmp"])
        if wm_file is not None:
            wm_raw = cv2.imdecode(np.asarray(bytearray(wm_file.read()), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            resized_wm = cv2.resize(wm_raw, (wm_size, wm_size), interpolation=cv2.INTER_AREA)
            raw_logo = (resized_wm > 128).astype(np.uint8)
            wm_text = wm_file.name
        else:
            wm_text = "LOGO"
            raw_logo = generate_binary_logo("LOGO", size=wm_size)

with cfg_col3:
    st.markdown("**3. Security and Defense**")
    enable_arnold = st.checkbox("Arnold Cat Map Scrambling (Resists Cropping)", value=True)
    arnold_rounds = st.slider("Scramble Rounds:", 1, 10, 4) if enable_arnold else 0
    
    enable_key = st.checkbox("Secret Key Bit Permutation", value=True)
    secret_key = st.number_input("Cryptographic Seed Key:", value=1337, step=1) if enable_key else 0
    
    st.markdown("**4. Optional Distortion Attack**")
    attack_choice = st.selectbox(
        "Apply Test Attack:",
        [
            "None (Clean)",
            "JPEG Compression (Quality 40)",
            "Gaussian Noise (Variance 20)",
            "Salt and Pepper Noise (3%)",
            "Median Filter (3x3)",
            "Gaussian Blur (5x5)",
            "Border Cropping (20%)",
            "Downsample/Upsample (50%)",
            "Brightness/Contrast Adjustment"
        ]
    )

st.markdown("---")

# =========================================================================
# SECTION 2: WATERMARK PROCESSING PIPELINE
# =========================================================================
# Prepare Watermark Payload
processed_logo = raw_logo.copy()
if enable_arnold:
    processed_logo = arnold_cat_map_encrypt(raw_logo, iterations=arnold_rounds)

raw_bits = processed_logo.flatten()
if enable_key:
    embed_bits, perm_indices = key_permute_bits(raw_bits, secret_key=secret_key)
else:
    embed_bits = raw_bits
    perm_indices = None

total_bits = len(embed_bits)

# Execute Embedding
meta_dict = {}
if "Adaptive HVS" in scheme_choice:
    engine = AdaptiveWatermarker(block_size=8, base_alpha=14.0, alpha_min=5.0, alpha_max=32.0)
    watermarked_img, meta_dict = engine.embed(img_bgr, embed_bits, adaptive=True)
elif "Fixed Baseline" in scheme_choice:
    engine = AdaptiveWatermarker(block_size=8, base_alpha=12.0)
    watermarked_img, meta_dict = engine.embed(img_bgr, embed_bits, adaptive=False)
elif "DWT-DCT-SVD" in scheme_choice:
    engine = DWT_DCT_SVD_Watermarker(alpha=16.0)
    watermarked_img, meta_dict = engine.embed(img_bgr, embed_bits)
elif "Block-DCT" in scheme_choice:
    engine = DCTBlockWatermarker(block_size=8, alpha=16.0)
    watermarked_img, meta_dict = engine.embed(img_bgr, embed_bits)
elif "DWT-Subband" in scheme_choice:
    engine = DWTSubbandWatermarker(alpha=15.0)
    watermarked_img, meta_dict = engine.embed(img_bgr, embed_bits)
else:
    engine = SpatialLSBWatermarker(bit_depth=1)
    watermarked_img, meta_dict = engine.embed(img_bgr, embed_bits)

# Compute Imperceptibility
psnr_score = compute_psnr(img_bgr, watermarked_img)
ssim_score = compute_ssim(img_bgr, watermarked_img)

# Apply Selected Attack (if any)
attacked_img = watermarked_img.copy()
if attack_choice == "JPEG Compression (Quality 40)":
    attacked_img = attack_jpeg(watermarked_img, quality=40)
elif attack_choice == "Gaussian Noise (Variance 20)":
    attacked_img = attack_gaussian_noise(watermarked_img, var=20.0)
elif attack_choice == "Salt and Pepper Noise (3%)":
    attacked_img = attack_salt_and_pepper(watermarked_img, amount=0.03)
elif attack_choice == "Median Filter (3x3)":
    attacked_img = attack_median_filter(watermarked_img, ksize=3)
elif attack_choice == "Gaussian Blur (5x5)":
    attacked_img = attack_gaussian_blur(watermarked_img, ksize=5)
elif attack_choice == "Border Cropping (20%)":
    attacked_img = attack_cropping(watermarked_img, crop_percent=0.20)
elif attack_choice == "Downsample/Upsample (50%)":
    attacked_img = attack_resize(watermarked_img, scale=0.50)
elif attack_choice == "Brightness/Contrast Adjustment":
    attacked_img = attack_brightness(watermarked_img, factor=1.2, beta=20)

# Extract Watermark
raw_extracted = engine.extract(attacked_img, total_bits=total_bits)

if enable_key:
    unpermuted_bits = key_unpermute_bits(raw_extracted, perm_indices)
else:
    unpermuted_bits = raw_extracted

unscrambled_logo = unpermuted_bits.reshape((wm_size, wm_size))
if enable_arnold:
    final_extracted_logo = arnold_cat_map_decrypt(unscrambled_logo, iterations=arnold_rounds)
else:
    final_extracted_logo = unscrambled_logo

final_extracted_bits = final_extracted_logo.flatten()
nc_score = compute_nc(raw_logo.flatten(), final_extracted_bits)
ber_score = compute_ber(raw_logo.flatten(), final_extracted_bits)

# =========================================================================
# SECTION 3: BEFORE AND AFTER COMPARISON
# =========================================================================
st.subheader("Image Comparison: Before and After Processing")

col_before, col_after = st.columns(2)

with col_before:
    st.markdown("**Before Processing (Original Image)**")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.caption(f"Original Resolution: {w} x {h} pixels")

with col_after:
    if attack_choice == "None (Clean)":
        st.markdown("**After Processing (Watermarked Image)**")
        st.image(cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("Clean Watermarked Image")
    else:
        st.markdown(f"**After Processing (Watermarked + {attack_choice})**")
        st.image(cv2.cvtColor(attacked_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(f"Distortion Applied: {attack_choice}")

# =========================================================================
# SECTION 4: METRICS & RECOVERED WATERMARK
# =========================================================================
st.markdown("---")
st.subheader("Quantitative Evaluation and Extracted Watermark")

res_col1, res_col2, res_col3, res_col4 = st.columns([1, 1, 1, 1])

with res_col1:
    st.metric("PSNR (Imperceptibility)", f"{psnr_score:.2f} dB")
    st.caption("Standard threshold: > 38.00 dB")

with res_col2:
    st.metric("SSIM (Structural Similarity)", f"{ssim_score:.4f}")
    st.caption("Standard threshold: > 0.9800")

with res_col3:
    st.metric("Normalized Correlation (NC)", f"{nc_score:.4f}")
    st.caption("Detection threshold: >= 0.9000")

with res_col4:
    st.metric("Bit Error Rate (BER)", f"{ber_score:.4f}")
    st.caption("Ideal: 0.0000")

st.markdown("---")

wm_disp1, wm_disp2, wm_disp3 = st.columns([1, 1, 2])

with wm_disp1:
    st.markdown("**Original Watermark**")
    st.image(raw_logo * 255, width=130, clamp=True)
    st.caption(f"Payload: {wm_text}")

with wm_disp2:
    st.markdown("**Extracted Watermark**")
    st.image(final_extracted_logo * 255, width=130, clamp=True)
    st.caption(f"Recovery NC: {nc_score:.4f}")

with wm_disp3:
    st.markdown("**Log to Excel (`watermarking_log.xlsx`)**")
    if st.button("Save Current Run to Excel"):
        out_name = f"watermarked_{active_image_name}"
        out_path = os.path.join("results", out_name)
        os.makedirs("results", exist_ok=True)
        cv2.imwrite(out_path, watermarked_img)
        
        tracker.log_watermark_operation(
            image_name=active_image_name,
            resolution=f"{w}x{h}",
            megapixels=mp,
            scheme=scheme_choice,
            watermark_payload=wm_text,
            arnold_scrambled=enable_arnold,
            key_permuted=enable_key,
            base_alpha=14.0 if "Adaptive" in scheme_choice else 16.0,
            psnr=psnr_score,
            ssim=ssim_score,
            watermarked_path=out_path
        )
        st.success("Record saved to watermarking_log.xlsx")
        
    df_history = tracker.get_log_dataframe()
    if not df_history.empty:
        st.caption(f"Total logged operations: {len(df_history)}")
        with open("watermarking_log.xlsx", "rb") as f:
            st.download_button(
                label="Download watermarking_log.xlsx",
                data=f.read(),
                file_name="watermarking_log.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
