"""
app.py
======
Comprehensive Interactive Dashboard for:
Adaptive Watermarking for High-Resolution Images

Includes:
1. Multiple Watermarking Paradigms (Spatial LSB, DCT-Only, DWT-Only, DWT-DCT-SVD, and Proposed Adaptive HVS DWT-DCT)
2. Watermark Attack Protection & Security Suite:
   - Arnold Cat Map Scrambling (Protects against cropping & burst tampering)
   - Secret Key Permutation (Cryptographic extraction protection)
   - Spatial Redundancy & Majority Voting FEC
3. High-Resolution Spatial HVS Masking Heatmap Analyzer
4. Robustness Stress-Testing against 9 distinct distortion channels
5. Comparative Benchmark & Live Metric Radar/Bar graphs (PSNR, SSIM, NC, BER)
"""

import os
import sys

# Ensure local modules are found
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
    page_title="Advanced Watermarking & Defense Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Custom header styling
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 22px; border-radius: 10px; color: white; margin-bottom: 25px;">
    <h1 style="margin: 0; color: white;">🛡️ Multi-Scheme Watermarking & Attack Protection Dashboard</h1>
    <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1rem;">
        Digital Watermarking & Steganography | High-Resolution Image Domain
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================================
# SIDEBAR CONTROLS
# =========================================================================
st.sidebar.header("📁 1. High-Resolution Cover Image")

image_choice = st.sidebar.radio(
    "Cover Image Source:", 
    ["📤 Upload Your Own Image", "🖼️ Use 1080p Benchmark Image"]
)

if image_choice == "📤 Upload Your Own Image":
    uploaded = st.sidebar.file_uploader(
        "Upload High-Resolution Image", 
        type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"],
        help="Upload 1080p, 2K, 4K, or 8K images. High-resolution images benefit the most from HVS adaptive masking and tiling redundancy."
    )
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        active_image_name = uploaded.name
        file_size_mb = len(file_bytes) / (1024 * 1024)
        st.sidebar.caption(f"📁 **File:** `{uploaded.name}` ({file_size_mb:.2f} MB)")
    else:
        st.sidebar.info("💡 Upload an image or fallback to the sample 1080p image below.")
        active_image_name = "Synthetic_1080p_Cover.png"
        sample_path = os.path.join("results", "highres_test_1080p.png")
        if os.path.exists(sample_path):
            img_bgr = cv2.imread(sample_path)
        else:
            from experiment_runner import generate_synthetic_highres_image
            img_bgr = generate_synthetic_highres_image(1920, 1080)
            os.makedirs("results", exist_ok=True)
            cv2.imwrite(sample_path, img_bgr)
else:
    active_image_name = "Benchmark_1080p_Image.png"
    sample_path = os.path.join("results", "highres_test_1080p.png")
    if os.path.exists(sample_path):
        img_bgr = cv2.imread(sample_path)
    else:
        from experiment_runner import generate_synthetic_highres_image
        img_bgr = generate_synthetic_highres_image(1920, 1080)
        os.makedirs("results", exist_ok=True)
        cv2.imwrite(sample_path, img_bgr)

h, w = img_bgr.shape[:2]
mp = (w * h) / 1e6
st.sidebar.success(f"Resolution: **{w} × {h}** ({mp:.2f} Megapixels)")

# Optional downscaling safeguard for extreme ultra-high-res (e.g. > 16MP)
if mp > 16.0:
    st.sidebar.warning(f"⚠️ Extreme resolution detected ({mp:.1f} MP). Processing may take a few seconds.")

st.sidebar.markdown("---")
st.sidebar.header("🎯 2. Watermarking Technique")
scheme_choice = st.sidebar.selectbox(
    "Choose Watermarking Paradigm:",
    [
        "🌟 Proposed: Adaptive HVS (DWT-DCT)",
        "DWT-DCT (Fixed Baseline)",
        "Hybrid DWT-DCT-SVD",
        "Block-DCT Mid-Frequency",
        "DWT-Subband (Wavelet Details)",
        "Spatial Domain LSB"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("🔐 3. Attack Defense & Security Layer")
enable_arnold = st.sidebar.checkbox("Enable Arnold Cat Map Scrambling", value=True, help="Scrambles the 2D watermark pattern to protect against spatial burst attacks and localized cropping.")
arnold_rounds = st.sidebar.slider("Arnold Scramble Rounds", 1, 10, 4) if enable_arnold else 0

enable_key = st.sidebar.checkbox("Enable Secret Key Permutation", value=True, help="Permutes bit indices using a pseudo-random key to prevent unauthorized extraction.")
secret_key = st.sidebar.number_input("Cryptographic Secret Key", value=1337, step=1) if enable_key else 0

st.sidebar.markdown("---")
st.sidebar.header("🎛️ 4. Watermark Payload")

wm_input_type = st.sidebar.radio("Watermark Payload Type:", ["Text Signature", "Upload Custom Watermark Logo"])
wm_size = st.sidebar.selectbox("Watermark Matrix Size", [16, 32, 64], index=1)

if wm_input_type == "Text Signature":
    wm_text = st.sidebar.text_input("Signature Text", value="COPYRIGHT")
    raw_logo = generate_binary_logo(wm_text, size=wm_size)
else:
    wm_file = st.sidebar.file_uploader("Upload Watermark Logo (PNG/JPG)", type=["png", "jpg", "jpeg", "bmp"], key="wm_uploader")
    if wm_file is not None:
        wm_raw = cv2.imdecode(np.asarray(bytearray(wm_file.read()), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        resized_wm = cv2.resize(wm_raw, (wm_size, wm_size), interpolation=cv2.INTER_AREA)
        raw_logo = (resized_wm > 128).astype(np.uint8)
        wm_text = wm_file.name
    else:
        wm_text = "LOGO"
        raw_logo = generate_binary_logo("LOGO", size=wm_size)

# Apply Security Defense Pipeline
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

# =========================================================================
# EMBEDDING DISPATCHER
# =========================================================================
meta_dict = {}

if "Proposed: Adaptive HVS" in scheme_choice:
    engine = AdaptiveWatermarker(block_size=8, base_alpha=14.0, alpha_min=5.0, alpha_max=32.0)
    watermarked_img, meta_dict = engine.embed(img_bgr, embed_bits, adaptive=True)
elif "DWT-DCT (Fixed Baseline)" in scheme_choice:
    engine = AdaptiveWatermarker(block_size=8, base_alpha=12.0)
    watermarked_img, meta_dict = engine.embed(img_bgr, embed_bits, adaptive=False)
elif "Hybrid DWT-DCT-SVD" in scheme_choice:
    engine = DWT_DCT_SVD_Watermarker(alpha=16.0)
    watermarked_img, meta_dict = engine.embed(img_bgr, embed_bits)
elif "Block-DCT Mid-Frequency" in scheme_choice:
    engine = DCTBlockWatermarker(block_size=8, alpha=16.0)
    watermarked_img, meta_dict = engine.embed(img_bgr, embed_bits)
elif "DWT-Subband" in scheme_choice:
    engine = DWTSubbandWatermarker(alpha=15.0)
    watermarked_img, meta_dict = engine.embed(img_bgr, embed_bits)
else:
    engine = SpatialLSBWatermarker(bit_depth=1)
    watermarked_img, meta_dict = engine.embed(img_bgr, embed_bits)

psnr_score = compute_psnr(img_bgr, watermarked_img)
ssim_score = compute_ssim(img_bgr, watermarked_img)

# =========================================================================
# DASHBOARD TABS
# =========================================================================
t1, t2, t3, t4, t5 = st.tabs([
    "🛡️ 1. Embedding & Defense Pipeline", 
    "💥 2. Attack Simulation & Recovery", 
    "📈 3. Multi-Scheme Comparison", 
    "📊 4. Watermarking Log (Excel)",
    "📚 5. Project Guide & Viva Questions"
])

# TAB 1: EMBEDDING & DEFENSE
with t1:
    st.subheader("1. Visual Quality & Security Pre-processing")
    
    col_sec1, col_sec2, col_sec3 = st.columns(3)
    with col_sec1:
        st.markdown("**Original Clean Watermark**")
        st.image(raw_logo * 255, caption=f"Original Logo: '{wm_text}' ({wm_size}x{wm_size})", width=140, clamp=True)
    with col_sec2:
        st.markdown("**Arnold Cat Map Protected**")
        if enable_arnold:
            st.image(processed_logo * 255, caption=f"Scrambled ({arnold_rounds} rounds) - Resists Cropping", width=140, clamp=True)
        else:
            st.info("Arnold Scrambling Disabled")
    with col_sec3:
        st.markdown("**Key Permutation Security**")
        if enable_key:
            st.image(embed_bits.reshape((wm_size, wm_size)) * 255, caption=f"Permuted Bitstream (Key={secret_key})", width=140, clamp=True)
        else:
            st.info("Key Permutation Disabled")

    st.markdown("---")
    st.subheader(f"2. Watermarked Output — [{scheme_choice}]")
    
    c1, c2 = st.columns(2)
    with c1:
        st.image(cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB), caption="Watermarked High-Res Image", use_container_width=True)
        m_c1, m_c2 = st.columns(2)
        m_c1.metric("PSNR (Imperceptibility)", f"{psnr_score:.2f} dB", delta="Transparent (>38 dB)" if psnr_score > 38 else "Noticeable")
        m_c2.metric("SSIM (Structural Fidelity)", f"{ssim_score:.4f}", delta="Excellent (>0.98)" if ssim_score > 0.98 else "Degraded")
        
    with c2:
        if "alpha_map" in meta_dict:
            alpha_map = meta_dict["alpha_map"]
            norm_alpha = cv2.normalize(alpha_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap = cv2.applyColorMap(norm_alpha, cv2.COLORMAP_JET)
            st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), caption="HVS Adaptive α Heatmap (Red = High texture, Blue = Flat regions)", use_container_width=True)
            st.caption(f"Tiled {meta_dict.get('reps', 'N/A')} times across high-resolution blocks for high-redundancy majority voting.")
        else:
            diff = cv2.absdiff(img_bgr, watermarked_img) * 15
            st.image(diff, caption="Perturbation Difference Map (Amplified x15)", use_container_width=True)

# TAB 2: ATTACK SIMULATION & EXTRACTION
with t2:
    st.subheader("Simulate Hostile Attacks & Verify Protection Layer")
    
    col_atk_opt, col_atk_disp = st.columns([1, 2])
    with col_atk_opt:
        attack_type = st.selectbox(
            "Select Real-World Distortion Channel:",
            [
                "None (Clean Channel)",
                "JPEG Compression",
                "Gaussian Additive Noise",
                "Salt & Pepper Impulsive Noise",
                "Median Denoising Filter",
                "Gaussian Blur / Lowpass",
                "Cropping (Spatial Clipping)",
                "Downsample & Upsample Rescaling",
                "Brightness / Contrast Fluctuation"
            ]
        )
        
        attacked_img = watermarked_img.copy()
        if attack_type == "JPEG Compression":
            q_val = st.slider("JPEG Quality Factor", 10, 90, 40)
            attacked_img = attack_jpeg(watermarked_img, quality=q_val)
        elif attack_type == "Gaussian Additive Noise":
            v_val = st.slider("Noise Variance (σ²)", 5.0, 50.0, 20.0)
            attacked_img = attack_gaussian_noise(watermarked_img, var=v_val)
        elif attack_type == "Salt & Pepper Impulsive Noise":
            s_val = st.slider("Noise Density", 0.01, 0.10, 0.03, step=0.01)
            attacked_img = attack_salt_and_pepper(watermarked_img, amount=s_val)
        elif attack_type == "Median Denoising Filter":
            m_k = st.select_slider("Kernel Size", options=[3, 5, 7], value=3)
            attacked_img = attack_median_filter(watermarked_img, ksize=m_k)
        elif attack_type == "Gaussian Blur / Lowpass":
            b_k = st.select_slider("Blur Kernel Size", options=[3, 5, 7, 9], value=5)
            attacked_img = attack_gaussian_blur(watermarked_img, ksize=b_k)
        elif attack_type == "Cropping (Spatial Clipping)":
            c_p = st.slider("Cropping Ratio (Margin Loss %)", 0.05, 0.35, 0.20, step=0.05)
            attacked_img = attack_cropping(watermarked_img, crop_percent=c_p)
            st.caption("Notice: Arnold Cat Map distributes pixels so local crop doesn't erase contiguous watermark chunks!")
        elif attack_type == "Downsample & Upsample Rescaling":
            r_s = st.slider("Downscale Factor", 0.25, 0.75, 0.50, step=0.05)
            attacked_img = attack_resize(watermarked_img, scale=r_s)
        elif attack_type == "Brightness / Contrast Fluctuation":
            c_f = st.slider("Contrast Multiplier", 0.5, 1.5, 1.2, step=0.1)
            b_o = st.slider("Brightness Offset", -50, 50, 20)
            attacked_img = attack_brightness(watermarked_img, factor=c_f, beta=b_o)

        wrong_key = st.checkbox("Simulate Unauthorized Attacker (Wrong Secret Key)", value=False)

    with col_atk_disp:
        st.image(cv2.cvtColor(attacked_img, cv2.COLOR_BGR2RGB), caption=f"Distorted Image ({attack_type})", use_container_width=True)

    # Extraction Pipeline
    raw_extracted = engine.extract(attacked_img, total_bits=total_bits)
    
    # Decrypt Key Permutation
    if enable_key:
        if wrong_key:
            # Attacker uses wrong key
            bad_perm = np.random.default_rng(secret_key + 999).permutation(total_bits)
            unpermuted_bits = key_unpermute_bits(raw_extracted, bad_perm)
        else:
            unpermuted_bits = key_unpermute_bits(raw_extracted, perm_indices)
    else:
        unpermuted_bits = raw_extracted
        
    # Decrypt Arnold Cat Map
    unscrambled_logo = unpermuted_bits.reshape((wm_size, wm_size))
    if enable_arnold and not wrong_key:
        final_extracted_logo = arnold_cat_map_decrypt(unscrambled_logo, iterations=arnold_rounds)
    else:
        final_extracted_logo = unscrambled_logo

    final_extracted_bits = final_extracted_logo.flatten()
    nc_score = compute_nc(raw_logo.flatten(), final_extracted_bits)
    ber_score = compute_ber(raw_logo.flatten(), final_extracted_bits)

    st.markdown("---")
    st.subheader("Extraction & Security Verification Results")
    res_c1, res_c2, res_c3 = st.columns(3)
    with res_c1:
        st.markdown("**Original Watermark**")
        st.image(raw_logo * 255, width=140, clamp=True)
    with res_c2:
        st.markdown("**Recovered Watermark**")
        st.image(final_extracted_logo * 255, width=140, clamp=True)
    with res_c3:
        st.markdown("**Verification Metrics**")
        st.metric("Normalized Correlation (NC)", f"{nc_score:.4f}", delta="Valid (>= 0.90)" if nc_score >= 0.90 else "Compromised")
        st.metric("Bit Error Rate (BER)", f"{ber_score:.4f}")
        if wrong_key:
            st.error("🔒 Unauthorized attempt: Extraction without valid secret key produces pure noise!")

# TAB 3: MULTI-SCHEME BENCHMARKING
with t3:
    st.subheader("Cross-Scheme Performance Comparison")
    st.markdown("""
    Compare how different classical and modern watermarking paradigms behave under standard image transmission attacks.
    """)
    
    if st.button("🚀 Run Live Multi-Scheme Benchmark Test"):
        with st.spinner("Benchmarking 5 watermarking paradigms under attacks..."):
            test_attacks = {
                "JPEG (Q=40)": lambda img: attack_jpeg(img, quality=40),
                "Gaussian Noise (σ²=20)": lambda img: attack_gaussian_noise(img, var=20.0),
                "Salt & Pepper (3%)": lambda img: attack_salt_and_pepper(img, amount=0.03),
                "Border Crop (20%)": lambda img: attack_cropping(img, crop_percent=0.20),
                "Downscale 50%": lambda img: attack_resize(img, scale=0.50)
            }
            
            schemes_to_test = {
                "Spatial LSB": SpatialLSBWatermarker(bit_depth=1),
                "Block-DCT": DCTBlockWatermarker(block_size=8, alpha=16.0),
                "DWT-Subband": DWTSubbandWatermarker(alpha=15.0),
                "DWT-DCT-SVD": DWT_DCT_SVD_Watermarker(alpha=16.0),
                "Proposed Adaptive HVS": AdaptiveWatermarker(block_size=8, base_alpha=14.0, alpha_min=5.0, alpha_max=32.0)
            }
            
            summary_data = []
            for s_name, s_eng in schemes_to_test.items():
                w_img, _ = s_eng.embed(img_bgr, embed_bits, adaptive=True if "Adaptive" in s_name else False)
                p_val = compute_psnr(img_bgr, w_img)
                s_val = compute_ssim(img_bgr, w_img)
                
                row = {"Scheme": s_name, "PSNR (dB)": round(p_val, 2), "SSIM": round(s_val, 4)}
                for a_name, a_fn in test_attacks.items():
                    a_img = a_fn(w_img)
                    ext = s_eng.extract(a_img, total_bits)
                    if enable_key:
                        ext = key_unpermute_bits(ext, perm_indices)
                    if enable_arnold:
                        ext = arnold_cat_map_decrypt(ext.reshape((wm_size, wm_size)), iterations=arnold_rounds).flatten()
                    nc = compute_nc(raw_logo.flatten(), ext)
                    row[a_name] = round(nc, 3)
                summary_data.append(row)
                
            df_schemes = pd.DataFrame(summary_data)
            st.session_state["df_schemes"] = df_schemes

    if "df_schemes" in st.session_state:
        df_schemes = st.session_state["df_schemes"]
        st.dataframe(df_schemes.style.highlight_max(subset=["PSNR (dB)", "SSIM", "JPEG (Q=40)", "Gaussian Noise (σ²=20)", "Salt & Pepper (3%)", "Border Crop (20%)", "Downscale 50%"], color="#90EE90"), use_container_width=True)
        
        # Render Bar Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        df_melt = df_schemes.melt(id_vars=["Scheme"], value_vars=["JPEG (Q=40)", "Gaussian Noise (σ²=20)", "Salt & Pepper (3%)", "Border Crop (20%)"], var_name="Attack", value_name="NC")
        
        schemes = df_schemes["Scheme"].tolist()
        x = np.arange(len(schemes))
        width = 0.2
        
        for i, atk in enumerate(["JPEG (Q=40)", "Gaussian Noise (σ²=20)", "Salt & Pepper (3%)", "Border Crop (20%)"]):
            vals = df_schemes[atk].tolist()
            ax.bar(x + (i - 1.5)*width, vals, width, label=atk)
            
        ax.set_ylabel("Normalized Correlation (NC)")
        ax.set_title("Robustness Comparison Across Watermarking Paradigms")
        ax.set_xticks(x)
        ax.set_xticklabels(schemes, rotation=20, ha="right")
        ax.set_ylim([0.0, 1.1])
        ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='Acceptable Threshold (0.90)')
        ax.legend(loc='lower left', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

    # Show existing offline benchmark chart if available
    chart_file = os.path.join("results", "robustness_chart.png")
    if os.path.exists(chart_file):
        st.image(chart_file, caption="Precomputed Benchmark: Fixed Baseline vs. Proposed Adaptive Scheme", use_container_width=True)

# TAB 4: WATERMARKING LOG (EXCEL)
with t4:
    st.subheader("📊 Image Watermarking History & Operations Log")
    st.markdown("All high-resolution image uploads, watermarking schemes, parameters, and imperceptibility metrics are tracked in `watermarking_log.xlsx`.")
    
    col_log_act1, col_log_act2 = st.columns([1, 3])
    with col_log_act1:
        if st.button("💾 Log Current Watermarking Run to Excel"):
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
            st.success("Successfully logged run into `watermarking_log.xlsx`!")
            
    df_log = tracker.get_log_dataframe()
    if not df_log.empty:
        st.dataframe(df_log, use_container_width=True)
        
        # Download Excel file button
        if os.path.exists("watermarking_log.xlsx"):
            with open("watermarking_log.xlsx", "rb") as f:
                st.download_button(
                    label="📥 Download watermarking_log.xlsx",
                    data=f.read(),
                    file_name="watermarking_log.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.info("No operations logged yet. Click 'Log Current Watermarking Run to Excel' to save your first entry!")

# TAB 5: PROJECT GUIDE & VIVA QUESTIONS
with t5:
    st.subheader("🎓 Class Presentation & Viva Preparation Guide")
    st.markdown("""
    ### 1. Watermarking Paradigms Overview
    - **Spatial LSB**: Extremely simple, high PSNR, but **fragile** (destroyed by any compression or noise).
    - **Block-DCT**: Good energy compaction; resists JPEG compression by embedding in mid-frequency coefficients.
    - **DWT-Subband**: Multi-resolution frequency decomposition; separates smooth approximation ($LL$) from high-frequency edges ($LH, HL, HH$).
    - **Hybrid DWT-DCT-SVD**: Combines frequency transforms with Singular Value Decomposition (SVD); highly stable against geometric operations.
    - **Proposed Adaptive HVS DWT-DCT**: Dynamically adjusts embedding strength based on texture energy and luminance to maximize robustness in textures while safeguarding smoothness.

    ### 2. How the Watermark is Protected Against Attacks
    1. **Arnold Cat Map Scrambling**:
       - *Protection Against:* Burst errors and cropping attacks.
       - *Mechanism:* Scrambles pixel spatial correlation. If 25% of the host image is cropped or tampered, the damage is dispersed uniformly as mild random noise across the watermark rather than losing an entire character or logo segment.
    2. **Secret Key Permutation**:
       - *Protection Against:* Unauthorized readout and watermark stripping.
       - *Mechanism:* Only receivers with the exact seed key can reconstruct the correct bit order.
    3. **Spatial Redundancy & Majority-Voting Error Correction (FEC)**:
       - *Protection Against:* Random bit flips caused by heavy Gaussian / Salt & Pepper noise.
       - *Mechanism:* High-resolution images allow the 1024-bit watermark to be replicated across dozens of independent blocks. The receiver conducts majority voting per bit position.
    """)
