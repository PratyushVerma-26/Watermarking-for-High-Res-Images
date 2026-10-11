"""
experiment_runner.py
====================
Runs automated comparative benchmarking experiments between:
1. Baseline Fixed-Strength Watermarking (Standard DWT-DCT)
2. Adaptive HVS-Masked Watermarking (Our Method)

Across high-resolution test images and standard attacks.
Computes PSNR, SSIM, NC, BER and produces summary CSV and plots.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from watermarking import AdaptiveWatermarker, generate_binary_logo
from hvs_masking import compute_adaptive_alpha_map
from attacks import get_all_attacks
from metrics import compute_psnr, compute_ssim, compute_nc, compute_ber

def generate_synthetic_highres_image(width: int = 1920, height: int = 1080) -> np.ndarray:
    """
    Generates a realistic synthetic Full-HD image containing both:
    - Smooth gradient skies (flat regions sensitive to noise)
    - High-frequency procedural textures, geometric shapes, and edges.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 1. Smooth gradient sky
    for y in range(height // 2):
        color = [int(220 - y * 0.15), int(180 - y * 0.1), int(120 + y * 0.1)]
        img[y, :] = color
        
    # 2. Textured lower landscape
    np.random.seed(42)
    noise = np.random.randint(0, 50, (height // 2, width, 3), dtype=np.uint8)
    base_land = np.zeros((height // 2, width, 3), dtype=np.uint8)
    base_land[:, :] = [34, 139, 34] # Forest green
    img[height // 2:, :] = cv2.add(base_land, noise)
    
    # 3. High-contrast geometric shapes & text
    cv2.circle(img, (width // 4, height // 3), 120, (0, 215, 255), -1) # Golden sun
    cv2.rectangle(img, (width // 2, height // 3), (width // 2 + 300, height // 2 + 200), (40, 40, 180), -1)
    
    # Add high-texture grid lines
    for x in range(0, width, 60):
        cv2.line(img, (x, height // 2), (x + 30, height), (80, 80, 80), 2)
        
    cv2.putText(img, "HIGH-RES TEST BENCHMARK (1080p)", (width // 3, height // 8), 
                cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
                
    return img


def run_benchmark():
    os.makedirs("results", exist_ok=True)
    
    print("=" * 70)
    print(" ADAPTIVE WATERMARKING FOR HIGH-RESOLUTION IMAGES: BENCHMARK RUNNER")
    print("=" * 70)
    
    # Generate test image
    image_path = os.path.join("results", "highres_test_1080p.png")
    if not os.path.exists(image_path):
        print("Generating 1920x1080 high-resolution test image...")
        img = generate_synthetic_highres_image(1920, 1080)
        cv2.imwrite(image_path, img)
    else:
        img = cv2.imread(image_path)
        
    # Generate watermark (32x32 binary logo -> 1024 bits)
    wm_logo = generate_binary_logo(text="SECURE", size=32)
    wm_bits = wm_logo.flatten()
    total_bits = len(wm_bits)
    cv2.imwrite(os.path.join("results", "original_watermark.png"), wm_logo * 255)
    
    watermarker = AdaptiveWatermarker(block_size=8, base_alpha=12.0, alpha_min=5.0, alpha_max=28.0)
    
    # Embed Watermarks
    print("\n[1/3] Embedding Watermark...")
    print(" -> Embedding with Baseline (Fixed alpha = 12.0)...")
    fixed_wm_img, meta_fixed = watermarker.embed(img, wm_bits, adaptive=False)
    
    print(" -> Embedding with Adaptive HVS Masking...")
    adapt_wm_img, meta_adapt = watermarker.embed(img, wm_bits, adaptive=True)
    
    # Save watermarked images
    cv2.imwrite(os.path.join("results", "watermarked_fixed.png"), fixed_wm_img)
    cv2.imwrite(os.path.join("results", "watermarked_adaptive.png"), adapt_wm_img)
    
    # Imperceptibility Metrics
    psnr_fixed = compute_psnr(img, fixed_wm_img)
    ssim_fixed = compute_ssim(img, fixed_wm_img)
    
    psnr_adapt = compute_psnr(img, adapt_wm_img)
    ssim_adapt = compute_ssim(img, adapt_wm_img)
    
    print(f"\n--- Transparency Comparison (Imperceptibility) ---")
    print(f"Baseline (Fixed)   -> PSNR: {psnr_fixed:.2f} dB | SSIM: {ssim_fixed:.4f}")
    print(f"Adaptive (HVS)     -> PSNR: {psnr_adapt:.2f} dB | SSIM: {ssim_adapt:.4f}")
    
    # Save HVS Mask Visualization
    alpha_map = meta_adapt["alpha_map"]
    norm_alpha = cv2.normalize(alpha_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_alpha, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join("results", "hvs_alpha_heatmap.png"), heatmap)
    
    # Run Attack Suite
    print("\n[2/3] Evaluating Robustness across Signal Attacks...")
    attacks = get_all_attacks()
    
    records = []
    
    for attack_name, attack_fn in attacks.items():
        # Attack fixed watermarked image
        att_fixed = attack_fn(fixed_wm_img)
        ext_fixed = watermarker.extract(att_fixed, total_bits)
        nc_fixed = compute_nc(wm_bits, ext_fixed)
        ber_fixed = compute_ber(wm_bits, ext_fixed)
        
        # Attack adaptive watermarked image
        att_adapt = attack_fn(adapt_wm_img)
        ext_adapt = watermarker.extract(att_adapt, total_bits)
        nc_adapt = compute_nc(wm_bits, ext_adapt)
        ber_adapt = compute_ber(wm_bits, ext_adapt)
        
        records.append({
            "Attack": attack_name,
            "Fixed_NC": round(nc_fixed, 4),
            "Adaptive_NC": round(nc_adapt, 4),
            "Fixed_BER": round(ber_fixed, 4),
            "Adaptive_BER": round(ber_adapt, 4),
            "NC_Improvement": round(nc_adapt - nc_fixed, 4)
        })
        print(f"  {attack_name:<25} | Fixed NC: {nc_fixed:.3f} (BER {ber_fixed:.3f}) | Adapt NC: {nc_adapt:.3f} (BER {ber_adapt:.3f})")
        
    df = pd.DataFrame(records)
    csv_path = os.path.join("results", "benchmark_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[3/3] Results saved to: {csv_path}")
    
    # Generate Matplotlib Comparison Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df["Attack"]))
    width = 0.35
    
    ax.bar(x - width/2, df["Fixed_NC"], width, label='Fixed Strength (Baseline)', color='#4A90E2')
    ax.bar(x + width/2, df["Adaptive_NC"], width, label='Adaptive HVS (Proposed)', color='#50E3C2')
    
    ax.set_ylabel('Normalized Correlation (NC)', fontsize=12)
    ax.set_title('Robustness Comparison: Fixed vs. Adaptive Watermarking on 1080p Image', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df["Attack"], rotation=35, ha='right', fontsize=10)
    ax.set_ylim([0.0, 1.08])
    ax.axhline(0.9, color='red', linestyle='--', alpha=0.6, label='Ideal Detection Threshold (0.90)')
    ax.legend(loc='lower left', frameon=True)
    plt.tight_layout()
    chart_path = os.path.join("results", "robustness_chart.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
    print(f"Comparison chart exported to: {chart_path}")
    print("=" * 70)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark()
