"""
metrics.py
==========
Objective evaluation metrics for image watermarking:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- NC (Normalized Correlation / Bit Correlation)
- BER (Bit Error Rate)
"""

import numpy as np
import cv2

def compute_psnr(original: np.ndarray, watermarked: np.ndarray) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in dB between two images.
    Values > 38-40 dB are considered visually transparent.
    """
    if original.shape != watermarked.shape:
        raise ValueError("Images must have identical dimensions.")
    
    orig_f = original.astype(np.float64)
    wm_f = watermarked.astype(np.float64)
    mse = np.mean((orig_f - wm_f) ** 2)
    if mse == 0:
        return float('inf')
    max_val = 255.0
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes Structural Similarity Index (SSIM) between two grayscale images or luminance channels.
    """
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def compute_nc(orig_bits: np.ndarray, ext_bits: np.ndarray) -> float:
    """
    Normalized Correlation (NC) between original binary watermark bits and extracted bits.
    Range: [-1.0, 1.0], 1.0 means exact match.
    """
    orig_b = orig_bits.flatten().astype(np.float64)
    ext_b = ext_bits.flatten().astype(np.float64)
    
    denom = np.sqrt(np.sum(orig_b ** 2) * np.sum(ext_b ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(orig_b * ext_b) / denom)


def compute_ber(orig_bits: np.ndarray, ext_bits: np.ndarray) -> float:
    """
    Bit Error Rate (BER) between original binary watermark and extracted watermark.
    Range: [0.0, 1.0], 0.0 means 0% error.
    """
    orig_b = (orig_bits.flatten() > 0.5).astype(int)
    ext_b = (ext_bits.flatten() > 0.5).astype(int)
    bit_errors = np.sum(orig_b != ext_b)
    return float(bit_errors / len(orig_b))
