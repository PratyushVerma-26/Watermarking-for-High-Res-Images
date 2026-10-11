"""
hvs_masking.py
==============
Human Visual System (HVS) characteristics modeling:
Computes an adaptive masking map based on:
1. Luminance sensitivity (Weber's Law / Brightness masking)
2. Texture / edge energy (Spatial masking via Sobel / Laplacian variance)
3. Frequency sensitivity weights (Watson's CSF principles)

Returns block-by-block adaptive embedding strengths alpha(i, j).
"""

import numpy as np
import cv2

def compute_luminance_mask(block: np.ndarray) -> float:
    """
    Computes luminance masking weight.
    The human eye is less sensitive to modifications in very dark and very bright areas
    than in mid-gray areas (Weber-Fechner Law / U-shaped response curve).
    """
    mean_lum = np.mean(block)
    # Normalized distance from mid-gray (128)
    # Values close to 0 or 255 allow higher embedding strength.
    diff = abs(mean_lum - 128.0) / 128.0
    lum_factor = 1.0 + 0.8 * (diff ** 2)
    return float(lum_factor)


def compute_texture_mask(block: np.ndarray) -> float:
    """
    Computes texture/edge masking weight.
    High-texture regions have high spatial frequency variance and edge energy,
    masking high-frequency changes effectively from human perception.
    Smooth/flat regions have low variance and artifacts are immediately noticeable.
    """
    # Gradient magnitude via Sobel
    grad_x = cv2.Sobel(block.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(block.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = np.mean(np.sqrt(grad_x**2 + grad_y**2))
    
    # Local variance
    variance = np.var(block)
    
    # Combined texture score
    texture_score = np.log1p(edge_energy + np.sqrt(variance))
    return float(texture_score)


def compute_adaptive_alpha_map(
    img_gray: np.ndarray, 
    block_size: int = 8, 
    base_alpha: float = 12.0, 
    alpha_min: float = 5.0, 
    alpha_max: float = 35.0
) -> np.ndarray:
    """
    Computes a 2D matrix of adaptive embedding strengths alpha(row_blk, col_blk)
    for each block in the high-resolution image.
    
    Parameters:
    - img_gray: Grayscale 2D image (uint8 or float)
    - block_size: Block size (typically 8x8)
    - base_alpha: Baseline embedding strength
    - alpha_min: Minimum embedding strength (applied in flat/sensitive regions to protect transparency)
    - alpha_max: Maximum embedding strength (applied in rich texture regions for maximum robustness)
    
    Returns:
    - alpha_map: 2D numpy array of shape (num_blocks_y, num_blocks_x)
    """
    h, w = img_gray.shape
    num_blocks_y = h // block_size
    num_blocks_x = w // block_size
    
    texture_scores = np.zeros((num_blocks_y, num_blocks_x), dtype=np.float32)
    lum_factors = np.zeros((num_blocks_y, num_blocks_x), dtype=np.float32)
    
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            block = img_gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            lum_factors[i, j] = compute_luminance_mask(block)
            texture_scores[i, j] = compute_texture_mask(block)
            
    # Normalize texture scores to [0, 1] across the image
    t_min = np.min(texture_scores)
    t_max = np.max(texture_scores)
    if t_max > t_min:
        norm_texture = (texture_scores - t_min) / (t_max - t_min)
    else:
        norm_texture = np.zeros_like(texture_scores)
        
    # Combine texture and luminance:
    # High texture -> high alpha
    # Extreme dark/bright -> slight boost
    raw_alpha = base_alpha * (1.0 + 1.5 * norm_texture) * (0.8 + 0.4 * lum_factors)
    alpha_map = np.clip(raw_alpha, alpha_min, alpha_max)
    
    return alpha_map
