"""
watermarking.py
===============
Core Adaptive Watermarking Engine for High-Resolution Images.

Features:
1. Multi-scale block-based transform embedding:
   - DWT (Discrete Wavelet Transform, Haar or Daubechies) + Block-DCT (Discrete Cosine Transform)
2. HVS (Human Visual System) Adaptive Masking:
   - Modulates embedding strength dynamically per 8x8 block using texture variance and luminance.
3. High-Resolution Optimization:
   - High-capacity payload replication / majority-voting error correction across tiles.
   - Preserves edge fidelity while maintaining robustness against compression and filtering.
4. Baseline Comparison:
   - Non-adaptive uniform embedding mode for direct comparative experimental analysis.
"""

import numpy as np
import cv2
import pywt
from hvs_masking import compute_adaptive_alpha_map

# Recommended mid-frequency DCT coefficients for 8x8 block
# (4, 3) and (3, 4) or (2, 3) and (3, 2)
COEFF_A = (3, 2)
COEFF_B = (2, 3)

def generate_binary_logo(text: str = "COPYRIGHT", size: int = 32) -> np.ndarray:
    """Generates a binary watermark image (0 and 1) containing text."""
    canvas = np.zeros((size, size), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = size / 48.0
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    tx = max(0, (size - tw) // 2)
    ty = min(size - 2, (size + th) // 2)
    cv2.putText(canvas, text, (tx, ty), font, scale, 255, thickness, cv2.LINE_AA)
    binary_logo = (canvas > 120).astype(np.uint8)
    return binary_logo


class AdaptiveWatermarker:
    def __init__(self, block_size: int = 8, base_alpha: float = 12.0, alpha_min: float = 5.0, alpha_max: float = 30.0):
        self.block_size = block_size
        self.base_alpha = base_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def embed(
        self, 
        image: np.ndarray, 
        watermark_bits: np.ndarray, 
        adaptive: bool = True
    ) -> tuple[np.ndarray, dict]:
        """
        Embeds watermark_bits into image using DWT-DCT with HVS-adaptive strength.
        
        Parameters:
        - image: Input RGB or Grayscale image (uint8)
        - watermark_bits: 1D or 2D binary numpy array (0s and 1s)
        - adaptive: If True, uses HVS texture-luminance masking. If False, uses fixed base_alpha.
        
        Returns:
        - watermarked_image: Same shape and type as input
        - metadata: Dictionary containing embedding parameters, block counts, and alpha map.
        """
        is_color = (len(image.shape) == 3)
        if is_color:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y_channel = ycrcb[:, :, 0].copy()
        else:
            y_channel = image.copy()
            
        h, w = y_channel.shape
        # Pad slightly to ensure even dimensions for DWT and block multiples
        target_h = int(np.ceil(h / (self.block_size * 2)) * (self.block_size * 2))
        target_w = int(np.ceil(w / (self.block_size * 2)) * (self.block_size * 2))
        
        pad_y = np.pad(y_channel, ((0, target_h - h), (0, target_w - w)), mode='reflect').astype(np.float32)
        
        # 1. 1-Level 2D Discrete Wavelet Transform
        coeffs2 = pywt.dwt2(pad_y, 'haar')
        LL, (LH, HL, HH) = coeffs2
        
        # We embed into the LL band (robust against compression) or HL/LH band.
        # Embedding into LL with DCT provides state-of-the-art robustness against aggressive attacks.
        subband = LL.copy()
        sb_h, sb_w = subband.shape
        
        # Compute HVS Alpha Map on subband
        if adaptive:
            alpha_map = compute_adaptive_alpha_map(
                subband, 
                block_size=self.block_size, 
                base_alpha=self.base_alpha,
                alpha_min=self.alpha_min,
                alpha_max=self.alpha_max
            )
        else:
            num_by = sb_h // self.block_size
            num_bx = sb_w // self.block_size
            alpha_map = np.full((num_by, num_bx), fill_value=self.base_alpha, dtype=np.float32)

        bits = watermark_bits.flatten()
        total_bits = len(bits)
        num_by = sb_h // self.block_size
        num_bx = sb_w // self.block_size
        total_blocks = num_by * num_bx
        
        # Embed watermark with repetition for redundancy across high-resolution image
        reps = total_blocks // total_bits
        if reps < 1:
            raise ValueError(f"Watermark size ({total_bits} bits) exceeds host image capacity ({total_blocks} blocks).")
            
        wm_idx = 0
        for i in range(num_by):
            for j in range(num_bx):
                cur_bit = bits[wm_idx % total_bits]
                alpha = float(alpha_map[i, j])
                
                block = subband[i*self.block_size:(i+1)*self.block_size, j*self.block_size:(j+1)*self.block_size]
                dct_block = cv2.dct(block)
                
                # Modulate coefficients (COEFF_A, COEFF_B)
                r1, c1 = COEFF_A
                r2, c2 = COEFF_B
                val_a = dct_block[r1, c1]
                val_b = dct_block[r2, c2]
                
                # Differential coefficient quantization
                # If bit == 1: val_a > val_b by at least alpha
                # If bit == 0: val_b > val_a by at least alpha
                diff = val_a - val_b
                avg = (val_a + val_b) / 2.0
                
                if cur_bit == 1:
                    dct_block[r1, c1] = avg + alpha / 2.0
                    dct_block[r2, c2] = avg - alpha / 2.0
                else:
                    dct_block[r1, c1] = avg - alpha / 2.0
                    dct_block[r2, c2] = avg + alpha / 2.0
                    
                subband[i*self.block_size:(i+1)*self.block_size, j*self.block_size:(j+1)*self.block_size] = cv2.idct(dct_block)
                wm_idx += 1
                
        # Inverse DWT
        watermarked_pad_y = pywt.idwt2((subband, (LH, HL, HH)), 'haar')
        watermarked_y = np.clip(watermarked_pad_y[:h, :w], 0, 255).astype(np.uint8)
        
        if is_color:
            ycrcb[:, :, 0] = watermarked_y
            watermarked_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            watermarked_img = watermarked_y
            
        metadata = {
            "total_bits": total_bits,
            "reps": reps,
            "total_blocks": total_blocks,
            "adaptive": adaptive,
            "alpha_map": alpha_map,
            "base_alpha": self.base_alpha
        }
        return watermarked_img, metadata

    def extract(
        self, 
        watermarked_image: np.ndarray, 
        total_bits: int
    ) -> np.ndarray:
        """
        Blind extraction of watermark from watermarked or attacked image.
        Does not require the original host image.
        Uses majority voting across repeated tile blocks.
        
        Returns:
        - extracted_bits: 1D numpy array of length total_bits (0s and 1s)
        """
        is_color = (len(watermarked_image.shape) == 3)
        if is_color:
            ycrcb = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YCrCb)
            y_channel = ycrcb[:, :, 0].copy()
        else:
            y_channel = watermarked_image.copy()
            
        h, w = y_channel.shape
        target_h = int(np.ceil(h / (self.block_size * 2)) * (self.block_size * 2))
        target_w = int(np.ceil(w / (self.block_size * 2)) * (self.block_size * 2))
        
        pad_y = np.pad(y_channel, ((0, target_h - h), (0, target_w - w)), mode='reflect').astype(np.float32)
        coeffs2 = pywt.dwt2(pad_y, 'haar')
        LL, _ = coeffs2
        
        sb_h, sb_w = LL.shape
        num_by = sb_h // self.block_size
        num_bx = sb_w // self.block_size
        total_blocks = num_by * num_bx
        
        # Array to collect votes for each bit position
        votes = [[] for _ in range(total_bits)]
        
        wm_idx = 0
        for i in range(num_by):
            for j in range(num_bx):
                block = LL[i*self.block_size:(i+1)*self.block_size, j*self.block_size:(j+1)*self.block_size]
                dct_block = cv2.dct(block)
                
                r1, c1 = COEFF_A
                r2, c2 = COEFF_B
                val_a = dct_block[r1, c1]
                val_b = dct_block[r2, c2]
                
                bit_vote = 1 if (val_a > val_b) else 0
                votes[wm_idx % total_bits].append(bit_vote)
                wm_idx += 1
                
        # Majority voting
        extracted = np.zeros(total_bits, dtype=np.uint8)
        for b in range(total_bits):
            if len(votes[b]) > 0:
                extracted[b] = 1 if (np.mean(votes[b]) >= 0.5) else 0
            else:
                extracted[b] = 0
                
        return extracted
