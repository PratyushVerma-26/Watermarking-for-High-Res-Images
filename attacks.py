"""
attacks.py
==========
Robustness testing suite for digital watermarks.
Implements standard attacks encountered in image processing & transmission:
1. JPEG Compression (quality factors: 30, 50, 75, 90)
2. Gaussian Noise addition
3. Salt & Pepper Noise addition
4. Median Filtering
5. Gaussian Blurring
6. Geometric Attacks (Cropping with zero-padding restoration)
7. Scaling down and up (Resizing attack)
8. Brightness / Contrast adjustment
"""

import numpy as np
import cv2

def attack_jpeg(image: np.ndarray, quality: int = 50) -> np.ndarray:
    """Simulates lossy JPEG compression with variable quality."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR if len(image.shape) == 3 else cv2.IMREAD_GRAYSCALE)
    return decimg


def attack_gaussian_noise(image: np.ndarray, mean: float = 0.0, var: float = 15.0) -> np.ndarray:
    """Adds zero-mean Gaussian noise with specified variance."""
    noise = np.random.normal(mean, var, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def attack_salt_and_pepper(image: np.ndarray, amount: float = 0.02) -> np.ndarray:
    """Adds Salt and Pepper impulsive noise."""
    noisy = image.copy()
    num_salt = int(np.ceil(amount * image.size * 0.5))
    num_pepper = int(np.ceil(amount * image.size * 0.5))
    
    # Salt
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy[tuple(coords)] = 255
    
    # Pepper
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy[tuple(coords)] = 0
    
    return noisy


def attack_median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Applies Median filtering (denoising attack)."""
    return cv2.medianBlur(image, ksize)


def attack_gaussian_blur(image: np.ndarray, ksize: int = 5, sigma: float = 1.5) -> np.ndarray:
    """Applies Gaussian smoothing/blurring."""
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def attack_cropping(image: np.ndarray, crop_percent: float = 0.15) -> np.ndarray:
    """
    Simulates cropping: zeroes out borders or central region.
    Default: zero-outs outer border margins.
    """
    h, w = image.shape[:2]
    ch = int(h * crop_percent / 2)
    cw = int(w * crop_percent / 2)
    attacked = image.copy()
    # Zero out top, bottom, left, right borders
    attacked[:ch, :] = 0
    attacked[-ch:, :] = 0
    attacked[:, :cw] = 0
    attacked[:, -cw:] = 0
    return attacked


def attack_resize(image: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Downsamples image then upsamples back to original resolution."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored


def attack_brightness(image: np.ndarray, factor: float = 1.2, beta: float = 15) -> np.ndarray:
    """Adjusts contrast and brightness."""
    res = cv2.convertScaleAbs(image, alpha=factor, beta=beta)
    return res


def get_all_attacks():
    """Returns a dictionary of attacks and their callable functions."""
    return {
        "JPEG (Q=75)": lambda img: attack_jpeg(img, quality=75),
        "JPEG (Q=50)": lambda img: attack_jpeg(img, quality=50),
        "JPEG (Q=30)": lambda img: attack_jpeg(img, quality=30),
        "Gaussian Noise (var=10)": lambda img: attack_gaussian_noise(img, var=10.0),
        "Gaussian Noise (var=25)": lambda img: attack_gaussian_noise(img, var=25.0),
        "Salt & Pepper (2%)": lambda img: attack_salt_and_pepper(img, amount=0.02),
        "Median Filter (3x3)": lambda img: attack_median_filter(img, ksize=3),
        "Gaussian Blur (5x5)": lambda img: attack_gaussian_blur(img, ksize=5, sigma=1.2),
        "Cropping (15% border)": lambda img: attack_cropping(img, crop_percent=0.15),
        "Resize / Rescale (50%)": lambda img: attack_resize(img, scale=0.50),
        "Brightness Boost": lambda img: attack_brightness(img, factor=1.1, beta=15)
    }
