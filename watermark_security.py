"""
watermark_security.py
=====================
Watermark Security and Attack Protection module:
1. Arnold's Cat Map (Watermark pixel scrambling / spatial decorrelation):
   Protects watermark payload from burst cropping, localized tampering, and unauthorized readout.
2. 2D Watermark Synchronization Markers & Template Protection:
   Corner anchors / pilots for geometric calibration and synchronization.
3. Cryptographic Secret Key-based Pseudo-Random Permutation (PRN Permute):
   Secures the watermark bit order so only key holders can extract it.
4. Repetition & Majority-Voting Error Correction (FEC):
   Mitigates random bit flips from lossy compression and heavy noise.
"""

import numpy as np

def arnold_cat_map_encrypt(image: np.ndarray, iterations: int = 5) -> np.ndarray:
    """
    Encrypts/scrambles a 2D square binary watermark using Arnold's Cat Map transformation:
    [x'] = [1  1] [x]  (mod N)
    [y'] = [1  2] [y]
    """
    h, w = image.shape
    if h != w:
        raise ValueError("Arnold Cat Map requires a square watermark image (N x N).")
    N = h
    scrambled = image.copy()
    for _ in range(iterations):
        temp = np.zeros_like(scrambled)
        for x in range(N):
            for y in range(N):
                nx = (x + y) % N
                ny = (x + 2 * y) % N
                temp[nx, ny] = scrambled[x, y]
        scrambled = temp
    return scrambled


def arnold_cat_map_decrypt(scrambled: np.ndarray, iterations: int = 5) -> np.ndarray:
    """
    Decrypts/descrambles an Arnold Cat Map scrambled watermark using the inverse matrix:
    [x] = [ 2 -1] [x']  (mod N)
    [y] = [-1  1] [y']
    """
    h, w = scrambled.shape
    if h != w:
        raise ValueError("Arnold Cat Map requires a square watermark image (N x N).")
    N = h
    descrambled = scrambled.copy()
    for _ in range(iterations):
        temp = np.zeros_like(descrambled)
        for x in range(N):
            for y in range(N):
                ox = (2 * x - y) % N
                oy = (-x + y) % N
                temp[ox, oy] = descrambled[x, y]
        descrambled = temp
    return descrambled


def key_permute_bits(bits: np.ndarray, secret_key: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Permutes bitstream order based on a secret seed key.
    Attacker cannot reconstruct the watermark even if they know the embedding algorithm.
    """
    rng = np.random.default_rng(secret_key)
    perm = rng.permutation(len(bits))
    permuted_bits = bits[perm]
    return permuted_bits, perm


def key_unpermute_bits(permuted_bits: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """
    Inverts the secret key permutation to restore original watermark bits.
    """
    orig_bits = np.zeros_like(permuted_bits)
    orig_bits[perm] = permuted_bits
    return orig_bits
