"""
tracker.py
==========
Maintains an Excel log (`watermarking_log.xlsx`) tracking all images,
watermarking operations, parameters, and imperceptibility metrics.
"""

import os
from datetime import datetime
import pandas as pd

LOG_FILE = "watermarking_log.xlsx"

COLUMNS = [
    "Timestamp",
    "Image Name",
    "Resolution (WxH)",
    "Megapixels (MP)",
    "Scheme",
    "Watermark Payload",
    "Arnold Scrambled",
    "Key Permuted",
    "Base Alpha",
    "PSNR (dB)",
    "SSIM",
    "Watermarked File Path"
]

def init_tracker():
    """Initializes the Excel tracking file if it does not already exist."""
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=COLUMNS)
        df.to_excel(LOG_FILE, index=False, engine="openpyxl")

def log_watermark_operation(
    image_name: str,
    resolution: str,
    megapixels: float,
    scheme: str,
    watermark_payload: str,
    arnold_scrambled: bool,
    key_permuted: bool,
    base_alpha: float,
    psnr: float,
    ssim: float,
    watermarked_path: str = ""
):
    """Appends a new record to the Excel log."""
    init_tracker()
    
    new_record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Image Name": image_name,
        "Resolution (WxH)": resolution,
        "Megapixels (MP)": round(megapixels, 2),
        "Scheme": scheme,
        "Watermark Payload": watermark_payload,
        "Arnold Scrambled": "Yes" if arnold_scrambled else "No",
        "Key Permuted": "Yes" if key_permuted else "No",
        "Base Alpha": round(base_alpha, 2),
        "PSNR (dB)": round(psnr, 2) if psnr != float('inf') else "inf",
        "SSIM": round(ssim, 4),
        "Watermarked File Path": watermarked_path
    }
    
    try:
        df = pd.read_excel(LOG_FILE, engine="openpyxl")
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        df.to_excel(LOG_FILE, index=False, engine="openpyxl")
    except Exception as e:
        print(f"Error logging to Excel: {e}")

def get_log_dataframe() -> pd.DataFrame:
    """Reads and returns the current log as a DataFrame."""
    init_tracker()
    try:
        return pd.read_excel(LOG_FILE, engine="openpyxl")
    except Exception:
        return pd.DataFrame(columns=COLUMNS)
