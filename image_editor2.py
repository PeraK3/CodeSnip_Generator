from PIL import Image
from scipy import io
import cv2
import numpy as np
import pandas as pd


def adjust_image(image_input, alpha=1.0, beta=0):
    # If it's a path (string), read from disk
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
    
    # If it's already a numpy array, use it directly
    elif isinstance(image_input, np.ndarray):
        img = image_input
    
    # If it's a Streamlit UploadedFile
    else:
        file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image could not be loaded. Check the input type.")

    # Apply contrast (alpha) and brightness (beta)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted

def resize_to_average(image_file, ref_size=None):
    """
    Resize an image while preserving aspect ratio.
    If ref_size is given, both images will share the same final size.
    No padding or background fill is applied.
    """
    image = Image.open(image_file).convert("RGB")

    if ref_size is not None:
        image = image.resize(ref_size, Image.LANCZOS)
    else:
        w, h = image.size
        avg_dim = int((w + h) / 2)
        image.thumbnail((avg_dim, avg_dim), Image.LANCZOS)

    return np.array(image)

def auto_convert_columns(df):
    """
    Convert numeric-like columns to numeric values while keeping categorical columns as text.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        # Skip columns that are clearly categorical
        if col.lower() in ["label", "type", "band id", "sample", "source_file"]:
            categorical_cols.append(col)
            continue

        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            numeric_ratio = converted.notna().mean()
            if numeric_ratio >= 0.6:
                df[col] = converted
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        except Exception:
            categorical_cols.append(col)

    # Drop empty columns or rows
    df = df.dropna(axis=1, how="all").dropna(how="all")

    return df, numeric_cols, categorical_cols

