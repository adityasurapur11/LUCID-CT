"""
Image Preprocessing and Lung Segmentation Module
Handles Hounsfield Unit (HU) windowing, spatial resampling, lung parenchyma extraction,
and ROI cropping for CT scan slices and 3D volumes.
"""

import numpy as np
from scipy import ndimage

try:
    import cv2
except ImportError:
    cv2 = None

def apply_hu_window(volume: np.ndarray, window_center: float = -600.0, window_width: float = 1500.0) -> np.ndarray:
    """
    Applies Hounsfield Unit (HU) windowing for optimal lung tissue visualization.
    Lung window standard parameters: Center = -600, Width = 1500 [-1350, 150].
    """
    min_hu = window_center - (window_width / 2.0)
    max_hu = window_center + (window_width / 2.0)
    
    windowed = np.clip(volume, min_hu, max_hu)
    # Normalize to [0, 1]
    normalized = (windowed - min_hu) / (max_hu - min_hu)
    return normalized.astype(np.float32)


def segment_lung_mask_slice(slice_2d: np.ndarray) -> np.ndarray:
    """
    Morphological & thresholding-based lung parenchyma segmentation for a 2D slice.
    Returns binary mask (1 for lung tissue, 0 otherwise).
    """
    # Ensure float input in [0, 1]
    if slice_2d.max() > 1.0:
        img = (slice_2d / 255.0).astype(np.float32)
    else:
        img = slice_2d.astype(np.float32)

    if cv2 is not None:
        img_uint8 = (img * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img_uint8, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clear external border-connected air (scanner background outside patient body)
        h, w = thresh.shape
        flood = thresh.copy()
        mask_flood = np.zeros((h + 2, w + 2), np.uint8)
        for x in [0, w - 1]:
            for y in range(h):
                if flood[y, x] == 255:
                    cv2.floodFill(flood, mask_flood, (x, y), 0)
        for y in [0, h - 1]:
            for x in range(w):
                if flood[y, x] == 255:
                    cv2.floodFill(flood, mask_flood, (x, y), 0)
                    
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(flood, cv2.MORPH_CLOSE, kernel, iterations=2)
    else:
        # Fallback using scipy ndimage
        blurred = ndimage.gaussian_filter(img, sigma=1.0)
        thresh_val = np.mean(blurred) * 0.7
        closed = (blurred < thresh_val).astype(np.uint8) * 255
        # Clear border touching pixels
        closed[0, :] = 0
        closed[-1, :] = 0
        closed[:, 0] = 0
        closed[:, -1] = 0

    labels, num = ndimage.label(closed)
    if num > 0:
        sizes = ndimage.sum(closed, labels, range(1, num + 1))
        sorted_indices = np.argsort(sizes)[::-1]
        
        mask = np.zeros_like(closed, dtype=np.float32)
        min_lung_size = max(50.0, 0.005 * slice_2d.size)
        for idx in sorted_indices[:2]:
            if sizes[idx] >= min_lung_size:
                mask[labels == (idx + 1)] = 1.0
        # If no component met min_lung_size but components exist, take top 1
        if mask.sum() == 0 and sizes[sorted_indices[0]] > 20:
            mask[labels == (sorted_indices[0] + 1)] = 1.0
    else:
        mask = (closed > 0).astype(np.float32)
        
    return mask.astype(np.float32)



def segment_lung_volume(volume_3d: np.ndarray) -> np.ndarray:
    """
    Applies 2D lung segmentation slice by slice across a 3D volume.
    """
    mask_3d = np.zeros_like(volume_3d, dtype=np.float32)
    for i in range(volume_3d.shape[0]):
        mask_3d[i] = segment_lung_mask_slice(volume_3d[i])
    return mask_3d


def extract_roi(volume: np.ndarray, center: tuple, patch_size: tuple = (32, 64, 64)) -> np.ndarray:
    """
    Extracts a 3D crop centered around a candidate nodule location (z, y, x).
    """
    z, y, x = center
    pz, py, px = patch_size
    
    z_min = max(0, z - pz // 2)
    z_max = min(volume.shape[0], z + pz // 2)
    y_min = max(0, y - py // 2)
    y_max = min(volume.shape[1], y + py // 2)
    x_min = max(0, x - px // 2)
    x_max = min(volume.shape[2], x + px // 2)
    
    crop = volume[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # Pad if near boundaries
    pad_z = (pz - crop.shape[0]) // 2
    pad_y = (py - crop.shape[1]) // 2
    pad_x = (px - crop.shape[2]) // 2
    
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        crop = np.pad(crop, ((pad_z, pz - crop.shape[0] - pad_z),
                             (pad_y, py - crop.shape[1] - pad_y),
                             (pad_x, px - crop.shape[2] - pad_x)), mode='constant')
        
    return crop
