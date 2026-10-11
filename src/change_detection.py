"""Classical pixel-difference baseline for RGB images."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class ChangeResult:
    before_rgb: np.ndarray
    after_rgb: np.ndarray
    difference_gray: np.ndarray
    change_mask: np.ndarray
    overlay_rgb: np.ndarray
    changed_pixels: int
    total_pixels: int
    changed_percentage: float


def _validate_rgb(image):
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must have shape (height, width, 3).")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("Image must be at least 2x2 pixels.")
    return np.clip(image, 0, 255).astype(np.uint8)


def compare_images(before_rgb, after_rgb, threshold=25, blur_kernel=3,
                   min_region_size=0):
    """Compute thresholded grayscale difference; resizing is not registration."""
    if not 0 <= int(threshold) <= 255:
        raise ValueError("threshold must be between 0 and 255.")
    if int(blur_kernel) < 1 or int(blur_kernel) % 2 == 0:
        raise ValueError("blur_kernel must be a positive odd number.")
    if int(min_region_size) < 0:
        raise ValueError("min_region_size cannot be negative.")
    before = _validate_rgb(before_rgb)
    after = _validate_rgb(after_rgb)
    h, w = before.shape[:2]
    if after.shape[:2] != (h, w):
        after = cv2.resize(after, (w, h), interpolation=cv2.INTER_AREA)
    bgray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
    agray = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
    if blur_kernel > 1:
        bgray = cv2.GaussianBlur(bgray, (blur_kernel, blur_kernel), 0)
        agray = cv2.GaussianBlur(agray, (blur_kernel, blur_kernel), 0)
    diff = cv2.absdiff(bgray, agray)
    _, mask = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)
    if min_region_size > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask)
        for label in range(1, n):
            if stats[label, cv2.CC_STAT_AREA] >= min_region_size:
                filtered[labels == label] = 255
        mask = filtered
    else:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    changed = int(np.count_nonzero(mask))
    total = int(mask.size)
    overlay = after.copy()
    red = np.zeros_like(overlay)
    red[:] = (255, 50, 50)
    selected = mask > 0
    overlay[selected] = (0.45 * overlay[selected] + 0.55 * red[selected]).astype(np.uint8)
    return ChangeResult(before, after, diff, mask, overlay, changed, total,
                        changed / total * 100.0 if total else 0.0)
