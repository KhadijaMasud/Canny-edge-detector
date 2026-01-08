import numpy as np


def convolve2d(image, kernel):
    # image: 2D numpy array (H,W)
    H, W = image.shape
    kH, kW = kernel.shape
    pad_h = kH // 2
    pad_w = kW // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0.0)
    out = np.zeros_like(image, dtype=np.float64)
    # naive nested loops 
    for r in range(H):
        for c in range(W):
            region = padded[r:r+kH, c:c+kW].astype(np.float64)
            out[r, c] = float((region * kernel).sum())
    return out