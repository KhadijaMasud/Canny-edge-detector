# BSCS23144
# Khadija Masood
# Assignment 02

import argparse
import os
from PIL import Image
import numpy as np

from filters import calculate_filter_size, calculate_gradient, gaussian_filter
from convolve import convolve2d
from gradient import compute_magnitude_and_direction, quantize_directions
from nms import non_max_suppression
from hysteresis import hysteresis
from utils import save_uint8_image, to_grayscale

def process_image(path, out_folder, sigma=1.0, T=0.3, Tl=None, Th=None):
    name = os.path.splitext(os.path.basename(path))[0]
    im = Image.open(path).convert('L')
    img = np.array(im).astype(np.float64)

    # Step 1: Calculate filter size
    N, sHalf = calculate_filter_size(sigma, T)

    # Step 2: Apply Gaussian smoothing (normalized filter, sums to 1)
    G = gaussian_filter(sHalf, sigma)
    smoothed = convolve2d(img, G)
    save_uint8_image(scale_array_to_uint8(smoothed),
                     os.path.join(out_folder, f"{name}_smoothed_{sigma}.png"))

    # Step 3: Compute derivative of Gaussian filters
    Gx, Gy, scale = calculate_gradient(sHalf, sigma)

    # Step 4: Convolve with derivative filters
    fx = convolve2d(smoothed, Gx)
    fy = convolve2d(smoothed, Gy)

    # Save fx, fy
    save_uint8_image(scale_array_to_uint8(fx),
                     os.path.join(out_folder, f"{name}_fx_{sigma}.png"))
    save_uint8_image(scale_array_to_uint8(fy),
                     os.path.join(out_folder, f"{name}_fy_{sigma}.png"))

    # Step 5: Magnitude and angle
    magnitude, angle = compute_magnitude_and_direction(fx, fy, scale)
    save_uint8_image(scale_array_to_uint8(magnitude),
                     os.path.join(out_folder, f"{name}_magnitude_{sigma}.png"))

    # Step 6: Quantize directions
    q = quantize_directions(angle)
    save_uint8_image((q * 60).astype(np.uint8),
                     os.path.join(out_folder, f"{name}_quantized_{sigma}.png"))

    # Step 7: Non-maximum suppression
    nms = non_max_suppression(magnitude, q)
    save_uint8_image(scale_array_to_uint8(nms),
                     os.path.join(out_folder, f"{name}_nms_{sigma}.png"))

    # Step 8: Hysteresis thresholding
    maxv = np.max(magnitude)
    if Th is None: Th = 0.2 * maxv
    if Tl is None: Tl = 0.1 * maxv

    edges = hysteresis(nms, Tl, Th)
    save_uint8_image(edges.astype(np.uint8) * 255,
                     os.path.join(out_folder, f"{name}_canny_{sigma}_Th{int(Th)}_Tl{int(Tl)}.png"))

def scale_array_to_uint8(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)
    out = (arr - mn) * 255.0 / (mx - mn)
    return np.clip(out, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--input_ext", default="png")
    parser.add_argument("--output_ext", default="png")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=0.3)
    parser.add_argument("--Th", type=float, default=None)
    parser.add_argument("--Tl", type=float, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    files = sorted([os.path.join(args.input_folder, f)
                    for f in os.listdir(args.input_folder)
                    if f.endswith(args.input_ext)])
    for f in files:
        process_image(f, args.output_folder,
                      sigma=args.sigma, T=args.T, Th=args.Th, Tl=args.Tl)
