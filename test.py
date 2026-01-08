# YOUR_ROLL_NUMBER
# YOUR_NAME
# Assignment 02

import os
import argparse
import math
from PIL import Image
import numpy as np

# ------------------------
# Filters / masks
# ------------------------
def calculate_filter_size(sigma, T=0.3):
    sHalf = int(round(math.sqrt(-math.log(T) * 2.0 * (sigma**2))))
    N = 2 * sHalf + 1
    return N, sHalf

def calculate_gradient_masks(sHalf, sigma, scale=255.0):
    xs = np.arange(-sHalf, sHalf+1, dtype=np.float64)
    X, Y = np.meshgrid(xs, xs)   # columns X, rows Y
    G = np.exp(- (X**2 + Y**2) / (2.0 * sigma**2))
    # first derivative of Gaussian
    Gx = (-X / (sigma**2)) * G
    Gy = (-Y / (sigma**2)) * G
    # scale and round to integer masks (as assignment requests)
    Gx_i = np.round(Gx * scale).astype(np.int32)
    Gy_i = np.round(Gy * scale).astype(np.int32)
    return Gx_i, Gy_i, scale

# ------------------------
# Convolution
# ------------------------
def convolve2d(image, kernel):
    image = image.astype(np.float64)
    H, W = image.shape
    kH, kW = kernel.shape
    pad_h = kH // 2
    pad_w = kW // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=0.0)
    out = np.zeros((H, W), dtype=np.float64)
    # naive nested loops (explicit convolution)
    for r in range(H):
        for c in range(W):
            region = padded[r:r+kH, c:c+kW]
            out[r, c] = float((region * kernel).sum())
    return out

# ------------------------
# Magnitude & Direction
# ------------------------
def compute_magnitude_and_direction(fx, fy, scale):
    mag = np.sqrt(fx.astype(np.float64)**2 + fy.astype(np.float64)**2) / float(scale)
    ang = np.degrees(np.arctan2(fy, fx)) + 180.0  # range 0..360
    ang = np.mod(ang, 360.0)
    return mag, ang

# ------------------------
# Quantize directions (4 bins)
# returns values 0..3
# ------------------------
def quantize_directions(angle_deg):
    q = np.zeros_like(angle_deg, dtype=np.uint8)
    a = angle_deg
    mask0 = ((a >= 0) & (a < 22.5)) | ((a >= 157.5) & (a < 202.5)) | ((a >= 337.5) & (a < 360))
    mask1 = ((a >= 22.5) & (a < 67.5)) | ((a >= 202.5) & (a < 247.5))
    mask2 = ((a >= 67.5) & (a < 112.5)) | ((a >= 247.5) & (a < 292.5))
    mask3 = ((a >= 112.5) & (a < 157.5)) | ((a >= 292.5) & (a < 337.5))
    q[mask0] = 0
    q[mask1] = 1
    q[mask2] = 2
    q[mask3] = 3
    return q

# Create a colored visualization of quantized directions
def quantized_colored_image(qmap):
    # assign distinct RGB colors to each quantized value
    colors = {
        0: (255, 0, 0),    # red
        1: (0, 255, 0),    # green
        2: (0, 0, 255),    # blue
        3: (255, 255, 0)   # yellow
    }
    H, W = qmap.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for k, col in colors.items():
        mask = (qmap == k)
        rgb[mask] = col
    return rgb

# ------------------------
# Non-Maximum Suppression
# ------------------------
def non_max_suppression(mag, qdir):
    H, W = mag.shape
    out = np.zeros_like(mag, dtype=np.float64)
    neigh = {
        0: [(0, -1), (0, 1)],      # East-West (compare left & right)
        1: [(-1, 1), (1, -1)],     # NE-SW
        2: [(-1, 0), (1, 0)],      # North-South
        3: [(-1, -1), (1, 1)]      # NW-SE
    }
    # ignore border
    for r in range(1, H-1):
        for c in range(1, W-1):
            dirv = int(qdir[r, c])
            n1 = neigh[dirv][0]
            n2 = neigh[dirv][1]
            v = mag[r, c]
            v1 = mag[r + n1[0], c + n1[1]]
            v2 = mag[r + n2[0], c + n2[1]]
            if v >= v1 and v >= v2:
                out[r, c] = v
            else:
                out[r, c] = 0.0
    return out

# ------------------------
# Hysteresis Thresholding (flood from strong edges)
# ------------------------
def hysteresis(nms_img, Tl, Th):
    H, W = nms_img.shape
    out = np.zeros((H, W), dtype=np.uint8)
    visited = np.zeros((H, W), dtype=bool)

    # clear borders to avoid out-of-bounds neighbor checking
    nms_img[0, :] = 0
    nms_img[-1, :] = 0
    nms_img[:, 0] = 0
    nms_img[:, -1] = 0

    strong_coords = np.argwhere(nms_img >= Th)

    for (r, c) in strong_coords:
        if visited[r, c]:
            continue
        stack = [(r, c)]
        while stack:
            rr, cc = stack.pop()
            if visited[rr, cc]:
                continue
            visited[rr, cc] = True
            if nms_img[rr, cc] >= Tl:
                out[rr, cc] = 1
                # push 8 neighbors
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if nr <= 0 or nr >= H-1 or nc <= 0 or nc >= W-1:
                            continue
                        if (not visited[nr, nc]) and (nms_img[nr, nc] >= Tl):
                            stack.append((nr, nc))
    return out

# ------------------------
# Helpers to save images (uint8)
# ------------------------
def save_uint8_image(arr, path):
    if arr.dtype != np.uint8:
        # scale to 0..255 if float
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        if mx <= mn:
            out = np.zeros_like(arr, dtype=np.uint8)
        else:
            out = ((arr - mn) * 255.0 / (mx - mn)).astype(np.uint8)
    else:
        out = arr
    Image.fromarray(out).save(path)

def save_gray_float_image(arr, path):
    # scale float array to 0..255 and save
    save_uint8_image(arr, path)

# ------------------------
# Pipeline per image
# ------------------------
def process_image(path, out_folder, sigma=1.0, T=0.3, Th=None, Tl=None):
    name = os.path.splitext(os.path.basename(path))[0]
    pil = Image.open(path).convert('L')
    img = np.array(pil).astype(np.float64)

    N, sHalf = calculate_filter_size(sigma, T)
    Gx, Gy, scale = calculate_gradient_masks(sHalf, sigma)

    # Convolution -> fx, fy
    fx = convolve2d(img, Gx)
    fy = convolve2d(img, Gy)
    # Save fx and fy (we save as scaled uint8 images)
    save_uint8_image(fx, os.path.join(out_folder, f"{name}_fx_{sigma}.png"))
    save_uint8_image(fy, os.path.join(out_folder, f"{name}_fy_{sigma}.png"))

    # Magnitude and direction
    magnitude, angle = compute_magnitude_and_direction(fx, fy, scale)
    save_gray_float_image(magnitude, os.path.join(out_folder, f"{name}_magnitude_{sigma}.png"))

    # Quantize and colored visualization
    q = quantize_directions(angle)
    # Save quantized numeric image (values 0..3 scaled)
    save_uint8_image((q * 60).astype(np.uint8), os.path.join(out_folder, f"{name}_quantized_{sigma}.png"))
    # Colored image
    rgb = quantized_colored_image(q)
    Image.fromarray(rgb).save(os.path.join(out_folder, f"{name}_quantized_color_{sigma}.png"))

    # Non-Maximum Suppression
    nms = non_max_suppression(magnitude, q)
    save_gray_float_image(nms, os.path.join(out_folder, f"{name}_nms_{sigma}.png"))

    # thresholds defaults relative to magnitude max
    maxv = np.max(magnitude)
    if Th is None:
        Th = 0.2 * maxv
    if Tl is None:
        Tl = 0.1 * maxv

    edges = hysteresis(nms.copy(), Tl, Th)
    save_uint8_image(edges * 255, os.path.join(out_folder, f"{name}_canny_{sigma}_Th{int(Th)}_Tl{int(Tl)}.png"))

    # return paths for report convenience
    return {
        "fx": f"{name}_fx_{sigma}.png",
        "fy": f"{name}_fy_{sigma}.png",
        "magnitude": f"{name}_magnitude_{sigma}.png",
        "quantized": f"{name}_quantized_{sigma}.png",
        "quantized_color": f"{name}_quantized_color_{sigma}.png",
        "nms": f"{name}_nms_{sigma}.png",
        "canny": f"{name}_canny_{sigma}_Th{int(Th)}_Tl{int(Tl)}.png"
    }

# ------------------------
# Main CLI
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="Canny implementation pipeline (single-file).")
    parser.add_argument("--input_folder", required=True, help="folder with input images")
    parser.add_argument("--output_folder", required=True, help="folder to save results")
    parser.add_argument("--input_ext", default="png", help="input image extension (e.g. png, jpg)")
    parser.add_argument("--sigma", type=float, default=1.0, help="sigma for Gaussian derivative")
    parser.add_argument("--T", type=float, default=0.3, help="T parameter for filter size computation")
    parser.add_argument("--Th", type=float, help="High threshold (absolute). If omitted will use 0.2*max")
    parser.add_argument("--Tl", type=float, help="Low threshold (absolute). If omitted will use 0.1*max")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    files = sorted([f for f in os.listdir(args.input_folder) if f.lower().endswith(args.input_ext.lower())])
    if not files:
        print("No input images found. Check input_folder and input_ext.")
        return

    for fname in files:
        path = os.path.join(args.input_folder, fname)
        print("Processing:", fname)
        result_paths = process_image(path, args.output_folder, sigma=args.sigma, T=args.T, Th=args.Th, Tl=args.Tl)
        # print saved files for report copy-paste
        for k, v in result_paths.items():
            print(f"  {k}: {os.path.join(args.output_folder, v)}")

if __name__ == "__main__":
    main()
