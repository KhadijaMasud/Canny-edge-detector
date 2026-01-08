import numpy as np

def calculate_filter_size (sigma, T=0.3):
    sHalf = int(round(np.sqrt(-np.log(T) * 2 * (sigma**2))))
    N = 2 * sHalf + 1
    return N, sHalf
def calculate_gradient (sHalf, sigma,scale=255.0):
    xs = np.arange(-sHalf, sHalf+1)
    X, Y = np.meshgrid(xs, xs)   # X varies horizontally, Y vertically
    G = np.exp(- (X**2 + Y**2) / (2 * sigma**2))
    Gx = (-X / (sigma**2)) * G
    Gy = (-Y / (sigma**2)) * G
    # scale and round to integer masks
    Gx_i = np.round(Gx * scale).astype(np.int32)
    Gy_i = np.round(Gy * scale).astype(np.int32)
    return Gx_i, Gy_i, scale
def gaussian_filter(sHalf, sigma):
    xs = np.arange(-sHalf, sHalf+1)
    X, Y = np.meshgrid(xs, xs)
    G = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    G /= np.sum(G)   # normalize so sum = 1
    return G