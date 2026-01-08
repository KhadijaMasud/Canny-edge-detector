import numpy as np

def non_max_suppression(mag, qdir):
    H, W = mag.shape
    out = np.zeros_like(mag, dtype=np.float64)
    # directions mapping to neighbor offsets:
    neigh = {
        0: [(0, -1), (0, 1)],      # E-W
        1: [(-1, 1), (1, -1)],     # NE-SW
        2: [(-1, 0), (1, 0)],      # N-S
        3: [(-1, -1), (1, 1)]      # NW-SE
    }
    # ignore 1-pixel border
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
