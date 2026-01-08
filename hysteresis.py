import numpy as np

def hysteresis(nms_img, Tl, Th):
    H, W = nms_img.shape
    out = np.zeros((H, W), dtype=np.uint8)
    visited = np.zeros((H, W), dtype=bool)

    # set border to zero
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
                        if not visited[nr, nc] and nms_img[nr, nc] >= Tl:
                            stack.append((nr, nc))
    return out
