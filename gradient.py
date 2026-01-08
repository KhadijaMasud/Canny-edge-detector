import numpy as np

def compute_magnitude_and_direction(fx, fy, scale):
    # fx, fy: outputs of convolution with integer masks
    # magnitude scaled down by scale
    mag = np.sqrt(fx.astype(np.float64)**2 + fy.astype(np.float64)**2) / float(scale)
    # angle in degrees 0..360
    ang = np.degrees(np.arctan2(fy, fx)) + 180.0
    ang = np.mod(ang, 360.0)
    return mag, ang

def quantize_directions(angle_deg):
    # returns 0..3 per table
    q = np.zeros_like(angle_deg, dtype=np.uint8)
    a = angle_deg
    # Value 0: 0-22.5, 157.5-202.5, 337.5-360
    mask0 = ( (a >= 0) & (a < 22.5) ) | ( (a >= 157.5) & (a < 202.5) ) | ( (a >= 337.5) & (a < 360) )
    # Value 1: 22.5-67.5, 202.5-247.5
    mask1 = ( (a >= 22.5) & (a < 67.5) ) | ( (a >= 202.5) & (a < 247.5) )
    # Value 2: 67.5-112.5, 247.5-292.5
    mask2 = ( (a >= 67.5) & (a < 112.5) ) | ( (a >= 247.5) & (a < 292.5) )
    # Value 3: 112.5-157.5, 292.5-337.5
    mask3 = ( (a >= 112.5) & (a < 157.5) ) | ( (a >= 292.5) & (a < 337.5) )
    q[mask0] = 0
    q[mask1] = 1
    q[mask2] = 2
    q[mask3] = 3
    return q
