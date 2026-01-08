# BSCS23144
# Khadija Masood
# Assignment 02
# Problem 1: Derivative and Double Derivative of Gaussian Filter

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
import os
def gaussian_2d(x, y, sigma):
    """2D Gaussian function (unnormalized)."""
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))

def gaussian_derivative_x(x, y, sigma):
    """First derivative of Gaussian with respect to x."""
    G = gaussian_2d(x, y, sigma)
    return -(x / (sigma**2)) * G

def gaussian_double_derivative_x(x, y, sigma):
    """Second derivative of Gaussian with respect to x."""
    G = gaussian_2d(x, y, sigma)
    return ((x**2) / (sigma**4) - 1 / (sigma**2)) * G

def main():
    sigma = 1.0
    s = np.linspace(-3, 3, 121)  # grid from -3 to 3
    X, Y = np.meshgrid(s, s)

    # Compute first and second derivatives
    Gx = gaussian_derivative_x(X, Y, sigma)
    Gxx = gaussian_double_derivative_x(X, Y, sigma)
    out_dir = "mathresult"
    os.makedirs(out_dir, exist_ok=True)
    # Plot first derivative
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, Gx, cmap='viridis')
    ax1.set_title("First derivative of Gaussian (∂G/∂x), σ=1")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("∂G/∂x")
    plt.tight_layout()
    plt.show()

    # Plot second derivative
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(X, Y, Gxx, cmap='plasma')
    ax2.set_title("Second derivative of Gaussian (∂²G/∂x²), σ=1")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("∂²G/∂x²")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
