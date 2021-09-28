"""Utilities for interpretability tools."""
import numpy as np
from scipy import ndimage


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
  """Applies a Gaussian blur to a 3D (WxHxC) image.

  Args:
    image: 3 dimensional ndarray / input image (W x H x C).
    sigma: Standard deviation for Gaussian blur kernel.

  Returns:
    The blurred image.
  """
  if sigma == 0:
    return image
  return ndimage.gaussian_filter(
      image, sigma=[sigma, sigma, 0], mode='constant')
