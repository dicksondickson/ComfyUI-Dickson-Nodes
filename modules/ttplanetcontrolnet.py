import cv2
import numpy as np
from PIL import Image
import torch


def tensor2pil_tt(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def apply_gaussian_blur(image_np, ksize=5, sigmaX=1.0):
    if ksize % 2 == 0:
        ksize += 1  # ksize must be odd
    blurred_image = cv2.GaussianBlur(image_np, (ksize, ksize), sigmaX=sigmaX)
    return blurred_image

def apply_guided_filter(image_np, radius, eps):
    # Convert image to float32 for the guided filter
    image_np_float = np.float32(image_np) / 255.0
    # Apply the guided filter
    filtered_image = cv2.ximgproc.guidedFilter(image_np_float, image_np_float, radius, eps)
    # Scale back to uint8
    filtered_image = np.clip(filtered_image * 255, 0, 255).astype(np.uint8)
    return filtered_image

