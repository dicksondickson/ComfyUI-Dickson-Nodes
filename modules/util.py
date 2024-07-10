import torch

# Transformations for image processing
#from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms.v2 import ToTensor, ToPILImage

import numpy as np
import PIL.Image as Image
import torch


# Check if CUDA (GPU acceleration) is available and set the device accordingly
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")



# ========= pil2tensor ========= #
def pil2tensor(image):
    print("[DICKSON-NODES] pil2tensor")
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


"""
def pil2tensor_optimized(image):
    return ToTensor()(image).unsqueeze(0)


def pil2tensor_gpu(image):
    return ToTensor()(image).unsqueeze(0).to(device)
"""


# ========= tensor2pil ========= #
def tensor2pil(image):
    print("[DICKSON-NODES] tensor2pil")
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


"""
def tensor2pil_optimized(image):
    return ToPILImage()(image.squeeze().clamp(0, 1))



def tensor2pil_gpu(image):
    return ToPILImage()(image.squeeze().clamp(0, 1).cpu())
"""

# =============================== #


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return torch.nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return torch.nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return torch.nn.Conv3d(*args, **kwargs)


def linear(*args, **kwargs):
    return torch.nn.Linear(*args, **kwargs)


def normalization(channels):
    return torch.nn.GroupNorm(32, channels)
