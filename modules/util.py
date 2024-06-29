import numpy as np
import PIL.Image as Image
import torch


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)



def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


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
