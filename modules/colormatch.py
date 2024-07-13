# IMPORT LIBS

# Import pyTorch
import torch


# Import Tensor type for type hinting
from torch import Tensor


# Python Imaging Library for image operations
from PIL import Image, ImageOps


# PyTorch's functional API for neural network operations
from torch.nn import functional as F


# Transformations for image processing
#from torchvision.transforms import ToTensor, ToPILImage
# from torchvision.transforms.v2 import ToTensor, ToPILImage
from torchvision.transforms.v2 import ToTensor, ToPILImage, Resize, InterpolationMode

#import torchvision.transforms.v2.functional as TF

# From util
import numpy as np
#import PIL.Image as Image




# ========= pil2tensor ========= #
def pil2tensor(image):
    print("[DICKSON-NODES] pil2tensor")
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)



# ========= tensor2pil ========= #
def tensor2pil(image):
    print("[DICKSON-NODES] tensor2pil")
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )



# Main function for AdaIN color matching
def adain_color_match(target: Image, source: Image):
    print("[DICKSON-NODES] adain_color_match")
    
    # Convert images to tensors
    to_tensor = ToTensor()
    print("[DICKSON-NODES] ToTensor")
    
    
    target_tensor = to_tensor(target).unsqueeze(0)
    print("[DICKSON-NODES] target ToTensor")
    
    
    source_tensor = to_tensor(source).unsqueeze(0)
    print("[DICKSON-NODES] source ToTensor")

    # Apply adaptive instance normalization
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)

    # Convert tensor back to image
    print("[DICKSON-NODES] ToPILImage")
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image


# Main function for wavelet color matching
# def wavelet_color_match(target: Image, source: Image):
def wavelet_color_match(target: Tensor, source: Tensor):
    print("[DICKSON-NODES] wavelet_color_match")
    
    
    source = tensor2pil(source)
    source = source.resize(target.size, resample=Image.Resampling.LANCZOS)
    

    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply wavelet reconstruction
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)


    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))
    #result_image = result_tensor.squeeze(0).clamp_(0.0, 1.0)

    return result_image


# Function to calculate mean and standard deviation of a tensor
def calc_mean_std(feat: Tensor, eps=1e-5):
    """
    Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    print("[DICKSON-NODES] calc_mean_std")
    size = feat.size() # Get the size of the input tensor
    assert len(size) == 4, 'The input feature should be 4D tensor.' # Ensure input is 4D
    b, c = size[:2] # Get batch size and number of channels
    feat_var = feat.view(b, c, -1).var(dim=2) + eps # Calculate variance
    feat_std = feat_var.sqrt().view(b, c, 1, 1) # Calculate standard deviation
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1) # Calculate mean
    return feat_mean, feat_std # Return mean and standard deviation


# Function for adaptive instance normalization
def adaptive_instance_normalization(content_feat:Tensor, style_feat:Tensor):
    """
    Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    print("[DICKSON-NODES] adaptive_instance_normalization")
    size = content_feat.size() # Get size of content feature tensor
    style_mean, style_std = calc_mean_std(style_feat) # Calculate mean and std of style
    content_mean, content_std = calc_mean_std(content_feat) # Calculate mean and std of content
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size) # Normalize content feature
    return normalized_feat * style_std.expand(size) + style_mean.expand(size) # Apply style's mean and std to normalized content


# Function for wavelet blur
def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    print("[DICKSON-NODES] wavelet_blur")
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')  # Pad image
    # apply convolution
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output


# Function for wavelet decomposition
def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    print("[DICKSON-NODES] wavelet_decomposition")
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq


# Function for wavelet reconstruction
def wavelet_reconstruction(content_feat:Tensor, style_feat:Tensor):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    print("[DICKSON-NODES] wavelet_reconstruction")
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    return content_high_freq + style_low_freq

