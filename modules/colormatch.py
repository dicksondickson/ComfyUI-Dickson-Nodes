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
from torchvision.transforms.v2 import ToTensor, ToPILImage


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
def wavelet_color_match(target: Image, source: Image):
    
    # Print a message to indicate the function has been called
    print("[DICKSON-NODES] wavelet_color_match")
    
    # Check if CUDA is available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Check for available GPU devices
    if torch.cuda.is_available():
        # If CUDA (NVIDIA GPU) is available, use it
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # If MPS (Apple Silicon GPU) is available, use it
        device = torch.device("mps")
    else:
        # If no GPU is available, use CPU
        device = torch.device("cpu")
        
    
    # Print which device (CPU or GPU) is being used
    print(f"[DICKSON-NODES] Using device: {device}")
    
    
    
    # Resize the source image to match the target image size
    source = source.resize(target.size, resample=Image.Resampling.LANCZOS)
    

    # Create a function to convert PIL Images to PyTorch tensors
    to_tensor = ToTensor()
    
    #target_tensor = to_tensor(target).unsqueeze(0)
    #source_tensor = to_tensor(source).unsqueeze(0)

    # Convert target image to tensor, add a batch dimension, and move to the selected device
    target_tensor = to_tensor(target).unsqueeze(0).to(device)
    
    # Convert source image to tensor, add a batch dimension, and move to the selected device
    source_tensor = to_tensor(source).unsqueeze(0).to(device)


    # Apply wavelet reconstruction
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)


    # Move the result tensor back to CPU for image conversion
    result_tensor = result_tensor.cpu()


    # Create a function to convert PyTorch tensors back to PIL Images
    to_image = ToPILImage()
    
    # Convert the result tensor to a PIL Image, removing the batch dimension and clamping values between 0 and 1
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))


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
    # Print a message to indicate the function has been called
    print("[DICKSON-NODES] wavelet_blur")
    
    
    # Get the device (CPU or GPU) that the input image is on
    # Ensure the image is on the correct device
    device = image.device
    print(f"[DICKSON-NODES] Using device: {device}")
    
    
    
    # Define the blur kernel values
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    
    
    # Convert the kernel values to a PyTorch tensor on the same device as the input image
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    
    
    # Add two dimensions to the kernel to make it a 4D tensor (required for conv2d)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    
    
    # Repeat the kernel for each input channel (assuming 3 channels: R, G, B)
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    
    
    # Pad the input image to maintain size after convolution
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')  
    
    
    # Apply convolution to blur the image
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    
    # Return the blurred image
    return output




# Function for wavelet decomposition
def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    print("[DICKSON-NODES] wavelet_decomposition")
    
    # Initialize high_freq tensor with zeros, same shape as input image
    high_freq = torch.zeros_like(image)
    
    # Iterate through each level of decomposition
    for i in range(levels):
        # Calculate the blur radius for this level
        radius = 2 ** i
        
        # Apply wavelet blur to get low frequency component
        low_freq = wavelet_blur(image, radius)
        
        # Calculate and accumulate high frequency component
        high_freq += (image - low_freq)
        
        # Update image for next iteration
        image = low_freq

    # Return high frequency and low frequency components
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

