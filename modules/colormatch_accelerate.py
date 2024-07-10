# Import necessary libraries
import torch  # PyTorch library for tensor operations and neural networks
from torch import Tensor  # Import Tensor type for type hinting
from torchvision import transforms  # Transformations for image processing
from PIL import Image  # Python Imaging Library for image operations
from torch.nn import functional as F  # PyTorch's functional API for neural network operations

# Check if CUDA (GPU acceleration) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to convert PIL Image to tensor
def image_to_tensor(img):
    print("image_to_tensor")
    if isinstance(img, Image.Image):  # Check if input is a PIL Image
        transform = transforms.ToTensor()  # Create a transform to convert PIL Image to tensor
        return transform(img).unsqueeze(0).to(device)  # Convert to tensor, add batch dimension, move to device
    elif isinstance(img, Tensor):  # Check if input is already a PyTorch tensor
        return img.to(device)  # Just move the tensor to the correct device
    else:
        raise TypeError("Input must be PIL Image or PyTorch tensor")  # Raise error for invalid input types

# Function to convert tensor to PIL Image
def tensor_to_image(tensor):
    print("tensor_to_image")
    if tensor.is_cuda:  # Check if tensor is on GPU
        tensor = tensor.cpu()  # Move tensor to CPU if it's on GPU
    tensor = tensor.squeeze(0).clamp(0, 1)  # Remove batch dimension and clamp values between 0 and 1
    return transforms.ToPILImage()(tensor)  # Convert tensor to PIL Image

# Function to calculate mean and standard deviation of a tensor
@torch.jit.script  # This decorator enables JIT compilation for faster execution
def calc_mean_std(feat: Tensor, eps: float = 1e-5):
    print("calc_mean_std")
    size = feat.size()  # Get the size of the input tensor
    assert len(size) == 4, 'The input feature should be 4D tensor.'  # Ensure input is 4D
    b, c = size[:2]  # Get batch size and number of channels
    feat_var = feat.view(b, c, -1).var(dim=2) + eps  # Calculate variance
    feat_std = feat_var.sqrt().view(b, c, 1, 1)  # Calculate standard deviation
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)  # Calculate mean
    return feat_mean, feat_std  # Return mean and standard deviation

# Function for adaptive instance normalization
@torch.jit.script
def adaptive_instance_normalization(content_feat: Tensor, style_feat: Tensor):
    print("adaptive_instance_normalization")
    size = content_feat.size()  # Get size of content feature tensor
    style_mean, style_std = calc_mean_std(style_feat)  # Calculate mean and std of style
    content_mean, content_std = calc_mean_std(content_feat)  # Calculate mean and std of content
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)  # Normalize content feature
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)  # Apply style's mean and std to normalized content

# Function for wavelet blur
@torch.jit.script
def wavelet_blur(image: Tensor, radius: float):
    print("wavelet_blur")
    int_radius = max(1, int(radius))  # Convert radius to integer, ensuring it's at least 1
    
    # Define blur kernel
    kernel_vals = torch.tensor([
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ], dtype=image.dtype, device=image.device)
    
    kernel = kernel_vals.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)  # Reshape kernel for conv2d
    image = F.pad(image, (int_radius, int_radius, int_radius, int_radius), mode='replicate')  # Pad image
    return F.conv2d(image, kernel, groups=3, dilation=int_radius)  # Apply convolution

# Function for wavelet decomposition
@torch.jit.script
def wavelet_decomposition(image: Tensor, levels: int = 5):
    print("wavelet_decomposition")
    high_freq = torch.zeros_like(image)  # Initialize high frequency component
    low_freq = image  # Initialize low_freq with the original image
    for i in range(levels):
        radius = float(2 ** i)  # Calculate radius for each level, explicitly as float
        new_low_freq = wavelet_blur(low_freq, radius)  # Apply wavelet blur
        high_freq += (low_freq - new_low_freq)  # Add high frequency component
        low_freq = new_low_freq  # Update low_freq for next iteration
    return high_freq, low_freq  # Return high and low frequency components

# Function for wavelet reconstruction
@torch.jit.script
def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor):
    print("wavelet_reconstruction")
    content_high_freq, _ = wavelet_decomposition(content_feat)  # Decompose content
    _, style_low_freq = wavelet_decomposition(style_feat)  # Decompose style
    return content_high_freq + style_low_freq  # Combine high freq of content and low freq of style

# Main function for AdaIN color matching
def adain_color_match(target, source):
    print("adain_color_match")
    # Convert inputs to tensors if they're not already
    target_tensor = image_to_tensor(target)
    source_tensor = image_to_tensor(source)
    
    # Apply adaptive instance normalization
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)
    
    # Convert result back to PIL Image
    return tensor_to_image(result_tensor)

# Main function for wavelet color matching
def wavelet_color_match(target, source):
    print("wavelet_color_match")
    print(device) 
    # Convert inputs to tensors if they're not already
    target_tensor = image_to_tensor(target)
    source_tensor = image_to_tensor(source)
    
    # Apply wavelet reconstruction
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)
    
    # Convert result back to PIL Image
    return tensor_to_image(result_tensor)

# Note: The DicksonColorMatch class might need slight modifications to work with these changes.
# Here's a possible update for the match_color method:

# def match_color(self, image, color_ref_image, color_match_mode):
#     color_match_func = wavelet_color_match if color_match_mode == "Wavelet" else adain_color_match
#     result_image = color_match_func(image, color_ref_image)
#     return (transforms.ToTensor()(result_image).unsqueeze(0),)

# This updated method would:
# 1. Choose the appropriate color matching function based on the mode
# 2. Apply the color matching function to the input images
# 3. Convert the resulting PIL Image back to a tensor and add a batch dimension
# 4. Return the result as a tuple containing the tensor, as expected by the class
