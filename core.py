"""
@author dicksondickson
@title: Dickson Nodes
@nickname: Dickson Nodes
@description: This is a set of custom nodes that I've either written myself or adapted from other authors for my own convenience. Currently includes color matching node forked from StableSR and TTPlanet's controlnet preprocessor. https://github.com/dicksondickson

"""

import os

import torch
from torch import Tensor

import cv2
import numpy as np
from PIL import Image



# Color Match
#from .modules.colormatch import adain_color_match, wavelet_color_match # standard original
from .modules.colormatch_accelerate import adain_color_match, wavelet_color_match # accelerated
from .modules.util import pil2tensor, tensor2pil


# TT Planet Controlnet Preprocessor
from .modules.ttplanetcontrolnet import tensor2pil_tt, apply_gaussian_blur, apply_guided_filter



# ========= Dickson Color Match ========= #

class DicksonColorMatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_ref_image": ("IMAGE",),
                "image": ("IMAGE",),
                "color_match_mode": (
                    [
                        "Wavelet",
                        "AdaIN",
                    ],
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match_color"
    CATEGORY = "Dickson-Nodes/Color"

    def match_color(self, image, color_ref_image, color_match_mode):
        color_match_func = (
            wavelet_color_match if color_match_mode == "Wavelet" else adain_color_match
        )
        result_image = color_match_func(tensor2pil(image), tensor2pil(color_ref_image))
        refined_image = pil2tensor(result_image)
        return (refined_image,)


# ========= TTPlanet ========= #

class TTPlanet_Tile_Preprocessor_GF:
    def __init__(self, blur_strength=3.0, radius=7, eps=0.01):
        self.blur_strength = blur_strength
        self.radius = radius
        self.eps = eps

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 1.00, "min": 1.00, "max": 8.00, "step": 0.05}),
                "blur_strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "radius": ("INT", {"default": 7, "min": 1, "max": 20, "step": 1}),
                "eps": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_output",)
    FUNCTION = 'process_image'
    CATEGORY = 'TTP_TILE'

    def process_image(self, image, scale_factor, blur_strength, radius, eps):
        ret_images = []
    
        for i in image:
            # Convert tensor to PIL for processing
            _canvas = tensor2pil_tt(torch.unsqueeze(i, 0)).convert('RGB')
            img_np = np.array(_canvas)[:, :, ::-1]  # RGB to BGR
            
            # Apply Gaussian blur
            img_np = apply_gaussian_blur(img_np, ksize=int(blur_strength), sigmaX=blur_strength / 2)            

            # Apply Guided Filter
            img_np = apply_guided_filter(img_np, radius, eps)


            # Resize image
            height, width = img_np.shape[:2]
            new_width = int(width / scale_factor)
            new_height = int(height / scale_factor)
            resized_down = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
            resized_img = cv2.resize(resized_down, (width, height), interpolation=cv2.INTER_CUBIC)
            


            # Convert OpenCV back to PIL and then to tensor
            pil_img = Image.fromarray(resized_img[:, :, ::-1])  # BGR to RGB
            tensor_img = pil2tensor(pil_img)
            ret_images.append(tensor_img)
        
        return (torch.cat(ret_images, dim=0),)
        
class TTPlanet_Tile_Preprocessor_Simple:
    def __init__(self, blur_strength=3.0):
        self.blur_strength = blur_strength

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 2.00, "min": 1.00, "max": 8.00, "step": 0.05}),
                "blur_strength": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_output",)
    FUNCTION = 'process_image'
    CATEGORY = 'TTP_TILE'

    def process_image(self, image, scale_factor, blur_strength):
        ret_images = []
    
        for i in image:
            # Convert tensor to PIL for processing
            _canvas = tensor2pil_tt(torch.unsqueeze(i, 0)).convert('RGB')
        
            # Convert PIL image to OpenCV format
            img_np = np.array(_canvas)[:, :, ::-1]  # RGB to BGR
        
            # Resize image first if you want blur to apply after resizing
            height, width = img_np.shape[:2]
            new_width = int(width / scale_factor)
            new_height = int(height / scale_factor)
            resized_down = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
            resized_img = cv2.resize(resized_down, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
            # Apply Gaussian blur after resizing
            img_np = apply_gaussian_blur(resized_img, ksize=int(blur_strength), sigmaX=blur_strength / 2)
        
            # Convert OpenCV back to PIL and then to tensor
            _canvas = Image.fromarray(img_np[:, :, ::-1])  # BGR to RGB
            tensor_img = pil2tensor(_canvas)
            ret_images.append(tensor_img)
    
        return (torch.cat(ret_images, dim=0),)        

class TTPlanet_Tile_Preprocessor_cufoff:
    def __init__(self, blur_strength=3.0, cutoff_frequency=30, filter_strength=1.0):
        self.blur_strength = blur_strength
        self.cutoff_frequency = cutoff_frequency
        self.filter_strength = filter_strength

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 1.00, "min": 1.00, "max": 8.00, "step": 0.05}),
                "blur_strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "cutoff_frequency": ("INT", {"default": 100, "min": 0, "max": 256, "step": 1}),
                "filter_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_output",)
    FUNCTION = 'process_image'
    CATEGORY = 'TTP_TILE'

    def process_image(self, image, scale_factor, blur_strength, cutoff_frequency, filter_strength):
        ret_images = []
    
        for i in image:
            # Convert tensor to PIL for processing
            _canvas = tensor2pil_tt(torch.unsqueeze(i, 0)).convert('RGB')
            img_np = np.array(_canvas)[:, :, ::-1]  # RGB to BGR

            # Apply low pass filter with new strength parameter
            img_np = apply_low_pass_filter(img_np, cutoff_frequency, filter_strength)

            # Resize image
            height, width = img_np.shape[:2]
            new_width = int(width / scale_factor)
            new_height = int(height / scale_factor)
            resized_down = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
            resized_img = cv2.resize(resized_down, (width, height), interpolation=cv2.INTER_LANCZOS4)
            
            # Apply Gaussian blur
            img_np = apply_gaussian_blur(img_np, ksize=int(blur_strength), sigmaX=blur_strength / 2)
            
            # Convert OpenCV back to PIL and then to tensor
            pil_img = Image.fromarray(resized_img[:, :, ::-1])  # BGR to RGB
            tensor_img = pil2tensor(pil_img)
            ret_images.append(tensor_img)
        
        return (torch.cat(ret_images, dim=0),)





NODE_CLASS_MAPPINGS = {
    "DicksonColorMatch": DicksonColorMatch,
    "TTPlanet_Tile_Preprocessor_GF": TTPlanet_Tile_Preprocessor_GF,
    "TTPlanet_Tile_Preprocessor_Simple": TTPlanet_Tile_Preprocessor_Simple,
    "TTPlanet_Tile_Preprocessor_cufoff": TTPlanet_Tile_Preprocessor_cufoff
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DicksonColorMatch": "Dickson Color Match",
    "TTPlanet_Tile_Preprocessor_GF": "🪐TTPlanet Tile Preprocessor GF",
    "TTPlanet_Tile_Preprocessor_Simple": "🪐TTPlanet Tile Preprocessor Simple",
    "TTPlanet_Tile_Preprocessor_cufoff": "🪐TTPlanet Tile Preprocessor cufoff"
}

