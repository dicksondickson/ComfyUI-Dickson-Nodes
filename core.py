"""
@author dicksondickson
@title: Dickson Nodes
@nickname: Dickson Nodes
@description: This is a set of custom nodes that I've either written myself or adapted from other authors for my own convenience. Currently includes color matching node forked from StableSR and TTPlanet's controlnet preprocessor. https://github.com/dicksondickson

"""

import os
import time


import torch
from torch import Tensor


import cv2
import numpy as np


from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo


# Image Load
import folder_paths
import latent_preview
import node_helpers


# Color Match Functions
from .modules.colormatch import adain_color_match, wavelet_color_match, pil2tensor, tensor2pil 


# TT Planet Controlnet Preprocessor
from .modules.ttplanetcontrolnet import tensor2pil_tt, apply_gaussian_blur, apply_guided_filter






# ========= Dickson Image Load ========= #



class DicksonLoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "text": ("STRING", {"forceInput": True}),
                },
        }

    CATEGORY = "Dickson-Nodes/Image"

    #RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT","INT", "STRING",)
    RETURN_NAMES = ("IMAGE", "MASK", "filename", "width", "height", "imageSize",)
    
    #OUTPUT_NODE = True
    #OUTPUT_IS_LIST = (False, False, False, False, False, True,)
    
    
    FUNCTION = "load_image"
    
    def load_image(self, image):
        
        image_path = folder_paths.get_annotated_filepath(image)
        #print(image)
        #filename = os.path.splitext(os.path.basename(image_path))[0]
        filename = image.rsplit('.', 1)[0]
        #print(filename)
        
        img = node_helpers.pillow(Image.open, image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None
        
        

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            
            
            # get image size
            shape = image.shape
            width = shape[2]
            height = shape[1]
            imageSizeStr = f"Width: {height} \n Height: {width}"
            imageSize = {"ui": {"text": imageSizeStr}, "result": (imageSizeStr,)}
            
            
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask, filename, width, height, imageSize,)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True




# ========= Dickson Color Match ========= #

class DicksonColorMatch:
    CATEGORY = "Dickson-Nodes/Color"
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
    

    # Color Match Func
    def match_color(self, image, color_ref_image, color_match_mode):
        
        
        print("[DICKSON-NODES] DicksonColorMatch")
        
        
        # Benchmark time
        start_time = time.time()
        
        color_match_func = (
            wavelet_color_match if color_match_mode == "Wavelet" else adain_color_match
        )
                
        result_image = color_match_func(tensor2pil(image), tensor2pil(color_ref_image))
        #result_image = color_match_func(image, color_ref_image)
        
        
        refined_image = pil2tensor(result_image)
        #refined_image = result_image
        
        
        # Benchmark time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[DICKSON-NODES] Execution time: {execution_time:.4f} seconds\n")
        
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
    "DicksonLoadImage": DicksonLoadImage,
    "TTPlanet_Tile_Preprocessor_GF": TTPlanet_Tile_Preprocessor_GF,
    "TTPlanet_Tile_Preprocessor_Simple": TTPlanet_Tile_Preprocessor_Simple,
    "TTPlanet_Tile_Preprocessor_cufoff": TTPlanet_Tile_Preprocessor_cufoff
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DicksonColorMatch": "Dickson Color Match",
    "DicksonLoadImage": "Dickson Load Image",
    "TTPlanet_Tile_Preprocessor_GF": "🪐TTPlanet Tile Preprocessor GF",
    "TTPlanet_Tile_Preprocessor_Simple": "🪐TTPlanet Tile Preprocessor Simple",
    "TTPlanet_Tile_Preprocessor_cufoff": "🪐TTPlanet Tile Preprocessor cufoff"
}

