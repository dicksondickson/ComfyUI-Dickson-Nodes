# Dickson's ComfyUI Custom Nodes Collection  

This is a set of custom nodes that I've either written myself or adapted from other authors for my own convenience.  


# Changelog  
v1.0.6 - Add image info output to imageload node.
v1.0.5 - Added ImageLoad node - provides filename and image width and height as integer.  
v1.0.4 - ColorMatch node is now 15x to 17x faster.   
v1.0.0 - Release.   



# Nodes  

**Color match node**  
Forked from StableSR as it seems that project is not maintained.  
Credit to: WSJUSA, StableSR, LIightChaser and Jianyi Wang  
  
**TTPlanet controlnet tile preprocessor node**  
Forked from [https://huggingface.co/TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic](https://huggingface.co/TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic) for my own convienence.
TTPlanet's Controlnet model is meant to be used with this node which you can download on his Huggingface repo.
Credit to: Aaron Xie / TTPlanet  

**Image Load Node**  
Provides filename output as string, image width and height as integer output, and image info output as string.   



# Installation  
Install via ComfyUI Manager ( search for dicksondickson ) or git clone this repo to your comfyui/customnodes folder.  
  


# Models Required  
TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic.  
Download all the models and put it into your ComfyUI/models/controlnet folder:  
[https://huggingface.co/TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic/tree/main](https://huggingface.co/TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic/tree/main)  
  

  

# Acknowledgment  

Thanks to [@comfyanonymous](https://github.com/comfyanonymous) [@Dr.Lt.Data](https://github.com/ltdrdata) and other authors for creating and sharing their work.  



