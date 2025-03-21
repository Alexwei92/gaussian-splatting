#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import colorsys
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def id2rgb(ids, dtype=np.uint8):
    # Handle invalid IDs
    if dtype == np.uint8:
        max_num_obj = 256
    elif dtype == np.uint16:
        max_num_obj = 65535
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if not np.all((0 <= ids) & (ids <= max_num_obj)):
        raise ValueError(f"All IDs should be in range(0, {max_num_obj})")
    
    # Pre-allocate output array
    rgb = np.zeros((len(ids), 3), dtype=dtype)
    
    # Skip computation for invalid regions (id == 0)
    valid_mask = ids != 0
    if not np.any(valid_mask):
        return rgb
    
    valid_ids = ids[valid_mask]
    
    # HSL to RGB conversion
    golden_ratio = 1.6180339887
    h = ((valid_ids * golden_ratio) % 1)
    s = 0.5 + (valid_ids % 2) * 0.5
    l = np.full_like(h, 0.5)
    
    # HSL to RGB conversion
    rgb_valid = np.array([colorsys.hls_to_rgb(h_val, l_val, s_val) for h_val, l_val, s_val in zip(h, l, s)])
    
    if dtype == np.uint8:
        rgb_valid = (rgb_valid * 255).astype(np.uint8)
    else:
        rgb_valid = (rgb_valid * 65535).astype(np.uint16)    
    
    # Assign colors to valid positions
    rgb[valid_mask] = rgb_valid
    return rgb

def objects_to_rgb(objects):
    # Get unique IDs and their positions
    unique_ids, inverse_indices = np.unique(objects, return_inverse=True)
    
    # Generate colors for all unique IDs
    colors = id2rgb(unique_ids, dtype=objects.dtype)
    
    # Reshape the result to match input dimensions
    rgb_mask = colors[inverse_indices].reshape((*objects.shape, 3))
    return rgb_mask