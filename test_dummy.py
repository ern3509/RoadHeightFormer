'''The goal of this module is to provide a dataset class for the CARDSet dataset on the RoadBEV model.'''
import os
import open3d as o3d
from typing import Any, Callable, Optional, Dict
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
import copy
import json, re, cv2
import PIL
from torch.utils.data import random_split
from torch.utils.data import Subset


#create a dummy height map and mask visualization function
dummy_height_map = torch.randn(164, 64) * 100  # Random height map in cm
dummy_mask = (torch.rand(164, 64) > 0.3).float()  # Random mask with 70% valid points


def visualize_height_map_and_mask(height_map, mask, colormap='plasma', save_path='Heightmap/height_map_visualization.png'):
        """ Visualize the height map and mask. Cells without points are black, and cells with height GT are mapped with a colormap.

        Parameters:
            height_map (torch.Tensor): The height map tensor of shape (H, W).
            mask (torch.Tensor): The mask tensor of shape (H, W), where 1 indicates valid points and 0 indicates no points.
            colormap (str): The colormap to use for valid height values (default: 'plasma').
            save_path (str): Path to save the visualization image.
        """
        # # Ensure height_map and mask are numpy arrays
        # height_map = height_map.cpu().numpy() if isinstance(height_map, torch.Tensor) else height_map
        # mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

        # # Create an RGBA image where invalid cells are black
        # height_map_normalized = (height_map - np.min(height_map[mask > 0])) / (np.max(height_map[mask > 0]) - np.min(height_map[mask > 0]))
        # height_map_normalized = np.clip(height_map_normalized, 0, 1)  # Normalize to [0, 1]
        # colormap_func = plt.cm.get_cmap(colormap)
        # height_map_colored = colormap_func(height_map_normalized)  # Apply colormap
        # height_map_colored = (height_map_colored[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB

        # # Set invalid cells (mask == 0) to black
        # height_map_colored[mask == 0] = [0, 0, 0]

        # # Display the visualization
        # plt.figure(figsize=(10, 6))
        # plt.imshow(height_map_colored)
        # plt.axis()
        # plt.title('Height Map Visualization')
        # plt.tight_layout()
        # cbar = plt.colorbar()
        # cbar.set_label('cm')
        # max = height_map[mask > 0].max()
        # min = height_map[mask > 0].min()

        # def to_cm(tick_val, pos=None):
        #     # tick_val is in [0, 255]; convert to cm proportionally to the original range
        #     return (tick_val / 255.0) * (max - min) + min
        

        # from matplotlib.ticker import FuncFormatter

        # cbar.formatter = FuncFormatter(lambda v, pos: f"{to_cm(v):.2f}")
        # cbar.update_ticks()


        # # Save the visualization
        # plt.savefig(save_path, dpi=300)
        # #plt.show()
        # ##print(f"Height map visualization saved to {save_path}")

    # Convert to numpy
        if isinstance(height_map, torch.Tensor):
            height_map = height_map.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        # Mask invalid values
        height_map_vis = height_map.astype(np.float32).copy()
        height_map_vis[mask == 0] = np.nan#height_map_vis = np.ma.masked_where(mask == 0, height_map)

        # Compute valid range (in cm)
        vmin = height_map_vis.min()
        vmax = height_map_vis.max()

        # Colormap with black for invalid
        cmap_obj = plt.cm.get_cmap(colormap).copy()
        cmap_obj.set_bad(color="black")
        # Plot
        plt.figure(figsize=(10, 6))
        im = plt.imshow(
            height_map_vis,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax
        )
        plt.axis("off")
        plt.title("Height Map (valid only)")

        # Colorbar in cm
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label("cm")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

visualize_height_map_and_mask(dummy_height_map, dummy_mask, colormap='plasma', save_path='Heightmap/dummy_height_map_visualization.png')