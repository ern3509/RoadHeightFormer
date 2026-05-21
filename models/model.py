# from __future__ import print_function
# import torch
# import torch.nn as nn
# import torch.utils.data
# import torch.nn.functional as F
# from typing import List
# from models.ele_head import *
# import math
# from .efficientnet import efficientnet_feature
# from utils.experiment import save_feature_map
# from .patch2feature import _make_scratch, _make_fusion_block, patch2feature, easy_transition_layer
# import warnings
# from contextlib import nullcontext
# from models.submodule import *
# from sklearn.decomposition import PCA
# import pandas as pd
# import cv2, os
# from PIL import Image

# import sys
# sys.path.append('/home/f9ql00v/depth-anything3/Depth-Anything-3-main/src')


# from depth_anything_3.api import DepthAnything3

# def print_types(obj, indent=0):
#     prefix = "  " * indent

#     if isinstance(obj, (list, tuple)):
#         print(f"{prefix}{type(obj).__name__} (len={len(obj)})")
#         for i, item in enumerate(obj):
#            print(f"{prefix}  [{i}]:")
#            print_types(item, indent + 2)
#     else:
#        print(f"{prefix}{type(obj).__name__}")

# class Elevation(nn.Module):
#     def __init__(self, stereo,  num_grids, ele_range, cla_res, regression=False, backbone = 'efficientnet', normalize=False, pred_dim =256, dino = "small"):
#         super(Elevation, self).__init__()

#         self.stereo = stereo
#         self.num_grids_x, self.num_grids_y, self.num_grids_z = num_grids
#         self.ele_range = ele_range   # in meter
#         self.regression = regression
#         self.backbone = backbone
#         self.context = torch.no_grad() if 'frozen' in self.backbone else nullcontext()
#         print(self.context)
#         self.cla_res = cla_res
#         self.num_classes = int(2 * self.ele_range*100 / self.cla_res)    #the smaller the clas_res, the more classes we have
#         ele_values = -torch.arange(self.num_classes, dtype=torch.float32, device='cuda')*self.cla_res + self.ele_range*100 - self.cla_res/2 #19.75, 19.25, 18.75, ..., -18.75, -19.25, -19.75
#         self.ele_values = ele_values.reshape(1, self.num_classes, 1, 1)
        
#         self.patch2feat = True
#         # Replace efficientnet_feature with DINOv2 backbone
#         model_name = ""
#         out_channels = None


#         if 'DepthAnything3' in backbone :
#             if dino == "small":
#                 embed_dim = 384
#                 model_name = "depth-anything/DA3-SMALL"
#                 out_channels = [48, 96, 192, 384]
#             elif dino == "Large":
#                 print("large model!!!")
#                 embed_dim = 3072
#                 model_name = "depth-anything/DA3NESTED-GIANT-LARGE"
#                 out_channels = [256, 512, 1024, 1024]
#             model = DepthAnything3.from_pretrained(model_name)


#             print("model_depthAnything3", model)
#             encoder = model.model.backbone if dino == "small" else model.model.da3.backbone
#             ##print("encoder", encoder)

#             #########################
#             #temporal change of backbone
#             encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

#             #################
#             self.feature_extraction = encoder
#             if self.patch2feat:
#                 self.transition_layer = patch2feature(embed_dim=embed_dim, patch_size=14, output_dim = pred_dim,
#                                                       out_channels = out_channels)
#             else:
#                 self.transition_layer = easy_transition_layer(embed_dim=embed_dim, patch_size=14, out_channels = pred_dim)
#             self.feat_channel = pred_dim
        
#         else:
#             self.feature_extraction = efficientnet_feature(self.stereo) 
#             self.feat_channel = self.feature_extraction.feat_channel
#         if regression:
#             #regressor for regression
#            #print("Using regression head")

#             self.ele_head = EleReg2D(self.feat_channel, num_grids, normalize)

#         else:
#             if self.stereo:
#                 #  regressor for stereo
#                 self.ele_head = EleCla3D(self.feat_channel, num_grids, self.num_classes)
#             else:
#                 #  regressor for mono
#                 self.ele_head = EleCla2D(self.feat_channel, num_grids, self.num_classes)

#         for m in self.modules():
            
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.Conv3d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()

#     def forward(self, imgs_left, proj_index_left, *args):
#         pca = PCA(n_components= 3)
#         # proj_index: [num_samples, 2, num_grids_z*num_grids_x*num_grids_y]
#         if 'DepthAnything3' in self.backbone:
#             with self.context:
#            #print("start feature extraction with DepthAnything3 backbone")
#             # add a dimension to input image
#                 imgs_left = imgs_left  #.unsqueeze(1) # [B, 1, C, H, W] 
#                 print(imgs_left.shape)
#                 #temporal features_dict = self.feature_extraction.forward_features(imgs_left)
#                 #temporal features = features_dict['x_norm_patchtokens']

#                 features = self.feature_extraction.get_intermediate_layers(imgs_left,  [5, 7, 9, 11])  #tuple of pair of feautures (patch embed and 1dfeature vector) # , aux_features
                
#                 #features_fresh = self.feature_extraction.pretrained.forward_features(imgs_left)
#                 #features_fresh =  features_fresh['x_norm_patchtokens']
#                 print(len(features))
#                 print(features[0].shape)
#                 B, N, C = features[0].shape
#                 #print("Extracted features before projection shape:", features[0][0].shape)
#                 features_ = [feat.reshape(B, N, C) for feat in features]

#                 # test_feat =  features_[-1][0].cpu().detach() #features[0].cpu().detach() #-1 last element of the list 0 for first element of the batch

#                 # #test_pca = pca.fit_transform(test_feat)
#                 # pca.fit(test_feat)
#                 # pca_features = pca.transform(test_feat)
#                 # print("PCA_features:", pca_features.shape)
#                 # pca_features = pca_features.reshape(38, 68, 3)

#                 # img = pca_features[:, :, 0]
#                 # img = (img - img.min()) / (img.max() - img.min() + 1e-8)
#                 # img = (img * 255).astype(np.uint8)

#                 # Image.fromarray(img).save("pca_first_component.png")

#                 #print_types(features)
#                 #print("features before projection", len(features_left[1]), features_left[1].shape, features_left[0][3][1].shape)
#             if self.patch2feat:
#                 features_left = self.transition_layer(features_, 532, 952, 133, 238)   #B*S, C, 952, 518
#             else:
#                 features_left = self.transition_layer(features_, 532, 952, 133, 238)   #B, C, 952, 518
#            #print("Extracted features shape:", features_left.shape)
        
#         else:
#            #print("start feature extraction with EfficientNet backbone")
#             features_left = self.feature_extraction(imgs_left)
#             #print("Extracted features shape:", features_left.shape)
            
#         B, C, H, W = features_left.shape
#         #visualize PCA
#         #pca = PCA(n_components= 3)
#         features_left_cpu = features_left[0].cpu().detach().permute(1, 2, 0).reshape(-1, C)
#         """ print("feature_left_shape:", features_left_cpu.shape)
#         features_pca = pca.fit_transform(features_left_cpu).reshape((H, W, 3))
#         print("pca_features: ", features_pca.shape)
#         if not os.path.exists("pca_of_features.png"):
#             cv2.imwrite("pca_of_features.png", features_pca)
#         img = features_pca[:, :, 0]
#         img = (img - img.min()) / (img.max() - img.min() + 1e-8)
#         img = (img * 255).astype(np.uint8)

#         Image.fromarray(img).save("pca_first_component_final.png") """


#         B, C, H, W = features_left.shape
#         features_left = features_left.reshape(B, C, -1)
#         linear_indices = proj_index_left[:, 1, :] * W + proj_index_left[:, 0, :]

#        #print("linear indices:" ,linear_indices.shape)
#         #voxel_feat_left = F.interpolate(features_left, (self.num_grids_z, self.num_grids_x*self.num_grids_y), mode='bilinear')
#         voxel_feat_left = features_left.gather(dim=2, index=linear_indices.unsqueeze(1).expand(-1, C, -1))


#         #print("voxel feet after gather shape:", voxel_feat_left.shape)

#         voxel_feat_left = voxel_feat_left.reshape(B, C, self.num_grids_z, self.num_grids_x, self.num_grids_y)
#        #print("voxel feat valid", torch.sum(torch.isnan(voxel_feat_left)))
#        #print("range of voxel_feat_left:", voxel_feat_left.min().item(), voxel_feat_left.max().item())
#         #save_feature_map(voxel_feat_left[0, 0, :, :, self.num_grids_y//2], "voxel_feature_map_left.png")

#         # proj_index: [num_samples, 2, num_grids_z*num_grids_x*num_grids_y]
#         if self.stereo:
#             imgs_right, proj_index_right = args[0], args[1]
#             features_right = self.feature_extraction(imgs_right)
#             features_right = features_right.reshape(B, C, -1)
#             linear_indices = proj_index_right[:, 1, :] * W + proj_index_right[:, 0, :]
#             voxel_feat_right = features_right.gather(dim=2, index=linear_indices.unsqueeze(1).expand(-1, C, -1))
#             voxel_feat_right = voxel_feat_right.reshape(B, C, self.num_grids_z, self.num_grids_x, self.num_grids_y)

#             voxel_feature = voxel_feat_left * voxel_feat_right
#             voxel_feature = voxel_feature.permute(0, 1, 4, 2, 3)  # [B, C, Y, Z, X]
#         else:
#             voxel_feature = voxel_feat_left    # [B, C, Z, X, Y]

#         ele_pred = self.ele_head(voxel_feature)    # [B, num_class, Z, X]   without softmax

#         if (not self.training) & (not self.regression):
#             ele_pred = F.softmax(ele_pred, dim=1)
#             ele_pred = torch.sum(ele_pred * self.ele_values, dim=1)

#             # pred_class = torch.max(ele_pred.data, 1)[1]
#             # ele_pred = self.ele_values[pred_class.type(torch.long)]

#         return ele_pred


# class DinoV2SpatialDecoder(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         patch_size: int = 14,
#         out_channels: int | None = None,
#         intermediate_layer_idx=(0, 1, 2, 3),
#     ):
#         super().__init__()

#         self.patch_size = patch_size
#         self.intermediate_layer_idx = intermediate_layer_idx

#         # keep channel size unless explicitly changed
#         self.out_channels = out_channels or embed_dim

#         # per-scale projection
#         self.projects = nn.ModuleList([nn.Sequential(
#             nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(embed_dim // 4, embed_dim // 8, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(embed_dim // 8, self.out_channels, kernel_size=1))
            
#             for _ in intermediate_layer_idx
#         ])

#         # simple fusion (sum)
#         self.fuse = nn.Conv2d(
#             self.out_channels,
#             self.out_channels,
#             kernel_size=3,
#             padding=1,
#         )

#         self.fpn1 = nn.Sequential(
#             nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
#             nn.SyncBatchNorm(embed_dim),
#             nn.GELU(),
#             nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
#         )

#         self.fpn2 = nn.Sequential(
#             nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
#         )

#         self.fpn3 = nn.Identity()

#         self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.norm = nn.LayerNorm(embed_dim)


#     def forwardää(
#         self,
#         feats: List[torch.Tensor],
#         H: int,
#         W: int,
#         H_out: int,
#         W_out: int,
#         patch_start_idx: int = 0,
#     ) -> torch.Tensor:
#         """
#         Args:
#             feats: list of 4 tensors, each [B, N, C]
#             H, W: target spatial resolution

#         Returns:
#             Tensor: (B, C, H, W)
#         """
#         assert len(feats) == len(self.intermediate_layer_idx)

#        #print("shape of one patch feature: ",feats[0].shape)

#         B, _, C = feats[0].shape
#         ph, pw = H // self.patch_size, W // self.patch_size

#         resized_feats = []

#         for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
#             x = feats[take_idx][:, patch_start_idx:]   # remove CLS start index is 1
#             #how does layer norm work here? what should be the input shape?
#             #x = self.norm(x) #[Batch, sequence length, Number of patches, channels]

#             # tokens → feature map
#             x = x.permute(0, 2, 1).reshape(B, C, ph, pw) # [B, C, ph, pw]
#             #save_feature_map(x[0, 0, :, :], f"patch_viz {stage_idx} .png" )
#             # project channels
#             x = self.projects[stage_idx](x) # [B, out_channels, ph, pw]

#            #print(f"after projection shape at stage {stage_idx}:", x.shape) 

#             # resize to target resolution
#             x = F.interpolate(
#                 x,
#                 size=(H_out, W_out),
#                 mode="bilinear",
#                 align_corners=False,
#             ) # [B, out_channels, H, W]

#             resized_feats.append(x)

#         # fuse multi-scale features
#         fused = torch.stack(resized_feats, dim=0) # [4, B, out_channels, H, W]
#         fused = fused.sum(dim=0)
#         fused = self.fuse(fused) 
#         #save_feature_map(fused[0, 0, :, :], "fused_feature_map.png")
#        #print("fused after conv shape:", fused.shape) #[B, out_channels, H, W]
#        #print("Fused feature map border values:", fused[0, :, 0, :], fused[0, :, -1, :])
#         return fused

#     def forward(
#             self,
#             feats: List[torch.Tensor],
#             H: int,
#             W: int,
#             H_out: int,
#             W_out: int,
#             patch_start_idx: int = 0,
#     ) -> torch.Tensor:
#         features =[]
#         feats = [feats[i][:, patch_start_idx:] for i in range(len(feats))]
#         ph, pw = H // self.patch_size, W // self.patch_size 
#         feats = [
#             feat.permute(0, 2, 1).reshape(feat.shape[0], feat.shape[2], ph, pw) 
#             for feat in feats
#         ]  # [B, C, ph, pw]
#        #print("shape of one patch feature: ",feats[-1].shape)
#         ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
#         if len(feats) > 1:
#             for i in range(len(ops)):
#                 features.append(feats[-1])
#             for i in range(len(features)):
#                 features[i] = ops[i](features[i])
#                 features[i] = self.projects[i](features[i])
#                #print(f"feature shape after fpn {i + 1} :", features[i].shape)
#                 features[i] = F.interpolate(
#                     features[i],
#                     size=(H_out, W_out),
#                     mode="bilinear",
#                     align_corners=False,
#                 )  # [B, out_channels, H, W]
            
#             features_fused = torch.stack(features, dim=0).sum(dim=0)
#            #print("features fused shape before conv:", features_fused.shape)

#         return features_fused
            
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from typing import List, Optional
from models.ele_head import *
import math
from .efficientnet import efficientnet_feature
from utils.experiment import save_feature_map
from .patch2feature import _make_scratch, _make_fusion_block, patch2feature, easy_transition_layer, make_pca, DinoUpsampler
import warnings
import cv2
from contextlib import nullcontext
from sklearn.decomposition import PCA
import numpy as np
import os

import sys
sys.path.append('/home/f9ql00v/depth-anything3/Depth-Anything-3-main/src')


from depth_anything_3.api import DepthAnything3

def print_types(obj, indent=0):
    prefix = "  " * indent

    if isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__} (len={len(obj)})")
        for i, item in enumerate(obj):
           print(f"{prefix}  [{i}]:")
           print_types(item, indent + 2)
    else:
       print(f"{prefix}{type(obj).__name__}")

def visualize_encoder_pca(features, save_path, patch_size=14, img_hw=None, layer_idx=-1, batch_idx=0):
    """
    Visualize encoder features via PCA -> RGB image.

    Args:
        features: encoder output. Supported shapes:
            - Tensor [B, C, H, W]                       (e.g. EfficientNet)
            - Tensor [B, N, C] with N = ph*pw (+ CLS)   (e.g. DINO single layer)
            - List/tuple of [B, N, C] tensors           (e.g. DINO intermediate layers)
        save_path:  where to write the PNG.
        patch_size: ViT patch size (only used for token-shaped features).
        img_hw:     (H, W) of the input image, required for token-shaped features
                    so we can recover the spatial grid (ph = H // patch_size).
        layer_idx:  which layer to visualize when `features` is a list.
        batch_idx:  which sample of the batch to visualize.
    """
    if isinstance(features, (list, tuple)):
        feat = features[layer_idx]
    else:
        feat = features

    feat = feat.detach().float().cpu()

    if feat.dim() == 4:
        f = feat[batch_idx].permute(1, 2, 0)
    elif feat.dim() == 3:
        assert img_hw is not None, "img_hw=(H, W) required for token-shaped features"
        H, W = img_hw
        ph, pw = H // patch_size, W // patch_size
        f = feat[batch_idx]
        if f.shape[0] == ph * pw + 1:
            f = f[1:]
        elif f.shape[0] != ph * pw:
            raise ValueError(
                f"Token count {f.shape[0]} does not match ph*pw={ph*pw} (+CLS)"
            )
        f = f.reshape(ph, pw, -1)
    else:
        raise ValueError(f"Unsupported feature shape: {tuple(feat.shape)}")

    H, W, C = f.shape
    flat = f.reshape(-1, C).numpy()

    pca = PCA(n_components=3)
    rgb = pca.fit_transform(flat).reshape(H, W, 3)
    mn = rgb.min(axis=(0, 1), keepdims=True)
    mx = rgb.max(axis=(0, 1), keepdims=True)
    rgb = (rgb - mn) / (mx - mn + 1e-8)
    rgb = (rgb * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, rgb[:, :, ::-1])
    return rgb


class Elevation(nn.Module):
    def __init__(self, stereo,  num_grids, ele_range, cla_res, regression=False, backbone = 'efficientnet', normalize=False, pred_dim =256, train_encoder=False):
        super(Elevation, self).__init__()
        self.stereo = stereo
        self.num_grids_x, self.num_grids_y, self.num_grids_z = num_grids
        self.ele_range = ele_range   # in meter
        self.regression = regression
        self.backbone = backbone
        self.train_encoder = train_encoder

        self.cla_res = cla_res
        self.num_classes = int(2 * self.ele_range*100 / self.cla_res)    #the smaller the clas_res, the more classes we have
        ele_values = -torch.arange(self.num_classes, dtype=torch.float32, device='cuda')*self.cla_res + self.ele_range*100 - self.cla_res/2 #19.75, 19.25, 18.75, ..., -18.75, -19.25, 19.75
        self.ele_values = ele_values.reshape(1, self.num_classes, 1, 1)
        
        self.patch2feat = True
        # Upsampler choice: 'patch2feature' (DPT-style, multi-stage fusion) or 'dino' (DinoUpsampler).
        self.upsampler_kind = 'patch2feature'
        # Replace efficientnet_feature with DINOv2 backbone
        self.patchsize = int(14)
        self._pca_viz_done = False
        if 'DepthAnything3' in backbone :
            model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
            #print("model_depthAnything3", model)
            encoder = model.model.backbone
            ##print("encoder", encoder)
            self.feature_extraction = encoder
            if self.patch2feat:
                self.transition_layer = patch2feature(
                        embed_dim=768, patch_size=14, output_dim=pred_dim,
                        out_channels=(48, 96, 192, 384),
                    )
            else:
                self.transition_layer = DinoUpsampler(
                    embed_dim=768, patch_size=14,
                    output_dim=pred_dim, num_layers=4,
                    upsample_factor=4,
                )
            self.feat_channel = pred_dim
        
        else:
            self.feature_extraction = efficientnet_feature(self.stereo) 
            self.feat_channel = self.feature_extraction.feat_channel
        if regression:
            #regressor for regression
           #print("Using regression head")

            self.ele_head = EleReg2D(self.feat_channel, num_grids, normalize)

        else:
            if self.stereo:
                #  regressor for stereo
                self.ele_head = EleCla3D(self.feat_channel, num_grids, self.num_classes)
            else:
                #  regressor for mono
                self.ele_head = EleCla2D(self.feat_channel, num_grids, self.num_classes)

        if 'DepthAnything3' in backbone:
            self.transition_layer.apply(self._init_weights)
            self.ele_head.apply(self._init_weights)
        else:
            # efficientnet branch (not pretrained)
            self.feature_extraction.apply(self._init_weights)
            self.ele_head.apply(self._init_weights)

    def _init_weights(self,m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        # elif isinstance(m, torch.nn.BatchNorm2d):
        #     m.eval()

    def forward(self, imgs_left, proj_index_left, *args):
        # proj_index: [num_samples, 2, num_grids_z*num_grids_x*num_grids_y]
        if 'DepthAnything3' in self.backbone:
            encoder_ctx = nullcontext() if self.train_encoder else torch.no_grad()
            with encoder_ctx:
           #print("start feature extraction with DepthAnything3 backbone")
            # add a dimension to input image
                B, C, W, H = imgs_left.shape
                imgs_left = imgs_left.unsqueeze(1) # [B, 1, C, H, W]
                #imgs_left = imgs_left.transpose(-2, -1)
                print("me", imgs_left.shape)
                features, _ = self.feature_extraction(imgs_left)  #tuple of pair of feautures (patch embed and 1dfeature vector)
                B, S, N, C = features[0][0].shape
                print("Extracted features before projection shape:", features[0][0].shape)
                features = [feat[0].reshape(B*S, N, C) for feat in features]
                #make_pca(features[-1].transpose(-1, -2).reshape(B*S, C, int(W/14), int(H/14)), "pca_before_projection.png", 768)
                #print_types(features)
                # if not self._pca_viz_done:
                #     visualize_encoder_pca(features, "pca_dino_encoder.png",
                #                           patch_size=self.patchsize, img_hw=(W, H), layer_idx=-1)
                #     self._pca_viz_done = True

            if self.patch2feat:
                features_left = self.transition_layer(features, W, H, int(W/4), int(H/4))   #B*S, C, 952, 518
            else:
                features_left = self.transition_layer(features, W, H, W/4, H/4)   #B, C, 952, 518
           #print("Extracted features shape:", features_left.shape)
        
        else:
           #print("start feature extraction with EfficientNet backbone")
            features_left = self.feature_extraction(imgs_left)
            #print("Extracted features shape:", features_left.shape)
            # if not self._pca_viz_done:
            #     visualize_encoder_pca(features_left, "pca_efficientnet_encoder.png")
            #     self._pca_viz_done = True

        B, C, H, W = features_left.shape
        self._last_features = features_left.detach()
        features_left = features_left.reshape(B, C, -1)
        linear_indices = proj_index_left[:, 1, :] * W + proj_index_left[:, 0, :]

        voxel_feat_left = features_left.gather(dim=2, index=linear_indices.unsqueeze(1).expand(-1, C, -1))
        #print("voxel feet after gather shape:", voxel_feat_left.shape)

        voxel_feat_left = voxel_feat_left.reshape(B, C, self.num_grids_z, self.num_grids_x, self.num_grids_y)
       #print("voxel feat valid", torch.sum(torch.isnan(voxel_feat_left)))
       #print("range of voxel_feat_left:", voxel_feat_left.min().item(), voxel_feat_left.max().item())
        #save_feature_map(voxel_feat_left[0, 0, :, :, self.num_grids_y//2], "voxel_feature_map_left.png")

        # proj_index: [num_samples, 2, num_grids_z*num_grids_x*num_grids_y]
        if self.stereo:
            imgs_right, proj_index_right = args[0], args[1]
            features_right = self.feature_extraction(imgs_right)
            features_right = features_right.reshape(B, C, -1)
            linear_indices = proj_index_right[:, 1, :] * W + proj_index_right[:, 0, :]
            voxel_feat_right = features_right.gather(dim=2, index=linear_indices.unsqueeze(1).expand(-1, C, -1))
            voxel_feat_right = voxel_feat_right.reshape(B, C, self.num_grids_z, self.num_grids_x, self.num_grids_y)

            voxel_feature = voxel_feat_left * voxel_feat_right
            voxel_feature = voxel_feature.permute(0, 1, 4, 2, 3)  # [B, C, Y, Z, X]
        else:
            voxel_feature = voxel_feat_left    # [B, C, Z, X, Y]

        ele_pred = self.ele_head(voxel_feature)    # [B, num_class, Z, X]   without softmax

        if (not self.training) & (not self.regression):
            ele_pred = F.softmax(ele_pred, dim=1)
            ele_pred = torch.sum(ele_pred * self.ele_values, dim=1)

            # pred_class = torch.max(ele_pred.data, 1)[1]
            # ele_pred = self.ele_values[pred_class.type(torch.long)]

        return ele_pred


class DinoV2SpatialDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_size: int = 14,
        out_channels: Optional[int] = None,
        intermediate_layer_idx=(0, 1, 2, 3),
    ):
        super().__init__()

        self.patch_size = patch_size
        self.intermediate_layer_idx = intermediate_layer_idx

        # keep channel size unless explicitly changed
        self.out_channels = out_channels or embed_dim

        # per-scale projection
        self.projects = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 8, self.out_channels, kernel_size=1))
            
            for _ in intermediate_layer_idx
        ])

        # simple fusion (sum)
        self.fuse = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            padding=1,
        )

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            nn.SyncBatchNorm(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )

        self.fpn3 = nn.Identity()

        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.norm = nn.LayerNorm(embed_dim)


    def forwardää(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        H_out: int,
        W_out: int,
        patch_start_idx: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            feats: list of 4 tensors, each [B, N, C]
            H, W: target spatial resolution

        Returns:
            Tensor: (B, C, H, W)
        """
        assert len(feats) == len(self.intermediate_layer_idx)

       #print("shape of one patch feature: ",feats[0].shape)

        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size

        resized_feats = []

        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]   # remove CLS start index is 1
            #how does layer norm work here? what should be the input shape?
            #x = self.norm(x) #[Batch, sequence length, Number of patches, channels]

            # tokens → feature map
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw) # [B, C, ph, pw]
            # save_feature_map(x[0, 0, :, :], f"patch_viz {stage_idx} .png" )
            # project channels
            x = self.projects[stage_idx](x) # [B, out_channels, ph, pw]

           #print(f"after projection shape at stage {stage_idx}:", x.shape) 

            # resize to target resolution
            x = F.interpolate(
                x,
                size=(H_out, W_out),
                mode="bilinear",
                align_corners=False,
            ) # [B, out_channels, H, W]

            resized_feats.append(x)

        # fuse multi-scale features
        fused = torch.stack(resized_feats, dim=0) # [4, B, out_channels, H, W]
        fused = fused.sum(dim=0)
        fused = self.fuse(fused) 
        # save_feature_map(fused[0, 0, :, :], "fused_feature_map.png")
       #print("fused after conv shape:", fused.shape) #[B, out_channels, H, W]
       #print("Fused feature map border values:", fused[0, :, 0, :], fused[0, :, -1, :])
        return fused

    def forward(
            self,
            feats: List[torch.Tensor],
            H: int,
            W: int,
            H_out: int,
            W_out: int,
            patch_start_idx: int = 0,
    ) -> torch.Tensor:
        features =[]
        feats = [feats[i][:, patch_start_idx:] for i in range(len(feats))]
        ph, pw = H // self.patch_size, W // self.patch_size 
        feats = [
            feat.permute(0, 2, 1).reshape(feat.shape[0], feat.shape[2], ph, pw) 
            for feat in feats
        ]  # [B, C, ph, pw]
       #print("shape of one patch feature: ",feats[-1].shape)
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        if len(feats) > 1:
            for i in range(len(ops)):
                features.append(feats[-1])
            for i in range(len(features)):
                features[i] = ops[i](features[i])
                features[i] = self.projects[i](features[i])
               #print(f"feature shape after fpn {i + 1} :", features[i].shape)
                features[i] = F.interpolate(
                    features[i],
                    size=(H_out, W_out),
                    mode="bilinear",
                    align_corners=False,
                )  # [B, out_channels, H, W]
            
            features_fused = torch.stack(features, dim=0).sum(dim=0)
           #print("features fused shape before conv:", features_fused.shape)

        return features_fused
            
            

           
