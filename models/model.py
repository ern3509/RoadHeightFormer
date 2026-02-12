from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from typing import List
from models.ele_head import *
import math
from .efficientnet import efficientnet_feature
from utils.experiment import save_feature_map
from .patch2feature import _make_scratch, _make_fusion_block, patch2feature, easy_transition_layer
import warnings

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

class Elevation(nn.Module):
    def __init__(self, stereo,  num_grids, ele_range, cla_res, regression=False, backbone = 'efficientnet', normalize=False):
        super(Elevation, self).__init__()
        self.stereo = stereo
        self.num_grids_x, self.num_grids_y, self.num_grids_z = num_grids
        self.ele_range = ele_range   # in meter
        self.regression = regression
        self.backbone = backbone

        self.cla_res = cla_res
        self.num_classes = int(2 * self.ele_range*100 / self.cla_res)
        ele_values = -torch.arange(self.num_classes, dtype=torch.float32, device='cuda')*self.cla_res + self.ele_range*100 - self.cla_res/2
        self.ele_values = ele_values.reshape(1, self.num_classes, 1, 1)
        
        self.patch2feat = True
        # Replace efficientnet_feature with DINOv2 backbone

        if backbone == 'DepthAnything3' :
            model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
            #print("model_depthAnything3", model)
            encoder = model.model.backbone
            # print("encoder", encoder)
            self.feature_extraction = encoder
            if self.patch2feat:
                self.transition_layer = patch2feature(embed_dim=768, patch_size=14, output_dim = 128,
                                                      out_channels = [48, 96, 192, 384])
            else:
                self.transition_layer = easy_transition_layer(embed_dim=768, patch_size=14, out_channels=128)
            self.feat_channel = 128  # DepthAnything3 feature map channel count

            self.projection = nn.Linear(3072, 64)   #get back to the vanilla feature channel
        
        else:
            self.feature_extraction = efficientnet_feature(self.stereo) 
            self.feat_channel = self.feature_extraction.feat_channel
        if regression:
            #regressor for regression
            print("Using regression head")

            self.ele_head = EleReg2D(self.feat_channel, num_grids, normalize)

        else:
            if self.stereo:
                #  regressor for stereo
                self.ele_head = EleCla3D(self.feat_channel, num_grids, self.num_classes)
            else:
                #  regressor for mono
                self.ele_head = EleCla2D(self.feat_channel, num_grids, self.num_classes)

        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
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

    def forward(self, imgs_left, proj_index_left, *args):
        # proj_index: [num_samples, 2, num_grids_z*num_grids_x*num_grids_y]
        if self.backbone == 'DepthAnything3':
            #with torch.no_grad():
            print("start feature extraction with DepthAnything3 backbone")
            # add a dimension to input image
            imgs_left = imgs_left.unsqueeze(1) # [B, 1, C, H, W] 
            features, aux_features = self.feature_extraction(imgs_left)  #tuple of pair of feautures (patch embed and 1dfeature vector)
            B, S, N, C = features[0][0].shape
            #print("Extracted features before projection shape:", features[0][0].shape)
            features = [feat[0].reshape(B*S, N, C) for feat in features]
            #print_types(features)
            #print("features before projection", len(features_left[1]), features_left[1].shape, features_left[0][3][1].shape)
            if self.patch2feat:
                features_left = self.transition_layer(features, 952, 518, 238, 130)   #B*S, C, 952, 518
            else:
                features_left = self.transition_layer(features, 952, 518, 238, 130)   #B, C, 952, 518
            print("Extracted features shape:", features_left.shape)
        
        else:
            print("start feature extraction with EfficientNet backbone")
            features_left = self.feature_extraction(imgs_left)
            #print("Extracted features shape:", features_left.shape)
            
        B, C, H, W = features_left.shape
        features_left = features_left.reshape(B, C, -1)
        linear_indices = proj_index_left[:, 1, :] * W + proj_index_left[:, 0, :]

        print("linear indices:" ,linear_indices.shape)
        voxel_feat_left = features_left.gather(dim=2, index=linear_indices.unsqueeze(1).expand(-1, C, -1))
        #print("voxel feet after gather shape:", voxel_feat_left.shape)

        voxel_feat_left = voxel_feat_left.reshape(B, C, self.num_grids_z, self.num_grids_x, self.num_grids_y)
        print("voxel feat valid", torch.sum(torch.isnan(voxel_feat_left)))
        print("range of voxel_feat_left:", voxel_feat_left.min().item(), voxel_feat_left.max().item())
        save_feature_map(voxel_feat_left[0, 0, :, :, self.num_grids_y//2], "voxel_feature_map_left.png")

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
        out_channels: int | None = None,
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

        print("shape of one patch feature: ",feats[0].shape)

        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size

        resized_feats = []

        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]   # remove CLS start index is 1
            #how does layer norm work here? what should be the input shape?
            #x = self.norm(x) #[Batch, sequence length, Number of patches, channels]

            # tokens → feature map
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw) # [B, C, ph, pw]
            save_feature_map(x[0, 0, :, :], f"patch_viz {stage_idx} .png" )
            # project channels
            x = self.projects[stage_idx](x) # [B, out_channels, ph, pw]

            print(f"after projection shape at stage {stage_idx}:", x.shape) 

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
        save_feature_map(fused[0, 0, :, :], "fused_feature_map.png")
        print("fused after conv shape:", fused.shape) #[B, out_channels, H, W]
        print("Fused feature map border values:", fused[0, :, 0, :], fused[0, :, -1, :])
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
        print("shape of one patch feature: ",feats[-1].shape)
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        if len(feats) > 1:
            for i in range(len(ops)):
                features.append(feats[-1])
            for i in range(len(features)):
                features[i] = ops[i](features[i])
                features[i] = self.projects[i](features[i])
                print(f"feature shape after fpn {i + 1} :", features[i].shape)
                features[i] = F.interpolate(
                    features[i],
                    size=(H_out, W_out),
                    mode="bilinear",
                    align_corners=False,
                )  # [B, out_channels, H, W]
            
            features_fused = torch.stack(features, dim=0).sum(dim=0)
            print("features fused shape before conv:", features_fused.shape)

        return features_fused
            
            

           

