from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.ele_head import *
import math
from .efficientnet import efficientnet_feature

import sys
sys.path.append('/home/f9ql00v/depth-anything3/Depth-Anything-3-main/src')


from depth_anything_3.api import DepthAnything3

class Elevation(nn.Module):
    def __init__(self, stereo,  num_grids, ele_range, cla_res, regression=False, backbone = 'efficientnet'):
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

        # Replace efficientnet_feature with DINOv2 backbone

        if backbone == 'DepthAnything3' :
            model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
            # print("model_depthAnything3", model)
            encoder = model.model.da3.backbone
            # print("encoder", encoder)
            self.feature_extraction = encoder
            self.feat_channel = 64  # DepthAnything3 feature map channel count

            self.projection = nn.Linear(3072, 64)   #get back to the vanilla feature channel
        
        else:
            self.feature_extraction = efficientnet_feature(self.stereo) 
            self.feat_channel = self.feature_extraction.feat_channel
        if regression:
            #regressor for regression
            print("Using regression head")
            self.ele_head = EleReg2D(self.feat_channel, num_grids, normalize = False)

        else:
            if self.stereo:
                #  regressor for stereo
                self.ele_head = EleCla3D(self.feat_channel, num_grids, self.num_classes)
            else:
                #  regressor for mono
                self.ele_head = EleCla2D(self.feat_channel, num_grids, self.num_classes)

        for m in self.modules():
            print
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

    def _initialize_dinov2_backbone(self):
        # Load pretrained DINOv2 backbone
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        dinov2.eval()  # Set to evaluation mode
        dinov2.feat_channel = 768  # DINOv2 base feature map channel count
        return dinov2
    
    def forward(self, imgs_left, proj_index_left, *args):
        # proj_index: [num_samples, 2, num_grids_z*num_grids_x*num_grids_y]
        if self.backbone == 'DepthAnything3':
            with torch.no_grad():
                print("start feature extraction with DepthAnything3 backbone")
                # add a dimension to input image
                imgs_left = imgs_left.unsqueeze(1) # [B, 1, C, H, W] 
                features_left = self.feature_extraction(imgs_left)  #tuple of pair of feautures (patch embed and 1dfeature vector)
                print("Extracted features shape:", features_left[0][0][0].shape)
                features_left = features_left[0][0][0]  # take the patch embed features
                #print("features before projection", len(features_left[1]), features_left[1].shape, features_left[0][3][1].shape)
            features_left = self.projection(features_left)
            print("Extracted features shape:", features_left.shape)
        
        else:
            print("start feature extraction with EfficientNet backbone")
            features_left = self.feature_extraction(imgs_left)
            print("Extracted features shape:", features_left.shape)
            
        B, C, H, W = features_left.shape
        features_left = features_left.reshape(B, C, -1)
        linear_indices = proj_index_left[:, 1, :] * W + proj_index_left[:, 0, :]

        voxel_feat_left = features_left.gather(dim=2, index=linear_indices.unsqueeze(1).expand(-1, C, -1))

        voxel_feat_left = voxel_feat_left.reshape(B, C, self.num_grids_z, self.num_grids_x, self.num_grids_y)

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
