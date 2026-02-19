import torch
import torch.nn as nn
from typing import List, Sequence, Tuple, Union
from .DPT_utils import Permute
from utils.experiment import save_feature_map
from models.submodule import *


class patch2feature(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        output_dim: int = 256, 
        patch_size: int = 16,
        out_channels:  Sequence[int] = (256, 512, 1024, 1024),
        intermediate_layer_idx=(0, 1, 2, 3),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.intermediate_layer_idx = intermediate_layer_idx

        # keep channel size unless explicitly changed
        self.out_channels = out_channels or embed_dim

        self.projects = nn.ModuleList(
            nn.Conv2d(embed_dim, oc, kernel_size=1, stride=1, padding=0, bias=True) for oc in out_channels
            #[convbn(embed_dim, oc, kernel_size=1, stride=1, pad=0, dilation=1) for oc in out_channels]
        )

        # -------------------- Spatial re-size (align to common scale before fusion) --------------------
        # Design consistent with original: relative to patch grid (x4, x2, x1, /2)
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0    #ph*pw -> 4*ph*4*pw
                ),
                nn.ConvTranspose2d(
                    out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0   #ph*pw -> 2*ph*2*pw
                ),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1), #ph*pw -> (1/2)*ph*(1/2)*pw
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

         # -------------------- scratch: stage adapters + main fusion chain --------------------
        self.scratch = _make_scratch(list(out_channels), output_dim, expand=False)
        self.scratch.output_conv1 = nn.Conv2d(
            output_dim, output_dim, kernel_size=3, stride=1, padding=1
        )
        #out_conv2 in dpt module
        self.out_norm = nn.Sequential(  
            Permute((0, 2, 3, 1)), nn.LayerNorm(output_dim), Permute((0, 3, 1, 2)),
        )

        # Main fusion chain
        self.scratch.refinenet1 = _make_fusion_block(output_dim, inplace=False)
        self.scratch.refinenet2 = _make_fusion_block(output_dim, inplace=False)
        self.scratch.refinenet3 = _make_fusion_block(output_dim, inplace=False)
        self.scratch.refinenet4 = _make_fusion_block(
            output_dim, has_residual=False, inplace=False
        )

    def _fuse(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        4-layer top-down fusion, returns finest scale features (after fusion, before neck1).
        """
        l1, l2, l3, l4 = feats

        l1_rn = self.scratch.layer1_rn(l1)
        #visualize_value(l1_rn, "l1_rn_feat.png")
       #print("valid elements L1 after the conv", torch.sum(torch.isnan(l1_rn)), torch.sum(torch.isinf(l1_rn)))
        l2_rn = self.scratch.layer2_rn(l2)
       #print("valid elements L2 after the conv", torch.sum(torch.isnan(l2_rn)), torch.sum(torch.isinf(l2_rn)))
        #visualize_value(l2_rn, "l2_rn_feat.png")
        l3_rn = self.scratch.layer3_rn(l3)
       #print("valid elements L3 after the conv", torch.sum(torch.isnan(l3_rn)), torch.sum(torch.isinf(l3_rn)))
        l4_rn = self.scratch.layer4_rn(l4)
        #visualize_value(l4_rn, "l4_rn_feat.png")

        # 4 -> 3 -> 2 -> 1
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
       #print("valid elements after the conv", torch.sum(torch.isnan(out)), torch.sum(torch.isinf(out)))
        #visualize_value(out, "l4_out.png")
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        #visualize_value(out, "l3_out.png")

        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        #visualize_value(out, "l2_out.png")

       #print("valid elements after the conv", torch.sum(torch.isnan(out)), torch.sum(torch.isinf(out)))
        out = self.scratch.refinenet1(out, l1_rn)
        return out
    
    def forward(self, feats: List[torch.Tensor],
                H: int,
                W: int, 
                h_out,
                w_out,
                patch_start_idx: int = 0) -> torch.Tensor:
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]  # [B*S, N_patch, C]
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw)  # [B*S, C, ph, pw] C=768
           #print("valid input data", torch.sum(torch.isnan(x)), torch.sum(torch.isinf(x)))
            x = self.projects[stage_idx](x)  # [B*S, C, ph, pw] C here is 48, 96, 192, 384
           #print("valid input projected data", torch.sum(torch.isnan(x)), torch.sum(torch.isinf(x)))

            x = self.resize_layers[stage_idx](x)  # Align scale
           #print("valid resized input data", torch.sum(torch.isnan(x)), torch.sum(torch.isinf(x)))
            #visualize_value(x, f"projected_and_resized{stage_idx}.png")
            resized_feats.append(x)

        # 2) Fusion pyramid (main branch only)
        fused = self._fuse(resized_feats)
       #print("valid fused data before interpolation", torch.sum(torch.isnan(fused)), torch.sum(torch.isinf(fused)))
        #visualize_value(fused, "fuse_before_outconv.png")
        
        fused = self.scratch.output_conv1(fused)
       #print("valid fused data", torch.sum(torch.isnan(fused)), torch.sum(torch.isinf(fused)))
        # Get index of largest value
        #visualize_value(fused, "fused_beforeinterpolation.png")

        fused = custom_interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)
       #print("valid interpolate fused data", torch.sum(torch.isnan(fused)), torch.sum(torch.isinf(fused)))
        #visualize_value(fused, "fused_afterinterpolation.png")

       #print("fused shape after interpolation:", fused.shape)

        #fused = self.out_norm(fused)
        return fused

def _make_scratch(
    in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False
) -> nn.Module:
    scratch = nn.Module()
    # Optional expansion by stage
    c1 = out_shape
    c2 = out_shape * (2 if expand else 1)
    c3 = out_shape * (4 if expand else 1)
    c4 = out_shape * (8 if expand else 1)

    scratch.layer1_rn = nn.Conv2d(in_shape[0], c1, 3, 1, 1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], c2, 3, 1, 1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], c3, 3, 1, 1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], c4, 3, 1, 1, bias=False, groups=groups)
    return scratch

def _make_fusion_block(
    features: int,
    size: Tuple[int, int] = None,
    has_residual: bool = True,
    groups: int = 1,
    inplace: bool = False,
) -> nn.Module:
    return FeatureFusionBlock(
        features=features,
        activation=nn.ReLU(inplace=inplace),
        deconv=False,
        bn=True,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )

class ResidualConvUnit(nn.Module):
    """Lightweight residual convolution block for fusion"""

    def __init__(self, features: int, activation: nn.Module, bn: bool, groups: int = 1) -> None:
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        if bn:
           #print("defining normalisation")
            self.norm1 = nn.BatchNorm2d(features)
            self.norm2 = nn.BatchNorm2d(features)
        else:
            self.norm1 = None
            self.norm2 = None
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
           #print("normalising")
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)

# -----------------------------------------------------------------------------
# Interpolation (safe interpolation, avoid INT_MAX overflow)
# -----------------------------------------------------------------------------
def custom_interpolate(
    x: torch.Tensor,
    size: Union[Tuple[int, int], None] = None,
    scale_factor: Union[float, None] = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Safe interpolation implementation to avoid INT_MAX overflow in torch.nn.functional.interpolate.
    """
    if size is None:
        assert scale_factor is not None, "Either size or scale_factor must be provided."
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736
    total = size[0] * size[1] * x.shape[0] * x.shape[1]

    if total > INT_MAX:
        chunks = torch.chunk(x, chunks=(total // INT_MAX) + 1, dim=0)
        outs = [
            nn.functional.interpolate(c, size=size, mode=mode, align_corners=align_corners)
            for c in chunks
        ]
        return torch.cat(outs, dim=0).contiguous()

    return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)


class FeatureFusionBlock(nn.Module):
    """Top-down fusion block: (optional) residual merge + upsampling + 1x1 contraction"""

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        deconv: bool = False,
        bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
        size: Tuple[int, int] = None,
        has_residual: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.size = size
        self.has_residual = has_residual
        self.resConfUnit1 = (
            ResidualConvUnit(features, activation, bn, groups=groups) if has_residual else None
        )
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=groups)

        out_features = (features // 2) if expand else features
        self.out_conv = convbn(features, out_features, 1, 1, 0, 1)
        
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs: torch.Tensor, size: Tuple[int, int] = None) -> torch.Tensor:  # type: ignore[override]
        """
        xs:
          - xs[0]: Top branch input
          - xs[1]: Lateral input (can do residual addition with top branch)
        """
        y = xs[0]
        if self.has_residual and len(xs) > 1 and self.resConfUnit1 is not None:
            y = self.skip_add.add(y, self.resConfUnit1(xs[1]))

        y = self.resConfUnit2(y)

        # Upsampling
        if (size is None) and (self.size is None):
            up_kwargs = {"scale_factor": 2}
        elif size is None:
            up_kwargs = {"size": self.size}
        else:
            up_kwargs = {"size": size}

        y = custom_interpolate(y, **up_kwargs, mode="bilinear", align_corners=self.align_corners)
        y = self.out_conv(y)
        return y




class easy_transition_layer(nn.Module):
    def __init__(self,
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
        self.norm = nn.LayerNorm(embed_dim)
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
    

    def forward(self, feats: List[torch.Tensor], H: int, W: int, H_out: int, W_out: int) -> torch.Tensor:
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        x = feats[-1]  # [B*S, N_patch, C]
        x = self.norm(x)
       #print("valid input data", torch.sum(torch.isnan(x)), torch.sum(torch.isinf(x)))
        x = x.permute(0, 2, 1).reshape(B, C, ph, pw)  # [B*S, C, ph, pw]

        x = self.projects[-1](x)
       #print("valid projected input data", torch.sum(torch.isnan(x)), torch.sum(torch.isinf(x)))

        x = custom_interpolate(x, size=(H_out, W_out), mode="bilinear", align_corners=True)
       #print("valid interpolated input data", torch.sum(torch.isnan(x)), torch.sum(torch.isinf(x)))

        return x
    

def visualize_value(fused, name_of_file):
    x = fused
    idx = torch.argmax(x)

    # Convert to coordinates
    coords = torch.unravel_index(idx, x.shape)

   #print("Max value location:", name_of_file, coords)
    if x.shape[1] < 82:
        save_feature_map(fused[0, x.shape[1] - 1], name_of_file)
    else:
        save_feature_map(fused[0, 82], name_of_file)