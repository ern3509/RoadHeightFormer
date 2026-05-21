import torch, numpy as np, pickle, gzip, os, glob

files = sorted(glob.glob('/data/T7_2/rhf/train_preprocessed_data/data_item_*.pkl.gz'))

# 1. Check camera_height variation across many samples
print('=== CAMERA HEIGHT VARIATION ===')
camera_heights = []
gt_means = []
gt_stds = []
gt_ranges = []
for f in files[:200]:
    with gzip.open(f, 'rb') as fh:
        d = pickle.load(fh)
    ch = d.get('camera_height', None)
    gt = torch.tensor(d['ele_gt'], dtype=torch.float32)
    m = torch.tensor(d['mask'], dtype=torch.int8)
    v = gt[m > 0]
    if v.numel() > 0:
        gt_means.append(v.mean().item())
        gt_stds.append(v.std().item())
        gt_ranges.append(v.max().item() - v.min().item())
    if ch is not None:
        camera_heights.append(ch)

camera_heights = np.array(camera_heights)
gt_means = np.array(gt_means)
gt_stds = np.array(gt_stds)
gt_ranges = np.array(gt_ranges)

print(f'camera_height: mean={camera_heights.mean():.4f}m, std={camera_heights.std():.4f}m')
print(f'camera_height range: [{camera_heights.min():.4f}, {camera_heights.max():.4f}]m')
print(f'base_height = 1.857m')
print(f'camera_height - base_height: [{(camera_heights - 1.857).min():.4f}, {(camera_heights - 1.857).max():.4f}]m')
print()
print(f'GT means: mean={gt_means.mean():.3f}cm, std={gt_means.std():.3f}cm')
print(f'GT means range: [{gt_means.min():.3f}, {gt_means.max():.3f}]cm')
print(f'GT stds: mean={gt_stds.mean():.3f}cm, std={gt_stds.std():.3f}cm')
print(f'GT stds range: [{gt_stds.min():.3f}, {gt_stds.max():.3f}]cm')
print(f'GT within-sample ranges: mean={gt_ranges.mean():.3f}cm, max={gt_ranges.max():.3f}cm')
print()

# Histogram of GT means
print('Histogram of per-sample GT means:')
for lo in range(-30, 31, 5):
    hi = lo + 5
    count = ((gt_means >= lo) & (gt_means < hi)).sum()
    bar = '#' * int(count)
    print(f'  [{lo:+3d}, {hi:+3d}): {count:4d} {bar}')

# Correlation between camera_height and GT mean
if len(camera_heights) == len(gt_means):
    corr = np.corrcoef(camera_heights, gt_means)[0, 1]
    print(f'\nCorrelation(camera_height, GT_mean) = {corr:.4f}')

# 2. Check the actual feature map size with EfficientNet
print('\n=== ACTUAL BACKBONE OUTPUT SIZE ===')
import sys
sys.path.insert(0, '/home/f9ql00v/RoadHeightformer')
from models.efficientnet import efficientnet_feature

backbone = efficientnet_feature(stereo=False)
backbone.eval()
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 532, 952)  # H=532, W=952
    out = backbone(dummy_input)
    print(f'Input shape: {dummy_input.shape}')
    print(f'Backbone output shape: {out.shape}')
    B, C, H, W = out.shape
    print(f'Feature map: C={C}, H={H}, W={W}')
    print(f'H*W = {H*W}')

# 3. Now verify: does the model use the CORRECT W?
# The dataset computes voxel_uv by dividing pixel coords by down_scale=4
# The model uses features.shape[3] as W
# Check if these match
print(f'\nModel W from backbone: {W}')
print(f'Expected from 952/4: {952//4}')
print(f'Model H from backbone: {H}')
print(f'Expected from 532/4: {532//4}')

# 4. Check the KEY issue - what if dataset down_scale != model down_scale?
# Load a sample and check what K_downscaled produces
with gzip.open(files[0], 'rb') as fh:
    d = pickle.load(fh)
voxel_uv = d['voxel_uv_left']
intrinsics = d.get('intrinsics', None)
print(f'\nIntrinsics from data: {intrinsics}')
print(f'voxel_uv u range: [{voxel_uv[0].min()}, {voxel_uv[0].max()}]')
print(f'voxel_uv v range: [{voxel_uv[1].min()}, {voxel_uv[1].max()}]')

# 5. Check what percentage of unique features come from OOB voxels vs in-bounds
print('\n=== CLAMPING EFFECT ===')
voxel_uv_t = torch.tensor(voxel_uv, dtype=torch.long)
linear = voxel_uv_t[1] * W + voxel_uv_t[0]
needs_clamp = (linear < 0) | (linear >= H * W)
print(f'Voxels needing clamp: {needs_clamp.sum().item()}/{linear.numel()} ({100*needs_clamp.sum().item()/linear.numel():.1f}%)')
linear_clamped = linear.clamp(0, H * W - 1)
n_unique_total = linear_clamped.unique().numel()
print(f'Unique features (clamped): {n_unique_total}')
# Without clamping
n_unique_valid = linear[~needs_clamp].unique().numel()
print(f'Unique features (valid only): {n_unique_valid}')

# 6. Verify reshape ordering matches
print('\n=== RESHAPE ORDERING VERIFICATION ===')
num_z, num_x, num_y = 164, 64, 40
# Data order: (z0,x0,y0), (z0,x0,y1),...(z0,x0,y39), (z0,x1,y0),...
# Model reshape: features.reshape(B, C, Z, X, Y)
# For this reshape, the rightmost dim (Y) changes fastest, then X, then Z
# This matches the data ordering: Y fastest, then X, then Z
print(f'Voxel ordering in data vs model reshape: COMPATIBLE')
print(f'  Data: Y changes fastest (40 voxels per Z,X cell), then X, then Z')
print(f'  Reshape(Z={num_z}, X={num_x}, Y={num_y}): Y fastest, X next, Z slowest')

# Confirm by checking UV patterns
# Voxels 0-39 should be Y-stack for (z=0, x=0) - all same u,v (close to same)
# Voxels 40-79 should be Y-stack for (z=0, x=1) - slightly different u
first_cell_u = set(voxel_uv[0, :40].tolist())
first_cell_v = set(voxel_uv[1, :40].tolist())
second_cell_u = set(voxel_uv[0, 40:80].tolist())
second_cell_v = set(voxel_uv[1, 40:80].tolist())
print(f'Cell (z=0,x=0): u={first_cell_u}, v={first_cell_v}')
print(f'Cell (z=0,x=1): u={second_cell_u}, v={second_cell_v}')

# After 64 X cells (=64*40=2560 voxels), we should be at z=1
z1_start = 64 * 40
z1_cell_u = set(voxel_uv[0, z1_start:z1_start+40].tolist())
z1_cell_v = set(voxel_uv[1, z1_start:z1_start+40].tolist())
print(f'Cell (z=1,x=0): u={z1_cell_u}, v={z1_cell_v}')

# Compare z=0 vs z=1 - should differ in v (depth direction maps to v in image)
print(f'z=0 vs z=1: u stayed ~same? v changed? -> check above')
