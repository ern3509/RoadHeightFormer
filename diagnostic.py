import torch, numpy as np, pickle, gzip, os, glob

# Find a preprocessed sample
files = sorted(glob.glob('/data/T7_2/rhf/train_preprocessed_data/data_item_*.pkl.gz'))
print(f'Found {len(files)} preprocessed files')
path = files[0]
print(f'Using: {path}')

with gzip.open(path, 'rb') as f:
    data = pickle.load(f)

print(f'Keys: {list(data.keys())}')

ele_gt = torch.tensor(data['ele_gt'], dtype=torch.float32)
mask = torch.tensor(data['mask'], dtype=torch.int8)
voxel_uv = torch.tensor(data['voxel_uv_left'], dtype=torch.long)

print()
print('=== GT STATISTICS ===')
print(f'ele_gt shape: {ele_gt.shape}')
print(f'mask shape: {mask.shape}')
print(f'mask coverage: {mask.sum().item()}/{mask.numel()} ({100*mask.sum().item()/mask.numel():.1f}%)')

valid = ele_gt[mask > 0]
print(f'GT valid range: [{valid.min():.3f}, {valid.max():.3f}] cm')
print(f'GT valid mean: {valid.mean():.3f} cm')
print(f'GT valid std: {valid.std():.3f} cm')
print(f'GT valid median: {valid.median():.3f} cm')

print()
print('=== GT HEIGHT HISTOGRAM (cm) ===')
for lo in range(-10, 11, 1):
    hi = lo + 1
    count = ((valid >= lo) & (valid < hi)).sum().item()
    bar = '#' * int(count / max(valid.numel(), 1) * 200)
    print(f'  [{lo:+3d}, {hi:+3d}): {count:5d} {bar}')

print()
print('=== FULL GT (including masked) ===')
print(f'ele_gt full range: [{ele_gt.min():.3f}, {ele_gt.max():.3f}]')
masked_vals = ele_gt[mask == 0]
if masked_vals.numel() > 0:
    print(f'ele_gt where mask=0: all zero? {(masked_vals == 0).all()}')

print()
print(f'=== VOXEL UV (PROJECTION) ===')
print(f'voxel_uv shape: {voxel_uv.shape}')
print(f'u range: [{voxel_uv[0].min()}, {voxel_uv[0].max()}]')
print(f'v range: [{voxel_uv[1].min()}, {voxel_uv[1].max()}]')

# Check actual feature map size used by model
# The model uses backbone on cropped image, then features are flattened
# Need to check what W, H the model actually uses
# From dataset: image is resized to (952, 532) then maybe cropped
# EfficientNet at down_scale=4: feat = 952/4 x 532/4 = 238 x 133
feat_W, feat_H = 238, 133
print(f'Assumed feat map: {feat_W}x{feat_H} (from 952x532 / 4)')

oob_u = ((voxel_uv[0] < 0) | (voxel_uv[0] >= feat_W)).sum().item()
oob_v = ((voxel_uv[1] < 0) | (voxel_uv[1] >= feat_H)).sum().item()
oob_either = ((voxel_uv[0] < 0) | (voxel_uv[0] >= feat_W) | (voxel_uv[1] < 0) | (voxel_uv[1] >= feat_H)).sum().item()
print(f'Out-of-bounds u: {oob_u}/{voxel_uv.shape[1]} ({100*oob_u/voxel_uv.shape[1]:.1f}%)')
print(f'Out-of-bounds v: {oob_v}/{voxel_uv.shape[1]} ({100*oob_v/voxel_uv.shape[1]:.1f}%)')
print(f'Out-of-bounds either: {oob_either}/{voxel_uv.shape[1]} ({100*oob_either/voxel_uv.shape[1]:.1f}%)')

neg_u = (voxel_uv[0] < 0).sum().item()
neg_v = (voxel_uv[1] < 0).sum().item()
print(f'Negative u: {neg_u}, Negative v: {neg_v}')

linear = voxel_uv[1] * feat_W + voxel_uv[0]
linear_clamped = linear.clamp(0, feat_H * feat_W - 1)
n_unique = linear_clamped.unique().numel()
total_voxels = voxel_uv.shape[1]
print(f'Unique feature positions (after clamp): {n_unique}/{total_voxels} ({100*n_unique/total_voxels:.1f}%)')

num_grids_x, num_grids_z, num_grids_y = 64, 164, 40
print()
print(f'=== GRID INFO ===')
print(f'num_grids: x={num_grids_x}, y={num_grids_y}, z={num_grids_z}')
print(f'Total voxels expected: {num_grids_x * num_grids_y * num_grids_z}')
print(f'Actual voxel_uv cols: {voxel_uv.shape[1]}')

print()
print('=== FEATURE DIVERSITY PER GT CELL ===')
voxels_per_cell = num_grids_y
n_cells = num_grids_z * num_grids_x
unique_per_cell = []
for cell_idx in range(min(n_cells, 2000)):
    start = cell_idx * voxels_per_cell
    end = start + voxels_per_cell
    if end > voxel_uv.shape[1]:
        break
    cell_linear = linear_clamped[start:end]
    unique_per_cell.append(cell_linear.unique().numel())
unique_per_cell = np.array(unique_per_cell)
print(f'Unique feats/cell: mean={unique_per_cell.mean():.1f}, min={unique_per_cell.min()}, max={unique_per_cell.max()}')
print(f'Cells with only 1 unique feature: {(unique_per_cell == 1).sum()}/{len(unique_per_cell)}')
print(f'Cells with <=2 unique features: {(unique_per_cell <= 2).sum()}/{len(unique_per_cell)}')

# Show distribution of unique counts
print('\nDistribution of unique feature counts per cell:')
for n in range(1, 15):
    c = (unique_per_cell == n).sum()
    if c > 0:
        print(f'  {n} unique: {c} cells')

# Voxel ordering check
print()
print('=== VOXEL ORDERING CHECK ===')
print(f'First 5 voxels UV (should be Y-stack for cell z=0,x=0):')
for i in range(5):
    print(f'  voxel {i}: u={voxel_uv[0,i]}, v={voxel_uv[1,i]}')
print(f'Voxels 40-44 UV (should be Y-stack for cell z=0,x=1):')
for i in range(40, 45):
    print(f'  voxel {i}: u={voxel_uv[0,i]}, v={voxel_uv[1,i]}')

# Check across multiple samples
print()
print('=== GT STATS ACROSS 20 SAMPLES ===')
means, stds, coverages = [], [], []
for f in files[:20]:
    with gzip.open(f, 'rb') as fh:
        d = pickle.load(fh)
    gt = torch.tensor(d['ele_gt'], dtype=torch.float32)
    m = torch.tensor(d['mask'], dtype=torch.int8)
    v = gt[m > 0]
    if v.numel() > 0:
        means.append(v.mean().item())
        stds.append(v.std().item())
        coverages.append(100 * m.sum().item() / m.numel())
print(f'Mean of means: {np.mean(means):.3f} cm')
print(f'Mean of stds: {np.mean(stds):.3f} cm')
print(f'Range of means: [{min(means):.3f}, {max(means):.3f}]')
print(f'Range of stds: [{min(stds):.3f}, {max(stds):.3f}]')
print(f'Mean coverage: {np.mean(coverages):.1f}%')

# CRITICAL: Check what happens in the model's forward pass
# The model does: linear_indices = proj_index_left[:, 1, :] * W + proj_index_left[:, 0, :]
# Then: features.gather(dim=2, index=linear_indices)
# What W does the model use?
print()
print('=== MODEL INDEX COMPUTATION CHECK ===')
# In model.py, W comes from features.shape[3] (the width dimension of backbone output)
# For EfficientNet with input 952x532:
# The backbone outputs a feature map, then model flattens H*W
# features shape: [B, C, H*W]
# linear_indices = v_coord * W + u_coord
# W is the feature map WIDTH
print(f'If model W=238 (952/4): max linear = {voxel_uv[1].max()} * 238 + {voxel_uv[0].max()} = {voxel_uv[1].max() * 238 + voxel_uv[0].max()}')
print(f'Feature map size = 238*133 = {238*133}')
print(f'Max valid index = {238*133 - 1}')

# Check what W the model ACTUALLY uses by reading model.py
print()
print('=== CHECKING MODEL W USAGE ===')
# The model gets W from: features = self.backbone(imgs_left)
# Then: H, W = features.shape[2], features.shape[3]
# For EfficientNet-B0, the final feature map with input 952x532:
# Layer5 output is input/32 = 952/32 x 532/32 ≈ 30 x 17
# But down_scale might use a different layer
# Let's check what the backbone actually returns
print('Need to check actual backbone output shape - will do in model test')
