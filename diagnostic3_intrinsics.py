import torch
import numpy as np
import pickle, gzip, glob

# Simulate what crop_image does to intrinsics
# Typical camera intrinsic matrix
K = torch.tensor([
    [2631.0,    0.0,  1904.0],
    [   0.0, 2416.0,  1066.0],
    [   0.0,    0.0,     1.0]
])
print('Original K:')
print(K)

# Image size 3808 x 2132 -> resize to 952 x 532
W, H = 3808, 2132
w_c, h_c = 952, 532
scale_x = W / w_c  # 4.0
scale_y = H / h_c  # ~4.0075

print(f'scale_x={scale_x}, scale_y={scale_y}')

intrinsic = K.clone()

# This is what the code does:
intrinsic[0:2] = intrinsic[0:2] / scale_x    # rows 0,1 divided by scale_x
print('After [0:2] /= scale_x:')
print(intrinsic)

intrinsic[1:2] = intrinsic[1:2] / scale_y    # row 1 divided by scale_y (AGAIN on row 1)
print('After [1:2] /= scale_y:')
print(intrinsic)

intrinsic[0:0] = intrinsic[0:0] / scale_x    # EMPTY SLICE - does nothing
print('After [0:0] /= scale_x (empty slice, no-op):')
print(intrinsic)

intrinsic[1:1] = intrinsic[1:1] / scale_y    # EMPTY SLICE - does nothing
print('After [1:1] /= scale_y (empty slice, no-op):')
print(intrinsic)

print()
print('=== RESULT ===')
print(f'Row 0 was divided by scale_x ONCE: fx={intrinsic[0,0]:.2f}, cx={intrinsic[0,2]:.2f}')
print(f'Row 1 was divided by scale_x THEN scale_y: fy={intrinsic[1,1]:.2f}, cy={intrinsic[1,2]:.2f}')
print(f'fy should be {K[1,1]/scale_y:.2f} but is {intrinsic[1,1]:.2f}')
print(f'cy should be {K[1,2]/scale_y:.2f} but is {intrinsic[1,2]:.2f}')
print(f'fx/fy ratio: {intrinsic[0,0]/intrinsic[1,1]:.4f} (should be ~1.0 for a real camera)')

# Now check with actual stored intrinsics from preprocessed data
print()
print('=== CHECK AGAINST STORED DATA ===')
files = sorted(glob.glob('/data/T7_2/rhf/train_preprocessed_data/data_item_*.pkl.gz'))
with gzip.open(files[0], 'rb') as f:
    data = pickle.load(f)

stored_K = data['intrinsics']
print(f'Stored intrinsics:\n{stored_K}')
print(f'Stored fx={stored_K[0,0]:.4f}, fy={stored_K[1,1]:.4f}')
print(f'Stored fx/fy ratio: {stored_K[0,0]/stored_K[1,1]:.4f}')

# The stored intrinsics were computed by crop_image (which has the bug)
# Then further divided by down_scale=4 in the projection step:
# intrinsic_downscaled = (intrinsic / self.down_scale)
# So the stored K might already be at the down_scaled level or at resize level

# Reverse engineer: what would the original camera K be?
# The bug: row 0 scaled by 1/scale_x, row 1 scaled by 1/(scale_x * scale_y)
# Then downscaled by 1/4
# Stored_fx = original_fx / scale_x / 4
# Stored_fy = original_fy / (scale_x * scale_y) / 4
# So: original_fx = stored_fx * scale_x * 4
# And: original_fy = stored_fy * scale_x * scale_y * 4
original_fx = stored_K[0,0] * 4  # assuming stored is already at downscaled level
original_fy = stored_K[1,1] * 4

print()
print(f'If stored K is at downscale=4 level:')
print(f'  resize-level fx = {original_fx:.2f}')
print(f'  resize-level fy = {original_fy:.2f}')
print(f'  original fx (before resize) = {original_fx * scale_x:.2f}')
print(f'  original fy (before resize, if correctly scaled) = {original_fy * scale_y:.2f}')
print(f'  original fy (before resize, with bug) = {original_fy * scale_x * scale_y:.2f}')

# What the intrinsics SHOULD be and what effect this has on projection
print()
print('=== PROJECTION IMPACT ===')
# With buggy fy (too small by factor scale_x/1 = 4x), 
# v coordinates are compressed by ~4x
# This means the entire ROI maps to a tiny vertical strip of the feature map
# v range is [33, 59] instead of what it should be

# The correct fy at downscale would be:
# correct_fy_ds = original_fy * scale_y / scale_x / down_scale  (if we undo the bug)
# Actually, let me think about this differently.

# If the stored K already has the bug baked in, and the projection uses:
# intrinsic_downscaled = stored_K / 4; intrinsic_downscaled[2,2] = 1
# Then K_projection = stored_K / 4
# v_pixel = (K_projection[1,1] * Y + K_projection[1,2] * Z) / Z

# The bug makes fy too small. This means v-coordinates are compressed.
# With correct fy, the v range would be ~4x larger (spanning more of the feature map)
correct_fy_at_ds = stored_K[1,1] * (stored_K[0,0] / stored_K[1,1])  # multiply by the fx/fy ratio to fix
print(f'Current fy at feat-map level: {stored_K[1,1]/4:.2f}')
print(f'Correct fy at feat-map level: {correct_fy_at_ds/4:.2f}')
print(f'Current v range: [33, 59] -> span of {59-33} pixels')
print(f'Expected v range with correct fy: span of ~{int((59-33) * stored_K[0,0]/stored_K[1,1])} pixels')
print(f'Feature map H = 133')
print()
print('The bug compresses ALL voxels into {:.0f}% of the feature maps vertical extent'.format(
    (59-33) / 133 * 100))
print('With correct fy, they would span {:.0f}% of it'.format(
    (59-33) * stored_K[0,0]/stored_K[1,1] / 133 * 100))

# Check the preprocessed data: was it created using CARDSetSmall or CARDSet?
# And does the preprocessing code also have this bug?
print()
print('=== KEY QUESTION ===')
print('Were the preprocessed voxel_uv computed with the buggy crop_image?')
print('Stored intrinsics suggest YES - the fx/fy ratio of {:.1f} matches the bug pattern'.format(
    stored_K[0,0]/stored_K[1,1]))
