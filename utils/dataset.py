import numpy as np
import math
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import PIL.Image
from torchvision import transforms
import open3d as o3d
import copy, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from RSRD_dev_toolkitmain.cam_extrinsic import Extrinsic

class RSRD(Dataset):
    def __init__(self, training=True, stereo=False, down_scale=2, backbone=None):
        super(RSRD, self).__init__()
        self.training = training
        self.stereo = stereo
        self.down_scale = down_scale
        self.backbone = backbone or ''
        # RHF / DINOv2 backbones use a ViT with patch_size=14, which requires
        # both image dimensions to be divisible by 14. RSRD's native crop is
        # 960x528 (neither dim is %14), so we right/bottom-pad on the fly.
        self.is_rhf = 'DINOv2' in self.backbone
        if self.is_rhf:
            self.rhf_pad_stride = 14 * self.down_scale // math.gcd(14, self.down_scale)

        self.calib_path = 'calibration_files'  # path for calibration files
        # path for training set of RSRD-dense. Both the train and test sets in this work are from the train set of RSRD-dense
        self.data_path = '/data/RSRD/RSRD-dense/train'   
        preprocessed_path = './preprocessed/'  # path for preprocessed GT maps

        if self.training:
            self.load_dataset_names('./filenames/train/')
            self.preprocessed_path = os.path.join(preprocessed_path, 'train')
        else:
            # IDENTICAL TO TRAIN FOR DEBUG: loads same training data
            self.load_dataset_names('./filenames/train/')
            self.preprocessed_path = os.path.join(preprocessed_path, 'train')

        #######################
        # settings about range of interest. If you change the params below, please confirm that the voxel ROI completely falls in the image view.
        #######################
        self.base_height = 1.1  # in meter, the reference height of the camera w.r.t. road surface
        self.y_range = 0.2  # in meter, the range of interest above and below the base height， i.e., [-20cm, 20cm]
        self.roi_x = torch.tensor([-1, 0.92])    # in meter, the lateral range of interest (in the horizontal coordinate of camera)
        self.roi_z = torch.tensor([2.16, 7.08])    # in meter, the longitudinal range of interest
        #######################
        
        self.grid_res = torch.tensor([0.03, 0.01, 0.03])  # in [x, y(vertical), z] order. The range of interest above should be integer times of resolution here
        

        self.num_grids_x = int((self.roi_x[1] - self.roi_x[0]) / self.grid_res[0])
        self.num_grids_z = int((self.roi_z[1] - self.roi_z[0]) / self.grid_res[2])
        self.num_grids_y = int(self.y_range*2 / self.grid_res[1])

        # generate the centers of every horizontal grid
        hori_centers = torch.zeros((self.num_grids_z, self.num_grids_x, 2), dtype=torch.float32)
        hori_centers[:, :, 0] = (torch.arange(self.num_grids_x) * self.grid_res[0] + self.roi_x[0] + self.grid_res[0]/2).unsqueeze(0).repeat([self.num_grids_z, 1])
        hori_centers[:, :, 1] = (-torch.arange(self.num_grids_z) * self.grid_res[2] + self.roi_z[1] - self.grid_res[2]/2).unsqueeze(1).repeat([1, self.num_grids_x])
        self.map_centers = hori_centers.reshape(-1, 2)
        self.num_center = self.map_centers.shape[0]
        self.hori_centers = hori_centers
        # generate the centers of every 3D voxel
        voxel_centers = torch.zeros((self.num_grids_z, self.num_grids_x, self.num_grids_y, 3), dtype=torch.float32)
        voxel_centers[:, :, :, [0, 2]] = hori_centers.unsqueeze(2).repeat([1, 1, self.num_grids_y, 1])
        voxel_centers[:, :, :, 1] = (torch.arange(self.num_grids_y) * self.grid_res[1] + self.base_height - self.y_range + self.grid_res[1]/2).unsqueeze(0).unsqueeze(0).repeat([self.num_grids_z, self.num_grids_x, 1])
        self.voxel_centers = voxel_centers.reshape(-1, 3).transpose(1, 0)

        # pre_read the extrinsic parameters between camera and lidar
        # intrinsics (after rectification): calib_params["K"]
        # stereo baseline(in mm): calib_params["B"]
        # lidar -> left camera extrinsics: calib_params["R"], calib_params["T"]
        calib_files = ['calib_20230317_half.pkl', 'calib_20230321_half.pkl', 'calib_20230406_half.pkl', 'calib_20230408_half.pkl', 'calib_20230409_half.pkl']
        self.calib_params_all = {}
        for file in calib_files:
            with open(os.path.join(self.calib_path, file), 'rb') as f:
                calib_params = pickle.load(f)
            calib_params['K'] = calib_params['K'].astype(np.float32)
            calib_params['R'] = calib_params['R'].astype(np.float32)
            calib_params['T'] = calib_params['T'].astype(np.float32)
            calib_params['B'] = calib_params['B']/1000   # mm -> m
            # 'K_feat_T' is the intrinsic of the reduced feature map
            calib_params['K_feat_T'] = torch.from_numpy(calib_params['K'] / self.down_scale)

            calib_params['K_feat_T'][2, 2] = 1
            calib_params['R_inv'] = np.linalg.inv(calib_params['R']).astype(np.float32)
            date = file[6:14]
            self.calib_params_all[date] = calib_params

        self.transform_jpg = transforms.Compose([
            transforms.ToTensor(),  # image --> [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0, 1] --> [-1, 1]
        ])

    def _pad_for_rhf(self, img):
        """Right/bottom-pad a [C, H, W] tensor so H and W are multiples of
        lcm(14, down_scale). Padding (vs. resize) preserves the intrinsics, so
        K_feat_T and the precomputed voxel_uv projections remain valid — the
        original image content stays at the top-left and padded pixels lie
        outside the road ROI."""
        if not self.is_rhf:
            return img
        s = self.rhf_pad_stride
        _, H, W = img.shape
        pad_h = (s - H % s) % s
        pad_w = (s - W % s) % s
        if pad_h == 0 and pad_w == 0:
            return img
        return F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0.0)

    def load_dataset_names(self, sample_path):
        data_all = []
        files = sorted(os.listdir(sample_path))
        for file in files:
            with open(os.path.join(sample_path, file), 'rb') as f:
                data = pickle.load(f)
            data_all += data
        self.data_all = data_all
        # print(f"Total number of samples loaded: {len(self.data_all)}")

    def get_lidar2cam(self, date_stamp):
        # name in format: 20230408023213.400
        date = date_stamp[:8]
        return self.calib_params_all[date]

    def yaw_convert(self, yaw):
        '''
            convert the yaw data from [0, 2pi] to [-pi, pi]
        '''
        if np.pi <= yaw <= 2 * np.pi:
            yaw -= 2 * np.pi
        return yaw

    def lla_to_enu(self, C_lat, C_lon, C_alt, O_lat, O_lon, O_alt):
        '''
            Calculate the relative location with respect to the selected origin in the local ENU coordinate. unit: meter
            C_lat, C_lon, C_alt: current location
            O_lat, O_lon, O_alt: origin location
        '''
        Ea = 6378137
        Eb = 6356752.3142
        C_lat = math.radians(C_lat)
        C_lon = math.radians(C_lon)
        O_lat = math.radians(O_lat)
        O_lon = math.radians(O_lon)
        Ec = Ea * (1 - (Ea - Eb) / Ea * (math.sin(C_lat)) ** 2) + C_alt
        d_lat = C_lat - O_lat
        d_lon = C_lon - O_lon
        e = d_lon * Ec * math.cos(C_lat)
        n = d_lat * Ec
        u = C_alt - O_alt
        return np.array([e, n, u])

    def get_RT_lidar(self, loc_pose):
        #### pre-process
        # convert from angle to radius， then rectify to range -pi~pi
        rotX_cur = 0.017453 * loc_pose['pitch']
        rotY_cur = 0.017453 * loc_pose['roll']
        rotZ_cur = 0.017453 * loc_pose['yaw']
        rotZ_cur = self.yaw_convert(rotZ_cur)

        # rotation order ZXY， the derived R is rotation matrix from current pose to local ENU
        R_X1 = np.array([[1, 0, 0], [0, np.cos(rotX_cur), -np.sin(rotX_cur)], [0, np.sin(rotX_cur), np.cos(rotX_cur)]])
        R_Y1 = np.array([[np.cos(rotY_cur), 0, np.sin(rotY_cur)], [0, 1, 0], [-np.sin(rotY_cur), 0, np.cos(rotY_cur)]])
        R_Z1 = np.array([[np.cos(rotZ_cur), -np.sin(rotZ_cur), 0], [np.sin(rotZ_cur), np.cos(rotZ_cur), 0], [0, 0, 1]])
        R_cur2enu = R_Z1 @ R_X1 @ R_Y1   # the rotation from current lidar to enu

        return R_cur2enu

    def get_gt_elevation(self, xyz):
        xyz = np.asarray(xyz.points)
        N, _ = xyz.shape
        points_y = xyz[:, 1]*100  # points, m --> cm
        # print(xyz[:, 1].max(), xyz[:, 1].min())
        # print(points_y.mean())
        points_xz = xyz[:, [0, 2]]
        grids_y = torch.zeros((self.num_grids_z, self.num_grids_x), dtype=torch.float32)
        grids_count = torch.zeros((self.num_grids_z, self.num_grids_x), dtype=torch.int32)  # int8 overflows at 127 with dense point clouds

        for xz, y in zip(points_xz, points_y):
            idx_x = torch.clip(((xz[0] - self.roi_x[0]) / self.grid_res[0]).int(), max=self.num_grids_x-1)
            idx_z = torch.clip(self.num_grids_z - 1 - ((xz[1] - self.roi_z[0]) / self.grid_res[2]).int(), min=0)
            grids_y[idx_z, idx_x] += y
            grids_count[idx_z, idx_x] += 1
        mask = grids_count > 0
        grids_y[mask] = self.base_height*100 - grids_y[mask] / grids_count[mask]

        return grids_y, mask

    def get_gt_preprocessed(self, time):
        with open(os.path.join(self.preprocessed_path, time)+'.pkl', 'rb') as f:
            [ele_gt, ele_mask] = pickle.load(f)
        return ele_gt, ele_mask

    def matrix2euler(self, m):
        # order='XYZ'
        d = np.clip
        m = m.reshape(-1)
        a, f, g, k, l, n, e = m[0], m[1], m[2], m[4], m[5], m[7], m[8]
        y = np.arcsin(d(g, -1, 1))
        if 0.99999 > np.abs(g):
            x = np.arctan2(- l, e)
            z = np.arctan2(- f, a)
        else:
            x = np.arctan2(n, k)
            z = 0
        return np.array([x, y, z], dtype=np.float32)

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):
        sample_cur = self.data_all[index]
        l2c_calib_cur = self.get_lidar2cam(sample_cur['time'])
        path_base = sample_cur['path']
        idx_str = path_base.find('/')
        path_base = path_base[idx_str + 1:]

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        #R = coord_frame.get_rotation_matrix_from_xyz((0.0, 0.0, np.pi))  # (Rx, Ry, Rz) in radians
        #coord_frame = coord_frame.rotate(R, center=(0, 0, 0))

        ########   calculate the euler angles of the camera (relative to local ENU coord)   ########
        R_cur2enu = self.get_RT_lidar(sample_cur)
        # print("aaaaa", np.linalg.inv(R_cur2enu))
        [pitch_cam, roll_cam, _] = self.matrix2euler(l2c_calib_cur['R'] @ np.linalg.inv(R_cur2enu))
        pitch_cam -= 1.5708  # pi/2
        R_X = np.array(
            [[1, 0, 0], [0, np.cos(pitch_cam), np.sin(pitch_cam)], [0, -np.sin(pitch_cam), np.cos(pitch_cam)]], dtype=np.float32)
        R_Z = np.array(
            [[np.cos(roll_cam), np.sin(roll_cam), 0], [-np.sin(roll_cam), np.cos(roll_cam), 0], [0, 0, 1]], dtype=np.float32)
        R_cam2vert = R_X @ R_Z  # the rotation matrix from the current camera coord to the vertical status
        R_vert2cam = torch.from_numpy(np.linalg.inv(R_cam2vert))

        mou = copy.deepcopy(coord_frame)
        mou = mou.rotate(l2c_calib_cur['R'], center=(0, 0, 0))
        mou = mou.translate(tuple(l2c_calib_cur['T'].reshape(-1)))

        road_frame = copy.deepcopy(mou)
        road_frame.rotate(R_cam2vert, center=(0, 0 , 0))
        #road_frame.translate((0, -1* self.base_height, 0)) 


        ######   create the GT elevation map  ########
        #import pdb; pdb.set_trace()
        ele_gt, ele_mask = self.get_gt_preprocessed(sample_cur['time'])
        # print(ele_gt.shape, ele_gt[:, 1])

        

        ##########  read the RGB images   ############
        path_img = os.path.join(self.data_path, path_base, 'left_half', sample_cur['time']) + '.jpg'
        img = PIL.Image.open(path_img).crop((0, 0, 960, 528))
        imgs_left = self._pad_for_rhf(self.transform_jpg(img))
        #print("mbou", imgs_left.shape)

        voxel_cam_left = R_vert2cam @ self.voxel_centers
        if self.stereo:
            #########   calculate the index relationship between 3D voxels and 2D pixels   ##############
            voxel_cam_right = copy.deepcopy(voxel_cam_left)
            # voxel_cam_right = voxel_cam_left
            voxel_cam_right[0, :] = voxel_cam_right[0, :] - l2c_calib_cur['B']
            uvz_left = l2c_calib_cur['K_feat_T'] @ voxel_cam_left
            uvz_right = l2c_calib_cur['K_feat_T'] @ voxel_cam_right  # projection index on right image plane
            voxel_uv_left = torch.floor(uvz_left[:2, :] / uvz_left[2:, :]).type(torch.long)
            voxel_uv_right = torch.floor(uvz_right[:2, :] / uvz_right[2:, :]).type(torch.long)

            path_img = os.path.join(self.data_path, path_base, 'right_half', sample_cur['time']) + '.jpg'
            img = PIL.Image.open(path_img).crop((0, 0, 960, 528))
            imgs_right = self._pad_for_rhf(self.transform_jpg(img))

            return imgs_left, imgs_right, ele_gt, ele_mask, voxel_uv_left, voxel_uv_right, sample_cur['time']
        else:
            uvz_left = l2c_calib_cur['K_feat_T'] @ voxel_cam_left
            voxel_uv_left = torch.floor(uvz_left[:2, :] / uvz_left[2:, :]).type(torch.long)

            # Validate voxel projections are within camera frustum
            feat_H = 528 // self.down_scale
            feat_W = 960 // self.down_scale
            
            # Check validity before clamping
            valid_mask = (voxel_uv_left[0] >= 0) & (voxel_uv_left[0] < feat_W) & \
                        (voxel_uv_left[1] >= 0) & (voxel_uv_left[1] < feat_H)

            valid_ratio = valid_mask.sum().item() / valid_mask.numel()
            if valid_ratio < 0.7 or index < 3:  # Always visualize first 3 samples + failures
                # ── Frustum debug visualization ───────────────────────────
                os.makedirs("frustum_debug", exist_ok=True)
                tag = f"rsrd_idx{index}_ts{sample_cur['time']}"

                u = voxel_uv_left[0].numpy()
                v = voxel_uv_left[1].numpy()

                fig, axes = plt.subplots(1, 4, figsize=(32, 7))

                # 1) UV scatter
                ax = axes[0]
                ax.scatter(u[valid_mask.numpy()], v[valid_mask.numpy()],
                           s=0.3, alpha=0.3, c='green', label='in-frustum')
                ax.scatter(u[~valid_mask.numpy()], v[~valid_mask.numpy()],
                           s=0.3, alpha=0.3, c='red', label='out-of-frustum')
                ax.axhline(0, color='blue', ls='--', lw=0.8)
                ax.axhline(feat_H - 1, color='blue', ls='--', lw=0.8, label=f'feat_H={feat_H}')
                ax.axvline(0, color='blue', ls='--', lw=0.8)
                ax.axvline(feat_W - 1, color='blue', ls='--', lw=0.8, label=f'feat_W={feat_W}')
                ax.set_xlabel('u (px)')
                ax.set_ylabel('v (px)')
                ax.set_title(f'Voxel UV projections — {valid_ratio*100:.1f}% valid')
                ax.legend(markerscale=10, fontsize=8)
                ax.invert_yaxis()

                # 2) Overlay on image with distance lines
                ax = axes[1]
                img_np = np.array(img)
                ax.imshow(img_np)
                scale = self.down_scale
                ax.scatter(u[valid_mask.numpy()] * scale, v[valid_mask.numpy()] * scale,
                           s=0.2, alpha=0.2, c='lime')
                ax.scatter(u[~valid_mask.numpy()] * scale, v[~valid_mask.numpy()] * scale,
                           s=0.2, alpha=0.2, c='red')
                # Draw distance reference lines on the image
                K_ds = l2c_calib_cur['K_feat_T'].numpy()
                for z_ref in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
                    v_line = K_ds[1, 1] * self.base_height / z_ref + K_ds[1, 2]
                    v_img = v_line * scale
                    if 0 <= v_img < img_np.shape[0]:
                        ax.axhline(v_img, color='yellow', ls='-', lw=0.6, alpha=0.7)
                        ax.text(5, v_img - 3, f'{z_ref}m', color='yellow', fontsize=7,
                                fontweight='bold', bbox=dict(boxstyle='round,pad=0.1',
                                facecolor='black', alpha=0.5))
                ax.set_title('Projections on image + distance lines')
                ax.axis('off')

                # 3) Overlay on unsqueezed image
                ax = axes[2]
                img_unsqueezed = img.resize((528, 528))  # make it square
                img_unsq_np = np.array(img_unsqueezed)
                ax.imshow(img_unsq_np)
                # Scale u,v from feature-map to unsqueezed image coords
                u_unsq = u * scale * (528.0 / 960.0)  # rescale horizontal
                v_unsq = v * scale  # vertical stays same
                valid_np = valid_mask.numpy()
                ax.scatter(u_unsq[valid_np], v_unsq[valid_np],
                           s=0.2, alpha=0.3, c='lime')
                ax.scatter(u_unsq[~valid_np], v_unsq[~valid_np],
                           s=0.2, alpha=0.3, c='red')
                for z_ref in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
                    v_line = K_ds[1, 1] * self.base_height / z_ref + K_ds[1, 2]
                    v_img = v_line * scale
                    if 0 <= v_img < 528:
                        ax.axhline(v_img, color='yellow', ls='-', lw=0.6, alpha=0.7)
                        ax.text(5, v_img - 3, f'{z_ref}m', color='yellow', fontsize=7,
                                fontweight='bold', bbox=dict(boxstyle='round,pad=0.1',
                                facecolor='black', alpha=0.5))
                ax.set_title('Unsqueezed + distance lines')
                ax.axis('off')

                # 4) Histogram of v-coordinates
                ax = axes[3]
                ax.hist(v, bins=200, color='steelblue', edgecolor='none')
                ax.axvline(0, color='red', ls='--', label='v=0')
                ax.axvline(feat_H - 1, color='red', ls='--', label=f'v={feat_H-1}')
                ax.set_xlabel('v (feature-map px)')
                ax.set_ylabel('count')
                ax.set_title('v-coordinate distribution')
                ax.legend(fontsize=8)

                plt.suptitle(f'{path_img}\n'
                             f'K={l2c_calib_cur["K"].tolist()}\n'
                             f'roi_z=[{self.roi_z[0]:.2f}, {self.roi_z[1]:.2f}], '
                             f'base_h={self.base_height:.3f}m\n'
                             f'Original image: 960x528 cropped'
                             , fontsize=8)
                plt.tight_layout()
                # plt.savefig(f"frustum_debug/{tag}.png", dpi=150)
                plt.close()

                # print(f"[FRUSTUM DEBUG RSRD] saved frustum_debug/{tag}.png  "
                #       f"valid={valid_ratio*100:.1f}%  "
                #       f"v range=[{v.min()}, {v.max()}]  "
                #       f"u range=[{u.min()}, {u.max()}]  "
                #       f"feat={feat_W}x{feat_H}")

            if valid_ratio < 0.7:
                raise ValueError(
                    f"RSRD: Only {valid_mask.sum()}/{valid_mask.numel()} "
                    f"({valid_ratio*100:.1f}%) voxels project within camera frustum. "
                    f"Feature map size: {feat_W}x{feat_H}, "
                    f"UV range: [{u.min()},{u.max()}] x [{v.min()},{v.max()}]. "
                    f"See frustum_debug/{tag}.png "
                    f"Data path: {path_img}")
            
            # Clamp to be safe
            voxel_uv_left[0] = voxel_uv_left[0].clamp(0, feat_W - 1)
            voxel_uv_left[1] = voxel_uv_left[1].clamp(0, feat_H - 1)

            if index == 0:
                # print(f"elevation{ele_gt.shape}")
                # print("image shape", imgs_left.shape)
                # print(voxel_uv_left.shape)
                # print(sample_cur['time'])

                T_l2c = l2c_calib_cur['T']
                R_l2c = l2c_calib_cur['R']
                extrinsic_matrix = np.eye(4, dtype=np.float32)
                extrinsic_matrix[:3, :3] = R_l2c
                extrinsic_matrix[:3, 3] = T_l2c.flatten()
                # print(extrinsic_matrix)
                # print(self.calib_params_all['20230317']['K_feat_T'])
                #draw_voxel_bounding_boxes(path_img, self.voxel_centers, torch.Tensor(self.calib_params_all['20230317']['K_feat_T']), torch.Tensor(extrinsic_matrix), self.down_scale )



            path_pcd = os.path.join(self.data_path, path_base, 'pcd', sample_cur['time']) + '.pcd'
            cloud = o3d.io.read_point_cloud(path_pcd)
            cloud = cloud.rotate(l2c_calib_cur['R'], center=(0, 0, 0))
            cloud = cloud.translate(tuple(l2c_calib_cur['T'].reshape(-1)))  # the point cloud in the camera's coord
            
            #print("God", R_cam2vert)
            cloud_camvert = cloud.rotate(R_cam2vert, center=(0, 0, 0))
            # self.save_gt_as_image(cloud_camvert, path_img, self.calib_params_all['20230317']['K'], " " , index)
            # points = np.array(cloud_camvert.points)
            # print(f"Height after transformation values (y): min={points[:, 1].min()}, max={points[:, 1].max()}")

            #self.save_gt_points(cloud_camvert)
            crop_bounding = np.array([[self.roi_x[0], 0, self.roi_z[0]],
                                   [self.roi_x[0], 0, self.roi_z[1]],
                                   [self.roi_x[1], 0, self.roi_z[1]],
                                   [self.roi_x[1], 0, self.roi_z[0]]]).astype("float64")
            
            vol_roi = o3d.visualization.SelectionPolygonVolume()
            vol_roi.orthogonal_axis = "Y"
            vol_roi.axis_max = 1.5
            vol_roi.axis_min = 0.5
            vol_roi.bounding_polygon = o3d.utility.Vector3dVector(crop_bounding)

            cloud_camvert = vol_roi.crop_point_cloud(cloud_camvert)

            #o3d.visualization.draw_geometries([road_frame, mou])
            # print(f"voxel_uv_left{voxel_uv_left.shape}")
            # print(voxel_uv_left[:, -1])
            # print(voxel_uv_left[:, 16000])
            #if index == 57:
                #self.save_gt_points(cloud_camvert, 'after')
            #print("mamamamama ", np.asarray(cloud_camvert.points).max())
            #self.save_gt_as_image(cloud_camvert, path_img, self.calib_params_all['20230317']['K'], "cam_vert", index)

            return imgs_left, ele_gt, ele_mask, voxel_uv_left, sample_cur['time']
    
    def save_gt_as_image(self, pcd, img_path, intrinsic, addi = ' ', index = 0):
        
        # Assuming you already have your point cloud object `pcd`
        # Extract points as NumPy array
        points = np.asarray(pcd.points)
        img = cv2.imread(img_path)

    # Project points to image plane
        uvz = (intrinsic @ points.T).T  # shape: (N, 3)
        uv = uvz[:, :2] / uvz[:, 2].reshape(-1, 1)

        # Draw points on image
        for (u, v) in uv:
            u_int, v_int = int(u), int(v)
            if 0 <= u_int < img.shape[1] and 0 <= v_int < img.shape[0]:
                cv2.circle(img, (u_int, v_int), 2, (0, 255, 0), -1)  # green dots

        # Save the overlay image
        # output_path = "visualization rsrd data pcd" + addi + str(index) + ".jpg"
        # cv2.imwrite(output_path, img)
        # print(f"Overlay image saved to {output_path}")
    
    def save_gt_points(self, pcd, str=''):
        pts = np.asarray(pcd.points)
        x = pts[:, 0]  # longitudinal
        y = pts[:, 1]  # height
        z = pts[:, 2]  # lateral

        # Plot f(x, z) = y using color encoding for y
        plt.figure(figsize=(8, 4))
        scatter = plt.scatter(x, z, c=y, cmap='plasma', s=2)
        plt.colorbar(scatter, label='Height (Y)')
        plt.xlabel('X (Lateral)')
        plt.ylabel('Z (Longitidunal)')
        plt.title('f(x, z) = y encoded with plasma colormap')
        plt.tight_layout()

        # Save the plot as a PNG file
        # plt.savefig('pointcloud_projection_baseline' + str + '.png', dpi=300)
        # plt.show()

def draw_voxel_bounding_boxes(img_path, voxel_centers, intrinsic, extrinsic, down_scale=1):
    """
    Draw bounding boxes of voxels on the image.

    Parameters:
        img_path (str): Path to the image file.
        voxel_centers (torch.Tensor): 3D voxel centers of shape [3, N].
        intrinsic (torch.Tensor): Camera intrinsic matrix of shape [3, 3].
        extrinsic (torch.Tensor): Camera extrinsic matrix of shape [4, 4].
        down_scale (float): Downscaling factor for the intrinsic matrix.
    """
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found at {img_path}")
    height, width, _ = img.shape

    # Downscale the intrinsic matrix
    intrinsic_downscaled = intrinsic / 1

    # Transform voxel centers to camera coordinates
    voxel_cam = extrinsic[:3, :3] @ voxel_centers + extrinsic[:3, 3].unsqueeze(1)  # [3, N]

    # Project voxel centers to the image plane
    uvz = intrinsic_downscaled @ voxel_cam  # [3, N]
    uv = uvz[:2, :] / uvz[2, :]  # Normalize by depth (z)

    # Convert to integer pixel coordinates
    uv = uv.T.cpu().numpy().astype(np.int32)  # Shape: [N, 2]

    # Draw bounding boxes on the image
    for (u, v) in uv:
        if 0 <= u < width and 0 <= v < height:  # Check if the point is within the image bounds
            cv2.rectangle(img, (u - 5, v - 5), (u + 5, v + 5), (0, 255, 0))  # Green box

    # Save or display the image
    # output_path = "voxel_bounding_boxes.jpg"
    # cv2.imwrite(output_path, img)
    # print(f"Image with voxel bounding boxes saved to {output_path}")


if __name__ == '__main__':
    dataset = RSRD(down_scale=2, training=False, stereo=False)
    for i in range(len(dataset)):
        dataset.__getitem__(i)

    