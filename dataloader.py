import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
import json
import numpy as np
import os
import scipy.io 
from scipy.spatial.transform import Rotation as R_scipy
from PIL import Image, ImageOps

from aug_utils import FastPoseAugmentations
from fastpose_vit.geometry_utils import (
    compute_apparent_rotation, 
    matrix_to_ortho6d, 
    compute_normalized_coords_translation
)

class SpeedDataset(Dataset):
    def __init__(self, csv_file, image_root, intrinsics_file=None, keypoints_file='tangoPoints.mat', mode='train'):
        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.mode = mode
        self.use_spatial_aug = False
        self.augmentor = FastPoseAugmentations(train=(mode == 'train'))
        
        self.W_full = 1920
        self.H_full = 1200
        if intrinsics_file and os.path.exists(intrinsics_file):
            with open(intrinsics_file, 'r') as f:
                intrinsics_data = json.load(f)
                if 'cameraMatrix' in intrinsics_data:
                    self.intrinsics_mat = np.array(intrinsics_data['cameraMatrix'], dtype=np.float32)
                else:
                    self.intrinsics_mat = np.array([
                        [2988.5795163815555, 0, 960],
                        [0, 2988.3401159176124, 600],
                        [0, 0, 1]
                    ], dtype=np.float32)
        else:
            self.intrinsics_mat = np.array([
                [2988.5795163815555, 0, 960],
                [0, 2988.3401159176124, 600],
                [0, 0, 1]
            ], dtype=np.float32)
            
        self.intrinsics = torch.from_numpy(self.intrinsics_mat)
        self.points_3d = self._load_keypoints(keypoints_file)

    def _load_keypoints(self, filename):
        path = filename if os.path.exists(filename) else os.path.join(self.image_root, filename)
        try:
            mat = scipy.io.loadmat(path)
            for k in ['keypoints', 'p_3D', 'tangoPoints', 'points']:
                if k in mat:
                    pts = np.array(mat[k], dtype=np.float32)
                    if pts.shape[0] == 3 and pts.shape[1] > 3: pts = pts.T
                    return pts
            keys = [k for k in mat.keys() if not k.startswith('__')]
            return np.array(mat[keys[0]], dtype=np.float32).T
        except Exception as e:
            print(f"Warning: Could not load {path}. Using dummy cube. Error: {e}")
            return np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                             [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def get_pose(self, row):
        t_vec = np.array([row['t_x'], row['t_y'], row['t_z']], dtype=np.float32)
        q_vec = [row['q_x'], row['q_y'], row['q_z'], row['q_w']]
        rot = R_scipy.from_quat(q_vec)
        r_mat = rot.as_matrix().astype(np.float32)
        return torch.from_numpy(r_mat), torch.from_numpy(t_vec)

    def _project_to_2d(self, R, T):
        P_cam = np.dot(self.points_3d, R.T) + T
        P_img = np.dot(P_cam, self.intrinsics_mat.T)
        z = P_img[:, 2:3]
        z[z < 1e-5] = 1e-5 
        return P_img[:, :2] / z

    def safe_crop(self, image_pil, x1, y1, x2, y2):
        w, h = image_pil.size
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            image_pil = ImageOps.expand(
                image_pil, 
                border=(pad_left, pad_top, pad_right, pad_bottom), 
                fill=0
            )
        crop_x1 = x1 + pad_left
        crop_y1 = y1 + pad_top
        crop_x2 = x2 + pad_left
        crop_y2 = y2 + pad_top
        
        return image_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    def __getitem__(self, idx):
            row = self.data.iloc[idx]
            filename = row['filename']
            if filename.startswith("synthetic/images/") and self.image_root.endswith("synthetic/images"):
                img_path = os.path.join(self.image_root.replace("synthetic/images", ""), filename)
            else:
                img_path = os.path.join(self.image_root, filename)

            img = cv2.imread(img_path)
            if img is None: 
                img = np.zeros((1200, 1920, 3), dtype=np.uint8)
            else: 
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            R_cam, T_cam = self.get_pose(row)
            K = self.intrinsics_mat
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            aug_results = self.augmentor.spatial_augmentations(
                img, 
                R_cam.numpy(), 
                T_cam.numpy(), 
                intrinsics=(fx, fy, cx, cy)
            )
            
            img_aug = aug_results['img_rot'] 
            R_aug = torch.from_numpy(aug_results['R_new']) 
            T_aug = torch.from_numpy(aug_results['T_new']) 
            kpts_2d = self._project_to_2d(R_aug.numpy(), T_aug.numpy())            
            x_min, y_min = np.min(kpts_2d, axis=0)
            x_max, y_max = np.max(kpts_2d, axis=0)
            w_tight = x_max - x_min
            h_tight = y_max - y_min
            max_dim = max(w_tight, h_tight)
            cx_tight, cy_tight = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0

            if self.mode == 'train':
                scale = np.random.uniform(1.1, 1.3)
                shift_x = np.random.uniform(-0.02, 0.02) * max_dim
                shift_y = np.random.uniform(-0.02, 0.02) * max_dim
            else:
                scale = 1.2
                shift_x, shift_y = 0.0, 0.0

            crop_size = max_dim * scale
            new_cx = cx_tight + shift_x
            new_cy = cy_tight + shift_y
            
            x1 = int(new_cx - crop_size / 2)
            y1 = int(new_cy - crop_size / 2)
            x2 = int(x1 + crop_size)
            y2 = int(y1 + crop_size)
            
            h_img, w_img = img_aug.shape[:2]
            pad_top = max(0, -y1)
            pad_bottom = max(0, y2 - h_img)
            pad_left = max(0, -x1)
            pad_right = max(0, x2 - w_img)
            
            if any([pad_top, pad_bottom, pad_left, pad_right]):
                img_aug = cv2.copyMakeBorder(img_aug, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
                x1 += pad_left; y1 += pad_top; x2 += pad_left; y2 += pad_top
                
            img_crop = img_aug[y1:y2, x1:x2]
            input_size = (224, 224) 
            if img_crop.size == 0: img_crop = cv2.resize(img_aug, input_size)
            else: img_crop = cv2.resize(img_crop, input_size)

            img_tensor = self.augmentor.pixel_augmentations(img_crop)
            R_target_input = R_aug.T 
            
            R_apparent = compute_apparent_rotation(R_target_input, T_aug)
            R6D_target = matrix_to_ortho6d(R_apparent)
            
            T_aug_batch = T_aug.unsqueeze(0)                
            bbox_batch = torch.tensor([new_cx, new_cy, float(crop_size), float(crop_size)], dtype=torch.float32).unsqueeze(0)
            intrinsics_batch = self.intrinsics.unsqueeze(0) 

            U_target = compute_normalized_coords_translation(
                T_aug_batch, bbox_batch, (self.W_full, self.H_full), input_size, intrinsics_batch 
            )
            
            U_target = U_target.squeeze(0) 
            crop_params = torch.tensor([new_cx, new_cy, float(crop_size), float(crop_size)], dtype=torch.float32)

            return {
                'pixel_values': img_tensor,
                'R6D_gt': R6D_target,
                'U_gt': U_target,
                'T_gt': T_aug,
                'crop_params': crop_params,
                'R_gt_true': R_target_input 
            }