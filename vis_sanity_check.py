import torch
import cv2
import numpy as np
import os
from torch.utils.data import DataLoader
from dataloader import SpeedDataset

# --- CONFIGURATION ---
CONFIG = {
    "csv_file": "/media/anil/hdd3/Arafat/ISRO PROJECT/Dataset/speedplusv2/synthetic/train.csv",
    "img_root": "/media/anil/hdd3/Arafat/ISRO PROJECT/Dataset/speedplusv2/",
    "intrinsics": "/media/anil/hdd3/Arafat/ISRO PROJECT/Dataset/speedplusv2/camera.json",
    "keypoints": "/media/anil/hdd3/Arafat/ISRO PROJECT/Dataset/speedplusv2/tangoPoints.mat"
}

def draw_axis(img, R, T, K, len=0.3):
    points_3d = np.float32([[0,0,0], [len,0,0], [0,len,0], [0,0,len]])
    points_cam = (R @ points_3d.T).T + T
    points_2d = (K @ points_cam.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]
    points_2d = points_2d.astype(int)
    
    origin = tuple(points_2d[0])
    img = cv2.line(img, origin, tuple(points_2d[1]), (0,0,255), 3) # X - Red
    img = cv2.line(img, origin, tuple(points_2d[2]), (0,255,0), 3) # Y - Green
    img = cv2.line(img, origin, tuple(points_2d[3]), (255,0,0), 3) # Z - Blue
    return img

def main():
    dataset = SpeedDataset(
        csv_file=CONFIG["csv_file"],
        image_root=CONFIG["img_root"],
        intrinsics_file=CONFIG["intrinsics"],
        keypoints_file=CONFIG["keypoints"],
        mode='train'
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    batch = next(iter(loader))
    img_tensor = batch['pixel_values'][0] # (3, 224, 224)
    R_gt = batch['R_gt_true'][0].numpy()  # (3, 3) This is R_aug.T from dataloader
    T_gt = batch['T_gt'][0].numpy()       # (3,)
    crop_params = batch['crop_params'][0].numpy() # cx, cy, w, h
    mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    img = (img_tensor.numpy() * std + mean) * 255
    img = img.transpose(1,2,0).astype(np.uint8).copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cx_crop, cy_crop, w_crop, h_crop = crop_params
    K_orig = dataset.intrinsics_mat.copy()
    crop_x1 = cx_crop - w_crop/2
    crop_y1 = cy_crop - h_crop/2
    
    K_new = K_orig.copy()
    K_new[0, 2] -= crop_x1
    K_new[1, 2] -= crop_y1
    scale = 224.0 / w_crop
    K_new[:2, :] *= scale
    vis = draw_axis(img, R_gt, T_gt, K_new, len=1.0) # len in meters
    
    cv2.imwrite("debug_visual_alignment.png", vis)
if __name__ == "__main__":
    main()