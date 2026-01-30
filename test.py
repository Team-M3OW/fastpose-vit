import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils 
from tqdm import tqdm
import numpy as np
import os
import wandb
from fastpose_vit.dataloader import SpeedDataset
from model import FastPoseViT
from fastpose_vit.geometry_utils import (
    recover_true_rotation, 
    recover_translation_components, 
    ortho6d_to_matrix
)


def translation_loss(u_pred, u_gt):
    return torch.sum((u_pred - u_gt) ** 2, dim=1).mean()

def rotation_loss(r6d_pred, r6d_gt):
    return torch.sum((r6d_pred - r6d_gt) ** 2, dim=1).mean()

def rotmat2quat(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    assert R.shape[-2:] == (3, 3)
    r00, r11, r22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    trace = r00 + r11 + r22
    q = torch.zeros((*R.shape[:-2], 4), device=R.device, dtype=R.dtype)

    mask = trace > 0
    s = torch.sqrt(trace[mask] + 1.0) * 2.0
    q[mask, 0] = 0.25 * s
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s

    mask = (trace <= 0) & (r00 >= r11) & (r00 >= r22)
    s = torch.sqrt(1.0 + r00[mask] - r11[mask] - r22[mask]) * 2.0
    q[mask, 0] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
    q[mask, 1] = 0.25 * s
    q[mask, 2] = (R[mask, 0, 1] + R[mask, 1, 0]) / s
    q[mask, 3] = (R[mask, 0, 2] + R[mask, 2, 0]) / s

    mask = (trace <= 0) & (r11 > r00) & (r11 >= r22)
    s = torch.sqrt(1.0 + r11[mask] - r00[mask] - r22[mask]) * 2.0
    q[mask, 0] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
    q[mask, 1] = (R[mask, 0, 1] + R[mask, 1, 0]) / s
    q[mask, 2] = 0.25 * s
    q[mask, 3] = (R[mask, 1, 2] + R[mask, 2, 1]) / s

    mask = (trace <= 0) & (r22 > r00) & (r22 > r11)
    s = torch.sqrt(1.0 + r22[mask] - r00[mask] - r11[mask]) * 2.0
    q[mask, 0] = (R[mask, 1, 0] - R[mask, 0, 1]) / s
    q[mask, 1] = (R[mask, 0, 2] + R[mask, 2, 0]) / s
    q[mask, 2] = (R[mask, 1, 2] + R[mask, 2, 1]) / s
    q[mask, 3] = 0.25 * s
    q = q / (q.norm(dim=-1, keepdim=True) + eps)
    return q

def translation_error_euclidian(T_pred, T_gt):
    return torch.norm(T_pred - T_gt, dim=1).mean()

def rotation_error_geodesic(R_pred, R_gt):
    q_pred = rotmat2quat(R_pred)
    q_gt = rotmat2quat(R_gt)
    dot = torch.abs(torch.sum(q_pred * q_gt, dim=1))
    dot = torch.clamp(dot, -1.0, 1.0)
    E_r = 2 * torch.acos(dot)
    return torch.rad2deg(E_r).mean()

def save_debug_images(images, path="debug_input_batch.png"):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    denorm_imgs = images * std + mean
    denorm_imgs = torch.clamp(denorm_imgs, 0, 1)
    vutils.save_image(denorm_imgs, path, nrow=8)
    print(f"[DEBUG] Saved input batch grid to {path}")

def validate(model, loader, device, intrinsics):
    model.eval()
    val_loss = 0.0
    val_rot_loss = 0.0
    val_trans_loss = 0.0
    total_t_error = 0.0
    total_r_error = 0.0
    fx, fy, cx, cy = intrinsics
    K_batch = torch.eye(3, device=device).unsqueeze(0)
    K_batch[0, 0, 0] = float(fx)
    K_batch[0, 1, 1] = float(fy)
    K_batch[0, 0, 2] = float(cx)
    K_batch[0, 1, 2] = float(cy)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            images = batch['pixel_values'].to(device)
            r6d_gt = batch['R6D_gt'].to(device)
            u_gt = batch['U_gt'].to(device)
            
            outputs = model(images)
            u_pred = outputs['U']
            r6d_pred = outputs['R6D']
            
            l_trans = translation_loss(u_pred, u_gt)
            l_rot = rotation_loss(r6d_pred, r6d_gt)
            
            val_loss += (l_trans + l_rot).item()
            val_rot_loss += l_rot.item()
            val_trans_loss += l_trans.item()
            
            if 'T_gt' in batch and 'R_gt_true' in batch:
                T_gt = batch['T_gt'].to(device)
                R_gt_true = batch['R_gt_true'].to(device)
                crop_params = batch['crop_params'].to(device)

                # 1. Recover Translation
                T_pred = recover_translation_components(
                    u_pred, 
                    crop_params, 
                    (1920, 1200), 
                    (224, 224), 
                    K_batch
                )
                
                R_apparent_pred = ortho6d_to_matrix(r6d_pred)
                R_pred_true = recover_true_rotation(R_apparent_pred, T_gt) 
                
                t_err = translation_error_euclidian(T_pred, T_gt)
                total_t_error += t_err.item()
                
                r_err = rotation_error_geodesic(R_pred_true, R_gt_true)
                total_r_error += r_err.item()

    avg_loss = val_loss / len(loader)
    
    if total_t_error > 0:
        avg_t_error = total_t_error / len(loader)
        avg_r_error = total_r_error / len(loader)
        print(f"Val Metrics | Loss: {avg_loss:.4f} | T_Err: {avg_t_error:.4f} m | R_Err: {avg_r_error:.4f} deg")
    else:
        avg_t_error = 0.0
        avg_r_error = 0.0
        print(f"Val Metrics | Loss: {avg_loss:.4f} (Metrics skipped)")
         
    return avg_loss


if __name__ == "__main__":
    
    config = {
        "val_csv" : "/media/anil/hdd3/Arafat/ISRO PROJECT/Dataset/speedplusv2/lightbox/test_lightbox.csv",
        "img_folder" : "/media/anil/hdd3/Arafat/ISRO PROJECT/Dataset/speedplusv2",
        "intrinsic_file" : "/media/anil/hdd3/Arafat/ISRO PROJECT/Dataset/speedplusv2/camera.json",
        "keypoints_file": "/media/anil/hdd3/Arafat/ISRO PROJECT/Dataset/speedplusv2/tangoPoints.mat",        
        "model_path": "./checkpoints_gem/best_model.pth",
        "batch_size" : 180
        }

    val_dataset = SpeedDataset(
        csv_file=config["val_csv"], 
        image_root=config["img_folder"], 
        intrinsics_file=config["intrinsic_file"], 
        keypoints_file=config["keypoints_file"],
        mode='val'
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(val_dataset, 'intrinsics_mat'):
        K = val_dataset.intrinsics_mat
        val_intrinsics = (K[0,0], K[1,1], K[0,2], K[1,2])
    else:
        K = val_dataset.intrinsics
        if torch.is_tensor(K): K = K.numpy()
        val_intrinsics = (K[0,0], K[1,1], K[0,2], K[1,2])
    model = FastPoseViT(
    model_name="google/vit-base-patch16-224",  # or local HF folder
    pretrained=False
    )
    ckpt = torch.load(config["model_path"], map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.to(device)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss = validate(model, val_loader, device, val_intrinsics)
    print("Test Loss:", loss)


