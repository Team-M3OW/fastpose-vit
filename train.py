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
from dataloader import SpeedDataset
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

def validate(model, loader, device, intrinsics, epoch, name="val"):
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
    if name=="val":   
        wandb.log({
            "val/total_loss": avg_loss,
            "val/t_error_m": avg_t_error,
            "val/r_error_deg": avg_r_error,
            "epoch": epoch
        })
    else:
        wandb.log({
            f"{name}/total_loss": avg_loss,
            f"{name}/t_error_m": avg_t_error,
            f"{name}/r_error_deg": avg_r_error,
            "epoch": epoch
        })        
    return avg_loss

if __name__ == "__main__":
    config = {
        "train_csv":  "/media/anil/hdd3/Arafat/ISRO PROJECT/Dataset/speedplusv2/synthetic/train.csv",
        "sunlamp_csv" : "/media/anil/hdd3/Arafat/ISRO PROJECT/Dataset/speedplusv2/sunlamp/test_sunlamp.csv",
        "lightbox_csv" : "/media/anil/hdd3/Arafat/ISRO PROJECT/Dataset/speedplusv2/lightbox/test_lightbox.csv",
        "val_csv" : "/media/anil/hdd3/Arafat/ISRO PROJECT/Dataset/speedplusv2/synthetic/val.csv",
        "img_folder" : "/media/anil/hdd3/Arafat/ISRO PROJECT/Dataset/speedplusv2/",
        "intrinsic_file" : "/media/anil/hdd3/Arafat/ISRO PROJECT/Dataset/speedplusv2/camera.json",
        "keypoints_file": "/media/anil/hdd3/Arafat/ISRO PROJECT/Dataset/speedplusv2/tangoPoints.mat",
        "save_dir" : "./checkpoints_gem/",
        "batch_size" : 180,  
        "epochs" : 200,
        "lr" : 1e-4,
        "lr_min" : 1e-6,
        "project-name" : "fastpose-vit224-gem",
        "resume_checkpoint": "/media/anil/hdd3/Arafat/ISRO PROJECT/ARSH_ARNABI_G_2/fastpose_vit/checkpoints_gem/checkpoint_epoch_150.pth" 
    }

    os.makedirs(config['save_dir'], exist_ok=True)
    wandb_id = None
    if config["resume_checkpoint"]:
        wandb_id = "o43mhue2"    
    wandb.init(project=config["project-name"], config=config, id=wandb_id, resume="allow")
    train_dataset = SpeedDataset(
        csv_file=config["train_csv"], 
        image_root=config["img_folder"], 
        intrinsics_file=config["intrinsic_file"], 
        keypoints_file=config["keypoints_file"],
        mode='train'
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = SpeedDataset(
        csv_file=config["val_csv"], 
        image_root=config["img_folder"], 
        intrinsics_file=config["intrinsic_file"], 
        keypoints_file=config["keypoints_file"],
        mode='val'
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    sunlamp_dataset = SpeedDataset(
        csv_file=config["sunlamp_csv"], 
        image_root=config["img_folder"], 
        intrinsics_file=config["intrinsic_file"], 
        keypoints_file=config["keypoints_file"],
        mode='val'
    )
    sunlamp_loader = DataLoader(sunlamp_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    lightbox_dataset = SpeedDataset(
        csv_file=config["lightbox_csv"], 
        image_root=config["img_folder"], 
        intrinsics_file=config["intrinsic_file"], 
        keypoints_file=config["keypoints_file"],
        mode='val'
    )
    lightbox_loader = DataLoader(lightbox_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    if hasattr(val_dataset, 'intrinsics_mat'):
        K = val_dataset.intrinsics_mat
        val_intrinsics = (K[0,0], K[1,1], K[0,2], K[1,2])
    else:
        K = val_dataset.intrinsics
        if torch.is_tensor(K): K = K.numpy()
        val_intrinsics = (K[0,0], K[1,1], K[0,2], K[1,2])
    model = FastPoseViT(model_name='google/vit-base-patch16-224', pretrained=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=config['lr_min'])
    start_epoch = 0
    best_val_loss = float('inf')

    if config["resume_checkpoint"] and os.path.isfile(config["resume_checkpoint"]):
        print(f"Loading checkpoint: {config['resume_checkpoint']}")
        checkpoint = torch.load(config["resume_checkpoint"], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1 # Start from the NEXT epoch
        
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']

        print(f"Resumed from Epoch {start_epoch}. Previous Best Val Loss: {best_val_loss:.4f}")
    else:
        if config["resume_checkpoint"]:
            print(f"{config['resume_checkpoint']} not found. Starting from scratch.")

    print(f"Starting Training on {device}...")
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_rot_loss = 0.0
        epoch_trans_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(loop):
            images = batch["pixel_values"].to(device)
            R6D_gt = batch["R6D_gt"].to(device)
            U_gt = batch["U_gt"].to(device)
            if epoch == 0 and batch_idx == 0:
                save_debug_images(images, "debug_input_batch.png")

            optimizer.zero_grad()
            outputs = model(images)
            R6D_pred = outputs["R6D"]
            U_pred = outputs["U"]
            
            loss_rot = rotation_loss(R6D_pred, R6D_gt)
            loss_trans = translation_loss(U_pred, U_gt)
            loss = loss_rot + loss_trans
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_rot_loss += loss_rot.item()
            epoch_trans_loss += loss_trans.item()
            
            loop.set_postfix(loss=loss.item())
            
            wandb.log({
                "train/batch_total_loss": loss.item(),
                "train/batch_rot_loss": loss_rot.item(),
                "train/batch_trans_loss": loss_trans.item()
            })
        val_loss = validate(model, val_loader, device, val_intrinsics, epoch+1)
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        avg_rot_loss = epoch_rot_loss / len(train_loader)
        avg_trans_loss = epoch_trans_loss / len(train_loader)      
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} (Rot: {avg_rot_loss:.4f}, Trans: {avg_trans_loss:.4f})")
        wandb.log({
            "train/epoch_total_loss": avg_train_loss,
            "train/epoch_rot_loss": avg_rot_loss,
            "train/epoch_trans_loss": avg_trans_loss,
            "epoch": epoch+1,
            "lr": optimizer.param_groups[0]['lr']
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config['save_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, save_path)
            print(f"Saved best model to {save_path}")
            print("Evaluating on Sunlamp and Lightbox datasets...")
            print("---------------------------------- Sunlamp Validation --------------------------------")
            _ = validate(model, sunlamp_loader, device, val_intrinsics, epoch+1, name="sunlamp")
            print("---------------------------------- Lightbox Validation--------------------------------#")
            _ = validate(model, lightbox_loader, device, val_intrinsics, epoch+1, name="lightbox")
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), # Added scheduler
                'best_val_loss': best_val_loss,                 # Added best_val_loss
                'loss': avg_train_loss,
            }, save_path)
            print(f"Saved checkpoint to {save_path}")