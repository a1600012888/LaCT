import argparse
import random
import os

import imageio
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup
from PIL import Image

from data import NVSDataset
from model import LaCTLVSM

def get_turntable_cameras_with_zoom_in(
    batch_size=1,
    hfov=50,
    num_views=8,
    w=256,
    h=256,
    min_radius=1.7,
    max_radius=3.0,
    elevation=30,
    up_vector=np.array([0, 0, 1]),
    device=torch.device("cuda"),
):
    '''
    rotate the camera around the object, and change the radius and elevation periodically
    '''
    fx = w / (2 * np.tan(np.deg2rad(hfov) / 2.0))
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    fxfycxcy = np.array([fx, fy, cx, cy]).reshape(1, 4).repeat(num_views, axis=0) # [num_views, 4]
    azimuths = np.linspace(0, 360, num_views, endpoint=False)
    elevations = np.ones_like(azimuths) * (elevation + 15 * np.sin(np.linspace(0, 2*np.pi, num_views)))
    radius = (min_radius + max_radius) / 2.0 + (max_radius - min_radius) / 2.0 * np.sin(np.linspace(0, 2*np.pi, num_views))
    c2ws = []

    for cur_radius, elev, azim in zip(radius, elevations, azimuths):
        elev, azim = np.deg2rad(elev), np.deg2rad(azim)
        z = cur_radius * np.sin(elev)
        base = cur_radius * np.cos(elev)
        x = base * np.cos(azim)
        y = base * np.sin(azim)
        cam_pos = np.array([x, y, z])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(forward, up_vector)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        R = np.stack((right, -up, forward), axis=1)
        c2w = np.eye(4)
        c2w[:3, :4] = np.concatenate((R, cam_pos[:, None]), axis=1)
        c2ws.append(c2w)
    c2ws = np.stack(c2ws, axis=0)  # [num_views, 4, 4]

    # Expand from [num_views, *] to [batch_size, num_views, *]
    fxfycxcy = fxfycxcy[None, ...].repeat(batch_size, axis=0) # [batch_size, num_views, 4]
    c2ws = c2ws[None, ...].repeat(batch_size, axis=0)
    return {
        "w": w,
        "h": h,
        "num_views": num_views,
        "fxfycxcy": torch.from_numpy(fxfycxcy).to(device).float(),
        "c2w": torch.from_numpy(c2ws).to(device).float(),
    }


def get_interpolated_cameras(
    cameras,
    num_views,
):
    """
    For each consecutive pair of cameras, add num_views linearly interpolated views.
    """
    fxfycxcy = cameras['fxfycxcy']  # [batch_size, num_input_views, 4]
    c2w = cameras['c2w']  # [batch_size, num_input_views, 4, 4]
    
    batch_size, num_input_views = fxfycxcy.shape[:2]
    
    interpolated_fxfycxcy = []
    interpolated_c2w = []
    
    for b in range(batch_size):
        batch_fxfycxcy = []
        batch_c2w = []
        
        for i in range(num_input_views - 1):
            # Add the current view
            batch_fxfycxcy.append(fxfycxcy[b, i])
            batch_c2w.append(c2w[b, i])
            
            curr_fxfycxcy = fxfycxcy[b, i]
            next_fxfycxcy = fxfycxcy[b, i + 1]
            curr_c2w = c2w[b, i]
            next_c2w = c2w[b, i + 1]
            
            # Create alpha values for all interpolations at once
            alphas = torch.linspace(1 / (num_views + 1), num_views / (num_views + 1), num_views, device=fxfycxcy.device)
            
            # Batch interpolation for camera intrinsics
            interp_fxfycxcy = (1 - alphas[:, None]) * curr_fxfycxcy[None, :] + alphas[:, None] * next_fxfycxcy[None, :]
            batch_fxfycxcy.extend(interp_fxfycxcy)
            
            # Batch interpolation for camera poses
            # For rotation, we should use SLERP, but for simplicity using linear interpolation
            interp_c2w = (1 - alphas[:, None, None]) * curr_c2w[None, :, :] + alphas[:, None, None] * next_c2w[None, :, :]
            batch_c2w.extend(interp_c2w)
        
        # Add the last view
        batch_fxfycxcy.append(fxfycxcy[b, -1])
        batch_c2w.append(c2w[b, -1])
        
        interpolated_fxfycxcy.append(torch.stack(batch_fxfycxcy))
        interpolated_c2w.append(torch.stack(batch_c2w))
    
    return {
        'fxfycxcy': torch.stack(interpolated_fxfycxcy),
        'c2w': torch.stack(interpolated_c2w)
    }



parser = argparse.ArgumentParser()
# Basic info
parser.add_argument("--config", type=str, default="config/lact_l24_d768_ttt2x.yaml")
parser.add_argument("--load", type=str, default="weight/obj_res256.pt")
parser.add_argument("--data_path", type=str, default="data_example/gso_sample_data_path.json")
parser.add_argument("--output_dir", type=str, default="output/")
parser.add_argument("--num_all_views", type=int, default=32)

parser.add_argument("--num_input_views", type=int, default=20)
parser.add_argument("--num_target_views", type=int, default=None)
parser.add_argument("--scene_inference", action="store_true")
parser.add_argument("--image_size", nargs=2, type=int, default=[256, 256], help="Image size H, W")

args = parser.parse_args()
if args.num_target_views is None:
    args.num_target_views = args.num_all_views - args.num_input_views
model_config = omegaconf.OmegaConf.load(args.config)
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Create output directory if specified
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# Seed everything
seed = 95
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
model = LaCTLVSM(**model_config).cuda()

# Load checkpoint
print(f"Loading checkpoint from {args.load}...")
checkpoint = torch.load(args.load, map_location="cpu")
model.load_state_dict(checkpoint["model"])

# Data
dataset = NVSDataset(args.data_path, args.num_all_views, tuple(args.image_size), sorted_indices=args.scene_inference, scene_pose_normalize=args.scene_inference)
dataloader_seed_generator = torch.Generator()
dataloader_seed_generator.manual_seed(seed)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    generator=dataloader_seed_generator,    # This ensures deterministic dataloader
)


for sample_idx, data_dict in enumerate(dataloader):
    data_dict = {key: value.cuda() for key, value in data_dict.items() if isinstance(value, torch.Tensor)}
    if args.scene_inference:
        # Randomly select input views and use remaining as target
        total_views = data_dict["image"].shape[1]
        all_indices = torch.randperm(total_views)
        input_indices = torch.sort(all_indices[:args.num_input_views])[0]   # Sort for video rendering only; model forward is permutation-invariant
        target_indices = all_indices[-args.num_target_views:]
        
        input_data_dict = {key: value[:, input_indices] for key, value in data_dict.items()}
        target_data_dict = {key: value[:, target_indices] for key, value in data_dict.items()}
    else:
        input_data_dict = {key: value[:, :args.num_input_views] for key, value in data_dict.items()}
        target_data_dict = {key: value[:, -args.num_target_views:] for key, value in data_dict.items()}

    with torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True) and torch.no_grad():
        rendering = model(input_data_dict, target_data_dict)

        target = target_data_dict["image"]
        psnr = -10.0 * torch.log10(F.mse_loss(rendering, target)).item()

        print(f"Sample {sample_idx}: PSNR = {psnr:.2f}")
        
        # Save rendered images if output directory is specified
        if output_dir:
            def save_image_rgb(tensor, filepath):
                """Save tensor as RGB image."""
                numpy_image = tensor.permute(1, 2, 0).cpu().numpy()
                numpy_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)
                Image.fromarray(numpy_image, mode='RGB').save(filepath)

            batch_size, num_views = rendering.shape[:2]
            for batch_idx in range(batch_size):
                for view_idx in range(num_views):
                    # Save rendered and target images
                    for img_type, img_tensor in [("rendered", rendering[batch_idx, view_idx]), 
                                                 ("target", target[batch_idx, view_idx])]:
                        filename = f"sample_{sample_idx:06d}_view_{view_idx:02d}_{img_type}.png"
                        save_image_rgb(img_tensor, os.path.join(output_dir, filename))
            
            print(f"Saved images for sample {sample_idx} to {output_dir}")
        
        # Rendering a video to circularly rotate the camera views
        if args.scene_inference:
            target_cameras = get_interpolated_cameras(
                cameras=input_data_dict,
                num_views=2,
            )
        else:
            target_cameras = get_turntable_cameras_with_zoom_in(
                batch_size=1,
                num_views=120,
                w=args.image_size[0],
                h=args.image_size[1],
                min_radius=1.7,
                max_radius=3.0,
                elevation=30,
                up_vector=np.array([0, 0, 1]),
                device=torch.device("cuda"),
            )
        print(target_cameras["c2w"].shape, target_cameras["fxfycxcy"].shape)
        states = model.reconstruct(input_data_dict)
        rendering = model.rendering(target_cameras, states, args.image_size[0], args.image_size[1])
        video_path = os.path.join(output_dir, f"sample_{sample_idx:06d}_turntable.gif")
        frames = (rendering[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        imageio.mimsave(video_path, frames, fps=30, quality=8)
        print(f"Saved turntable video to {video_path}")

            
                