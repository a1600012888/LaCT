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


parser = argparse.ArgumentParser()
# Basic info
parser.add_argument("--config", type=str, default="config/lact_l24_d768_ttt2x.yaml")
parser.add_argument("--load", type=str, default="weight/obj_res256.pt")
parser.add_argument("--data_path", type=str, default="data_example/gso_sample_data_path.json")
parser.add_argument("--output_dir", type=str, default="output/")
parser.add_argument("--num_all_views", type=int, default=32)

parser.add_argument("--num_input_views", type=int, default=20)
parser.add_argument("--num_target_views", type=int, default=None)
parser.add_argument("--image_size", nargs=2, type=int, default=[256, 256], help="Image size W, H")

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
dataset = NVSDataset(args.data_path, args.num_all_views, tuple(args.image_size))
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
        rendering = model(input_data_dict, target_cameras)
        video_path = os.path.join(output_dir, f"sample_{sample_idx:06d}_turntable.mp4")
        frames = (rendering[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        imageio.mimsave(video_path, frames, fps=30, quality=8)
        print(f"Saved turntable video to {video_path}")

            
                