import torch.nn.functional as F
from typing import Tuple, Literal
from torch import nn
import torch
import os

import time
from dataclasses import dataclass
from einops import rearrange
from einops.layers.torch import Rearrange

from minVid.utils.config_utils import instantiate_from_config, ObjectParamConfig
from typing import List


class VideoVAE(nn.Module):
    
    def __init__(self,
                 dim: int = 512,
                 patch_size: List[int] = [4, 8, 8],
                 inp_dim: int=3,
                 z_dim: int = 4,
                 num_layers: int=12,
                 block_config: ObjectParamConfig=None,
                 mlp_block_config: ObjectParamConfig=None):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.patch_size = patch_size

        self.inp_patchify = nn.Sequential(
            Rearrange('b c (f pt) (h ph) (w pw) -> b (f h w) (pt ph pw c)', pt=patch_size[0], ph=patch_size[1], pw=patch_size[2]),
            nn.Linear(patch_size[0] * patch_size[1] * patch_size[2] * inp_dim, dim),
        )

        self.transformer_input_layernorm = nn.LayerNorm(dim, bias=True)

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(instantiate_from_config(block_config, split_config=True))
            self.blocks.append(instantiate_from_config(mlp_block_config, split_config=True))

        self.transformer_output_layernorm = nn.LayerNorm(dim, bias=True)

        self.decoder = nn.Linear(dim, z_dim)

    def pad_video_frames(self, x):
        """
        x of shape [B, C, F, H, W]
        pad zeros to the begging frames, pad self.patch_size[0] -1 frames with all zeros in the beginning
        """
        B, C, F, H, W = x.shape
        padding_frames = self.patch_size[0] - 1
        
        if padding_frames > 0:
            # Create zero tensor for padding with same batch size, channels, height and width as input
            zero_pad = torch.zeros(B, C, padding_frames, H, W, device=x.device, dtype=x.dtype)
            
            # Concatenate the zero padding with the original tensor along the frame dimension
            padded_x = torch.cat([zero_pad, x], dim=2)
            return padded_x
        
        return x
        

    def forward(self, x):
        """
        x of shape [B, C, F, H, W]
        """
        
        x = self.pad_video_frames(x)

        x = self.inp_patchify(x)
        x = self.transformer_input_layernorm(x)
        
        for block in self.blocks:
            x = x + block(x, None, None, None)[0]

        x = self.transformer_output_layernorm(x)
        
        x = self.decoder(x)

        return x


@torch.no_grad()
def debug_vae():
    d = 1024
    use_swiglu = False
    if use_swiglu:
        block_config = {
            "class_name": "minVid.models.blocks.fast_weight_causal_swiglu.CausalFastWeightSwiGLU",
            "dim": d,
            "num_heads": 1,
            "qk_norm": True,
            "o_norm": True,
            "mini_batch_size": 6240 * 2,
            "qk_l2_norm": True,
            "w_init": "default",
            "inter_multi": 1,
            "lr_dim": 1
        }

    else:
        block_config = {
            "class_name": "minVid.models.blocks.linear_attention.BidirectionalFastWeightLinearSelfAttention",
            "dim": d,
            "num_heads": 2,
            "qk_norm": True,
            "o_norm": True,
            "mini_batch_size": 6240 * 2,
            "qk_l2_norm": True,
        }
    mlp_block_config = {
        "class_name": "minVid.models.blocks.fast_weight_causal_swiglu.MLP",
        "dim": d,
        "hidden_dim": d * 4,
        "out_dim": d,
    }

    # turn to ObjectParamConfig
    vae = VideoVAE(dim=d, patch_size=[4, 8, 8], inp_dim=3, z_dim=4, 
                   num_layers=12, block_config=block_config, mlp_block_config=mlp_block_config)
    vae = vae.to("cuda")
    vae.eval()

    # vae = torch.compile(vae)
    
    # Count and print number of parameters
    total_params = sum(p.numel() for p in vae.parameters()) / 1e6
    print(f"Total parameters: {total_params:.2f}M")
    
    B, C, F, H, W = 1, 3, 81, 480, 832
    x = torch.randn(B, C, F, H, W).to("cuda")
    
    # Run once to warm up
    with torch.no_grad():
        out = vae(x)
    
    # Speed test
    out = vae(x)
    num_iters = 10
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iters):
            _ = vae(x)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    avg_time_per_iter = elapsed_time / num_iters
    fps = num_iters / elapsed_time
    
    print(f"Speed test results:")
    print(f"Total time for {num_iters} iterations: {elapsed_time:.4f}s")
    print(f"Average time per iteration: {avg_time_per_iter*1:.4f} s")
    print(f"Throughput: {fps:.2f} iterations/second")

if __name__ == "__main__":
    debug_vae()