from minVid.models.wan.wan_base.modules.tokenizers import HuggingfaceTokenizer
from minVid.models.wan.wan_base.modules.vae import _video_vae
from minVid.models.wan.wan_base.modules.t5 import umt5_xxl
from typing import List

import torch
import torch.nn as nn
from dataclasses import dataclass
import os

class WanTextEncoder(nn.Module):
    @dataclass
    class Config:
        model_path: str

    def __init__(self,
                 config: Config):
        super().__init__()
        
        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)

        # model_path looks like:
        # wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth
        if os.path.exists(config.model_path):
            self.text_encoder.load_state_dict(
                torch.load(config.model_path,
                           map_location='cpu', weights_only=False)
            )
        else:
            print(f"Warning: Text Encoder model {config.model_path} does not exist !!!")
        
        model_dir = os.path.dirname(config.model_path)
        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(model_dir, "google/umt5-xxl/"), seq_len=512, clean='whitespace')

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, text_prompts: List[str]) -> dict:
        """
        Input:
            - text_prompts: a list of text prompts.
        Output:
            dict:
                - prompt_embeds: a tensor with shape [B, L, D] containing the text embeddings.
        """
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }

class WanVAEWrapper(nn.Module):
    @dataclass
    class Config:
        model_path: str

    def __init__(self, 
                 config: Config):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)
        inv_std = 1.0 / std
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("inv_std", inv_std)

        # init model
        # model_path looks like:
        # wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
        self.model = _video_vae(
            pretrained_path=config.model_path,
            z_dim=16,
        ).eval().requires_grad_(False)

    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        """
        latents: [batch_size, num_frames, num_channels, height, width]
        output: [batch_size, num_channels, num_frames, height, width]
            in range [-1, 1]
        """
        zs = latent.permute(0, 2, 1, 3, 4)

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.decode(u.unsqueeze(0),
                              scale).float().clamp_(-1, 1).squeeze(0)
            for u in zs
        ]
        output = torch.stack(output, dim=0) # [B, C, F, H, W]
        return output

    def encode(self, videos_rgb: torch.Tensor) -> torch.Tensor:
        """
        Input:
            - videos_rgb: a tensor with shape [B, F+1, C, H, W] in RGB format, [0-1]
        Output:
            - video_rgb: a tensor with shape [B, F/Ts + 1, C, H/hs, W/ws] in RGB format, [0-1]
        """
        # [B, F+1, C, H, W] -> [B, C, F+1, H, W]
        videos_rgb = videos_rgb.permute(0, 2, 1, 3, 4)
        device, dtype = videos_rgb.device, videos_rgb.dtype
        # TODO: debug, check if 1.0/self.std is correct or self.std is correct
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        
        output_list =  [
                self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
                for u in videos_rgb
            ]
        output = torch.stack(output_list, dim=0) # [B, c, f, h, w]
        output = output.permute(0, 2, 1, 3, 4) # [B, f, c, h, w]
        return output
