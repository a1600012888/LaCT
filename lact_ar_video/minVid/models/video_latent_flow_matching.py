"""
Video Latent Flow Matching with Bidirectional Diffusion Transformer.
"""
import torch.nn.functional as F
from typing import Tuple, Literal
from torch import nn
import torch
import os
import random
import time
from dataclasses import dataclass


from minVid.utils.config_utils import instantiate_from_config, ObjectParamConfig


@torch.no_grad()
@torch._dynamo.disable
def compute_simple_latent_statistics(x, prefix=""):
    """
    x of shape [B, f, c, h, w]
    """
    ret_dict = {}
    ret_dict[f"{prefix}_mean"] = x.mean().item()
    ret_dict[f"{prefix}_std"] = x.std().item()
    ret_dict[f"{prefix}_min"] = x.min().item()
    ret_dict[f"{prefix}_max"] = x.max().item()
    ret_dict[f"{prefix}_abs_mean"] = x.abs().mean().item()
    ret_dict[f"{prefix}_abs_std"] = x.abs().std().item()
    ret_dict[f"{prefix}_abs_min"] = x.abs().min().item()
    ret_dict[f"{prefix}_abs_max"] = x.abs().max().item()
    return ret_dict

class VideoLatentFlowMatching(nn.Module):
    @dataclass
    class Config:
        generator_ckpt: str
        diffusion_config: ObjectParamConfig
        vae_config: ObjectParamConfig
        text_encoder_config: ObjectParamConfig
        num_train_timestep: int = 1000
        timestep_shift: float = 5.0
        mixed_precision: bool = True
        timestep_sample_method: Literal["uniform", "logit_normal"] = "uniform"
        denoising_loss_type: Literal["flow"] = "flow"
        drop_text_prob: float = 0.0

    def __init__(self,
                 config: dict):
        super().__init__()
        self.config = config


        self.generator = instantiate_from_config(self.config.diffusion_config)
        default_generator_requires_grad_dict = {"model": True}
        self.generator.set_module_grad(default_generator_requires_grad_dict)


        # set vae and text encoder to non-trainable
        self.vae = instantiate_from_config(self.config.vae_config)
        self.text_encoder = instantiate_from_config(self.config.text_encoder_config)    
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Step 2: Initialize all hyperparameters
        self.num_train_timestep = self.config.num_train_timestep
        # TODO: not used for now, since inference is done with another function. 
        denoising_step_list = torch.arange(1, self.num_train_timestep+1)
        self.register_buffer("denoising_step_list", denoising_step_list)

        self.timestep_sample_method = self.config.timestep_sample_method
        self.denoising_loss_type = self.config.denoising_loss_type
        self.timestep_shift = self.config.timestep_shift
        self.timestep_quantile = False
        self.dtype = torch.bfloat16 if self.config.mixed_precision else torch.float32
        self.drop_text_prob = self.config.drop_text_prob

        # for latent frame with zero noise, we probablistically perturb it with an extra small noise
        # self.extra_noise_step = getattr(args, "extra_noise_step", 0)
        # self.scheduler = self.generator.get_scheduler()
        
        
    @torch.no_grad()
    def _prepare_input(self, video_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories, 
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - ode_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, num_channels, height, width].
            - noise: a tensor containing the noise [batch_size, num_frames, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep [batch_size].
                range from 0 - num_train_timestep
        """

        bs, f_latent, c_latent, h_latent, w_latent = video_latent.shape
        device = video_latent.device

        if self.timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif self.timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if self.timestep_quantile is not None and self.timestep_quantile:
            t = dist.icdf(torch.full((bs,), self.timestep_quantile, device=device))
        else:
            t = dist.sample((bs,)).to(device)   

        # add timestep shift
        if self.timestep_shift > 0:
            t = t * self.timestep_shift / (1 + (self.timestep_shift-1) * t)

        noise = torch.randn_like(video_latent)
        t_expanded = t.view(-1, 1, 1, 1, 1)

        # TODO, might add terminal snr. 
        noisy_input = (1 - t_expanded) * video_latent + t_expanded * noise

        t_train = t * self.num_train_timestep
        return noisy_input, noise, t_train

    
    def forward(self, data_dict: dict) -> dict:
        """
        Only support training right now

        Input:
            - data_dict: a dictionary containing the input data.
                all data that are tensors should be on the same device as the model.
                - video_rgb: a tensor containing the video frames [batch_size, num_frames, 3, height, width] in RGB format, [0-1]
                - text_prompts: a list of text prompts.
        Output:
            - output_dict: a dictionary containing the output data.
        """
        profile = False
        profile_dict = {}
        
        if profile:
            torch.cuda.synchronize()
            start_time = time.time()

        text_prompts = data_dict["text_prompts"]
        if self.training and random.random() < self.drop_text_prob:
            text_prompts = [""] * len(text_prompts)
        text_embeds = self.text_encoder(text_prompts)["prompt_embeds"] # [B, L, D], padded with 0.0

        if profile:
            torch.cuda.synchronize()
            profile_dict["text_encoder"] = time.time() - start_time
            print(f"Time taken to encode text: {profile_dict['text_encoder']:06f}s")
            start_time = time.time()

        video_rgb = data_dict["video_rgb"] # [B, F+1, C, H, W]
        with torch.no_grad():
            video_latent = self.vae.encode(video_rgb * 2.0 - 1.0) # [B, f, c, h, w]
        if profile:
            torch.cuda.synchronize()
            profile_dict["video_vae_encode"] = time.time() - start_time
            print(f"Time taken to encode video: {profile_dict['video_vae_encode']:06f}s")
            start_time = time.time()

        noisy_input, noise, t = self._prepare_input(video_latent)

        flow_pred, extra_info_list = self.generator(
            noisy_input,
            {"prompt_embeds": text_embeds},
            t,
            convert_to_x0=False
        ) # [B, f, c, h, w]

        # flow matching loss
        if self.denoising_loss_type == "flow":
            gt_velocity = noise - video_latent
            loss = F.mse_loss(flow_pred, gt_velocity)
        else:
            raise NotImplementedError()

        if profile:
            torch.cuda.synchronize()
            profile_dict["flow_matching_forward"] = time.time() - start_time
            print(f"Time taken to compute flow matching forward: {profile_dict['flow_matching_forward']:06f}s")
            start_time = time.time()

        return_dict = {
            "loss": loss,
            "t": t,
        }
        vis_dict = {}

        vis_dict.update(compute_simple_latent_statistics(video_latent, "video_latent"))
        vis_dict.update(compute_simple_latent_statistics(flow_pred, "flow_pred"))
        vis_dict.update(compute_simple_latent_statistics(noise, "noise"))
        vis_dict.update(compute_simple_latent_statistics(noisy_input, "noisy_input"))
        vis_dict.update(compute_simple_latent_statistics(video_rgb, "video_rgb"))
        vis_dict.update(compute_simple_latent_statistics(gt_velocity, "gt_velocity"))

        # parse the extra info list
        if extra_info_list is not None:
            block_idx = 0
            extra_loss = 0.0
            for extra_info in extra_info_list:
                if extra_info is not None:
                    for key, value in extra_info.items():
                        vis_dict[f"block_{block_idx}/{key}"] = value
                        if key.startswith("loss"):
                            extra_loss += value
                block_idx += 1
            return_dict["extra_loss"] = extra_loss

        return_dict["vis_dict"] = vis_dict
        if profile:
            return_dict["profile_dict"] = profile_dict

        return return_dict

    def get_trainable_params(self, attn_only=True, **kwargs):
        return self.generator.get_trainable_params(attn_only=attn_only, **kwargs)