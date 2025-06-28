# Copyright (c) 2025 Tianyuan Zhang and Hao Tan 
# This software is released under the MIT License.
"""
Autoregressive Video Latent Flow Matching with teacher forcing training. 

Implemented features:
1. time-step shift and logit normal weighting
2. sequence parallel support
3. teacher forcing with repeated noisy chunks for better token utilization

Major difference between video_latent_flow_matching.py:
- _prepare_ar_input will arrange the input sequence as interleaved noisy-clean-xxx
- extra support for logit_normal_integral
- extra support for sequence parallel
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
from einops import rearrange, repeat
import math
from minVid.utils.logit_normal_weighting import logit_normal_integral

from minVid.models.wan.wan_base.distributed import sp_support
import torch.distributed as dist


def get_lin_function(x1: float = 32760, x2: float = 73710, y1=3.0, y2=5.0):
    """
    Get a linear function that maps x to log(y), and returns exp(y').
    f(4680) = 2.1135
    f(10920) = 2.2846
    """
    y1 = math.log(y1)
    y2 = math.log(y2)
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    return lambda x: math.exp(k * x + b)

class VideoLatentFlowMatching(nn.Module):
    @dataclass
    class Config:
        generator_ckpt: str
        diffusion_config: ObjectParamConfig
        vae_config: ObjectParamConfig
        text_encoder_config: ObjectParamConfig
        num_train_timestep: int = 1000
        timestep_shift: float = 3.0
        mixed_precision: bool = True
        timestep_sample_method: Literal["uniform", "logit_normal"] = "uniform"
        denoising_loss_type: Literal["flow"] = "flow"
        drop_text_prob: float = 0.0
        ar_window_size: int = 3 # frame window size for AR video Gen. 
        adjust_timestep_shift: bool = False
        num_repeat: int = 1 # number of times to repeat the noise. 
        frame_independent_noise: bool = True # if True, the noise is frame independent. 
        logit_normal_weighting: bool = True # if True, use logit normal weighting. 

    def __init__(self,
                 config: dict):
        super().__init__()
        self.config = config
        self.ar_window_size = self.config.ar_window_size

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
        # not used for now, since inference is done with another python file. 
        denoising_step_list = torch.arange(1, self.num_train_timestep+1)
        self.register_buffer("denoising_step_list", denoising_step_list)

        self.timestep_sample_method = self.config.timestep_sample_method
        self.denoising_loss_type = self.config.denoising_loss_type
        self.timestep_shift = self.config.timestep_shift

        if self.config.adjust_timestep_shift:
            self.timestep_shift = get_lin_function(32760, 73710, 3.0, 5.0)(self.ar_window_size * 1560)
            print(f"Adjusted timestep shift to {self.timestep_shift}")

        self.timestep_quantile = False
        self.dtype = torch.bfloat16 if self.config.mixed_precision else torch.float32
        self.drop_text_prob = self.config.drop_text_prob


        self.num_repeat = self.config.num_repeat
        self.frame_independent_noise = self.config.frame_independent_noise
        self.logit_normal_weighting = self.config.logit_normal_weighting
        self.logit_normal_weighting_std = self.config.get("logit_normal_weighting_std", 1.0)
        
        
    @torch.no_grad()
    def _prepare_input(self, video_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories, 
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - video_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_frames, num_channels, height, width].
        Output:
            - noisy_input: a tensor containing the selected latent
                [batch_size * num_repeat, num_ar_chunks, ar_window_size, num_channels, height, width].
            - noise: a tensor containing the noise 
                [batch_size * num_repeat, num_ar_chunks, ar_window_size, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep 
                [batch_size * num_repeat, num_ar_chunks].
                range from 0 - num_train_timestep
        
        # note the batch size is num_repeat * batch_size
        """
        # repeat the video_latent num_repeat times
        # [b * num_repeat, f, c, h, w]
        video_latent = rearrange(video_latent, 'b (nw fw) c h w -> b 1 nw fw c h w', fw=self.ar_window_size)
        video_latent_repeated = video_latent.repeat(1, self.num_repeat, 1, 1, 1, 1, 1)

        bs, _, num_ar_chunks, ar_window_size, c_latent, h_latent, w_latent = video_latent_repeated.shape
        device = video_latent.device


        if self.timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif self.timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if self.timestep_quantile is not None and self.timestep_quantile:
            t = dist.icdf(torch.full((bs,), self.timestep_quantile, device=device))
            raise NotImplementedError()
        else:
            t = dist.sample((bs, self.num_repeat, num_ar_chunks)).to(device)


        original_t = t
        # add timestep shift
        if self.timestep_shift > 0:
            t = t * self.timestep_shift / (1 + (self.timestep_shift-1) * t)

        noise = torch.randn_like(video_latent_repeated)
        t_expanded = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Add 4 dimensions to match video_latent shape

        # TODO, might add terminal snr. 
        noisy_input = (1 - t_expanded) * video_latent + t_expanded * noise

        t_train = t * self.num_train_timestep

        # flatten the first two dimensions
        noisy_input = rearrange(noisy_input, 'b nr nw fw c h w -> (b nr) nw fw c h w')
        noise = rearrange(noise, 'b nr nw fw c h w -> (b nr) nw fw c h w')
        t_train = rearrange(t_train, 'b nr nw -> (b nr) nw')
        video_latent_repeated = rearrange(video_latent_repeated, 'b nr nw fw c h w -> (b nr) nw fw c h w')

        # keep the original t for logit normal weighting. 
        original_t_train = original_t * self.num_train_timestep
        original_t_train = rearrange(original_t_train, 'b nr nw -> (b nr) nw')

        return noisy_input, noise, t_train, video_latent_repeated, original_t_train

    @torch.no_grad()
    def _prepare_ar_input(self, noisy_input, video_latent, t) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the input for the AR video generation.
        input: 
            noisy_input: [b * num_repeat, num_ar_chunks, ar_window_size, c, h, w]
            video_latent: [b, f, c, h, w]
            t: [b * num_repeat, num_ar_chunks]
        output: 
            ar_ret: [b, 2 * f - ar_window_size, c, h, w] -> [b * ( (num_repeat + 1) * num_window - 1), ar_window_size, c, h, w]
            ar_t: [b * ( (num_repeat + 1) * num_window - 1)]

        # minibatch style AR Video Diffusion. 
        # suppose F frames of the video
        --- When num_repeat = 1, the input sequence is:
        # noisy_f0, clean_f0, noise_f1, clean_f1, ..., clean_fL-1, noise_fL
        with total length L = 2F - ar_window_size
        --- When num_repeat = 2, the input sequence is: 
        # noisy_f0_repeat_1, noisy_f0_repeat_2, clean_f0, noise_f1_repeat_1, noise_f1_repeat_2, clean_f1, ..., clean_fL-1, noise_fL_repeat_1, noise_fL_repeat_2
        with total length L = 3F - ar_window_size
        """
        bs, f_latent, c_latent, h_latent, w_latent = video_latent.shape

        num_window_per_video = f_latent // self.ar_window_size

        # n_noisy_frames = f_latent * self.num_repeat; 
        # n_clean_frames = f_latent - self.ar_window_size
        total_num_frames = f_latent * self.num_repeat + f_latent - self.ar_window_size

        noise_chunk_size = self.ar_window_size * self.num_repeat

        ret = video_latent.new_zeros(bs, total_num_frames, c_latent, h_latent, w_latent)
        
        # [b, (num_repeat + 1) * num_window_per_video - 1]
        ar_t = t.new_zeros(bs, (self.num_repeat + 1) * num_window_per_video - 1)
        for window_idx in range(num_window_per_video):
            t_noise = t[:, window_idx] # [bs * num_repeat]
            t_noise = rearrange(t_noise, '(b nr) -> b nr', nr=self.num_repeat)

            start_idx = window_idx * (self.num_repeat + 1)
            end_idx = start_idx + self.num_repeat
            ar_t[:, start_idx: end_idx] = t_noise
        
        ar_t = ar_t.reshape(-1)

        # drop last clean chunk
        clean_interleave = video_latent[:, :-self.ar_window_size, :, :, :]
        clean_interleave = rearrange(clean_interleave, 'b (nw fw) c h w -> b nw fw c h w', fw=self.ar_window_size)
        
        # note, [b * num_repeat, num_window_per_video, ar_window_size, c, h, w]
        noise_interleave = noisy_input
        noise_interleave = rearrange(noise_interleave, '(b nr) nw fw c h w -> b nw (nr fw) c h w', nr=self.num_repeat)

        clean_noise_interleave = torch.cat([clean_interleave, noise_interleave[:, 1:, :, :, :, :]], dim=2)
        clean_noise_interleave = rearrange(clean_noise_interleave, 'b nw fw c h w -> b (nw fw) c h w')

        first_noise_chunk = noise_interleave[:, :1] # [b, num_repeat * ar_window_size, c, h, w]


        ret[:, :noise_chunk_size :, :, :] = first_noise_chunk
        ret[:, noise_chunk_size:, :, :, :] = clean_noise_interleave

        # [b * (num_repeat * num_ar_chunks + num_ar_chunks - 1), ar_window_size, c, h, w]
        ret = rearrange(ret, 'b (nw fw) c h w -> (b nw) fw c h w', fw=self.ar_window_size)

        return ret, ar_t

    def _extract_ar_output_from_interleave(self, interleave_output, video_latent) -> torch.Tensor:
        """
        Extract the denoised output from the interleaved sequence.
        Input: 
            interleave_output: [b * ( (num_repeat + 1) * num_window - 1), ar_window_size, c, h, w] - interleaved sequence of clean and denoised frames
            video_latent: [b, f, c, h, w]
        Output: 
            denoised_output: [b * num_repeat, num_ar_window, ar_window_size, c, h, w]
            denoised_output_latent: [b * num_repeat, num_ar_window, ar_window_size, c, h, w]
        
        The input sequence has the following structure:
        - First ar_window_size frames are noisy frames
        - Remaining frames alternate between clean and denoised frames
        We need to extract only the denoised frames to get the final output.
        """
        fake_bs, l, c, h, w = interleave_output.shape
        true_bs, f_latent, _, _, _ = video_latent.shape
        # Calculate the number of frames in the original video
        num_window_per_video = f_latent // self.ar_window_size

        pad_last_clean_chunk = interleave_output.new_zeros(true_bs, self.ar_window_size, c, h, w)

        # pad to [b * (num_repeat + 1) * num_window_per_video, ar_window_size, c, h, w]
        padded_interleave_output = torch.cat([interleave_output, pad_last_clean_chunk], dim=0)

        padded_interleave_output = rearrange(padded_interleave_output, '(b nw nr_plus_one) fw c h w -> b nr_plus_one nw fw c h w', b=true_bs, nr_plus_one=self.num_repeat + 1, nw=num_window_per_video)
        # drop the last item in each chunk. since it's the clean chunk.
        #  [b, (num_repeat), num_window, ar_window_size, c, h, w]
        denoised_output = padded_interleave_output[:, :-1, :, :, :, :]
        denoised_output = rearrange(denoised_output, 'b nr nw fw c h w -> (b nr) nw fw c h w', nr=self.num_repeat, nw=num_window_per_video)
        
        repeated_video_latent = repeat(video_latent, 'b f c h w -> (b nr) f c h w', nr=self.num_repeat)
        repeated_video_latent = rearrange(repeated_video_latent, '(b nr) (nw fw) c h w -> (b nr) nw fw c h w', nr=self.num_repeat, nw=num_window_per_video, fw=self.ar_window_size)

        return denoised_output, repeated_video_latent

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
        profile = False # weather to profile the forward pass
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

        # noisy_input: [b * num_repeat, num_ar_chunks, ar_window_size, c, h, w]
        # noise: [b * num_repeat, num_ar_chunks, ar_window_size, c, h, w]
        # t: [b * num_repeat, num_ar_chunks]
        noisy_input, noise, t, repeated_video_latent, original_t_train = self._prepare_input(video_latent)
        logit_normal_weighting = logit_normal_integral(
            rearrange(original_t_train, 'b nw -> (b nw)'),
            self.num_train_timestep, 0.0, self.logit_normal_weighting_std)
        logit_normal_weighting = rearrange(logit_normal_weighting, '(b nw) -> b nw', nw=t.shape[-1])

        # prepare the AR input 
        # ar_input: [b * ((num_repeat + 1) * num_window_per_video - 1), ar_window_size, c, h, w]
        # ar_t: [b * ((num_repeat + 1) * num_window_per_video - 1)]
        ar_input, ar_t = self._prepare_ar_input(noisy_input, video_latent, t)

        fake_b, f_latent_per_video, c, h, w = ar_input.shape
        ar_seq_len = (f_latent_per_video * h * w) // 4

        text_embeds_repeated = text_embeds.unsqueeze(1) # [b, 1, L, D]
        text_embeds_repeated = text_embeds_repeated.repeat(1, ar_input.shape[0] // text_embeds.shape[0], 1, 1) # [b, num_ar_chunks, L, D]
        text_embeds_repeated = rearrange(text_embeds_repeated, 'b nr l d -> (b nr) l d')

        if sp_support.is_sp():
            original_extended_bs = ar_input.size(0)
            ar_input = sp_support.sp_input_broadcast_scatter(ar_input, scatter_dim=0)
            text_embeds_repeated = sp_support.sp_input_broadcast_scatter(text_embeds_repeated, scatter_dim=0)
            ar_t = sp_support.sp_input_broadcast_scatter(ar_t, scatter_dim=0)
            # print(f"Rank: {dist.get_rank()}, sp rank {sp_support.get_sp_rank()}, Shape of ar_input: {ar_input.shape}")


        # flow_pred: [b * ( (num_repeat + 1) * num_window - 1), ar_window_size, c, h, w]
        flow_pred, extra_info_list = self.generator(
            ar_input.clone(),
            {"prompt_embeds": text_embeds_repeated},
            ar_t,
            convert_to_x0=False,
            seq_len=ar_seq_len
        )

        if sp_support.is_sp():
            flow_pred = sp_support.sp_all_gather(flow_pred, gather_dim=0, length=original_extended_bs)

        flow_pred, _ = self._extract_ar_output_from_interleave(flow_pred, video_latent)

        # shape: [b * num_repeat, num_ar_chunks, ar_window_size, c, h, w]
        gt_velocity = noise - repeated_video_latent
        if sp_support.is_sp():
            gt_velocity = sp_support.sp_broadcast(gt_velocity)
            logit_normal_weighting = sp_support.sp_broadcast(logit_normal_weighting)
        
        if self.denoising_loss_type == "flow":
            if self.logit_normal_weighting:
                logit_normal_weighting = logit_normal_weighting.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                l2_loss = F.mse_loss(flow_pred, gt_velocity, reduction="none")
                l2_loss = l2_loss * logit_normal_weighting
                l2_loss = l2_loss.mean()
            else:
                l2_loss = F.mse_loss(flow_pred, gt_velocity)
        else:
            raise NotImplementedError()

        if profile:
            torch.cuda.synchronize()
            profile_dict["flow_matching_forward"] = time.time() - start_time
            print(f"Time taken to compute flow matching forward: {profile_dict['flow_matching_forward']:06f}s")
            start_time = time.time()


        loss = l2_loss
        return_dict = {
            "loss": loss,       
            "loss_flow": l2_loss,
            "t": t, # of shape [b * num_repeat, num_ar_chunks]
        }
        vis_dict = {}
        
        vis_dict["loss_flow"] = l2_loss.item()

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
        

        # Compute loss at different frame indices without loop
        # Calculate MSE per frame, keeping batch dimension
        frame_wise_mse = ((flow_pred - gt_velocity) ** 2) # [b, num_ar_window, ar_window_size, c, h, w]
        frame_wise_mse = rearrange(frame_wise_mse, 'b nw fw c h w -> b nw (fw c h w)', fw=self.ar_window_size)
        # Average across batch dimension
        frame_wise_loss = frame_wise_mse.mean(dim=[0, 2])  # [n_frames]
        
        # Add to visualization dict
        for frame_idx, loss_val in enumerate(frame_wise_loss):
            vis_dict[f"loss_breakdown/loss_chunk_{frame_idx}"] = loss_val.item()

        return return_dict

    def get_trainable_params(self, attn_only=True, **kwargs):
        return self.generator.get_trainable_params(attn_only=attn_only, **kwargs)