"""
Editted upon from https://github.com/tianweiy/CausVid/blob/master/causvid/models/wan/bidirectional_inference.py
Inference pipeline for Wan model with autoregressive video latent flow matching. 
"""
from typing import List, Tuple, Literal
import torch
from dataclasses import dataclass
from tqdm import tqdm
from contextlib import contextmanager
import math

from minVid.models.wan.wan_base.modules.model import WanModel
from minVid.utils.config_utils import instantiate_from_config, ObjectParamConfig


from .wan_base.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)

from .wan_base.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import random
import sys
import os
from einops import rearrange

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

class WanInferencePipeline(torch.nn.Module):
    # notes about checkpoint loading:
    # there would be two checkpoint paths:
    # diffusion_config.generator_ckpt: the checkpoint path for the diffusion model, which is the pretrained wan model
    # config.generator_ckpt: the checkpoint path for the Wan model, which is the model training checkpoint for us. 
    
    sample_neg_prompt: str = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
    @dataclass
    class Config:
        generator_ckpt: str
        diffusion_config: ObjectParamConfig
        vae_config: ObjectParamConfig
        text_encoder_config: ObjectParamConfig
        mixed_precision: bool = True
        # below is inference config
        sample_timestep_shift: float = 3.0
        sample_solver: Literal["unipc", "dpm++"] = "unipc"
        sampling_steps: int = 50
        guide_scale: float = 5.0
        adjust_timestep_shift: bool = False


    def __init__(self, config: Config, device: torch.device = torch.device("cuda")):
        super().__init__()
        # Step 1: Initialize all models
        self.config = config
        self.device = device
        self.ar_window_size = config.ar_window_size
        print("Initializing WanInferencePipeline", self.config)
        
        if hasattr(config.diffusion_config, "model_config"):
            self.generator = instantiate_from_config(config.diffusion_config.model_config, split_config=True)

            ####### Finetuned model loading #######
            generator_ckpt = config.get("generator_ckpt", None)
            if generator_ckpt: # generator_ckpt could be False
                assert generator_ckpt is not None, "generator_ckpt is required"
                assert os.path.exists(generator_ckpt), f"generator_ckpt {generator_ckpt} does not exist"

                print(f"Loading pretrained generator from {generator_ckpt}")
                if generator_ckpt.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(generator_ckpt)
                else:
                    _state_dict = torch.load(generator_ckpt, map_location="cpu")
                    state_dict = {}
                    for key, value in _state_dict.items():
                        key = key.replace("_checkpoint_wrapped_module.", "")
                        key = key.replace("_orig_mod.", "")
                        while key.startswith("module."):
                            key = key[len("module."):]
                        if key.startswith("model."):
                            key = key[len("model."):]
                        state_dict[key] = value

                # Load state dict and capture any missing or unexpected keys
                load_result = self.generator.load_state_dict(
                    state_dict, strict=True
                )
                
                # Log information about the loading process
                if len(load_result.missing_keys) > 0:
                    print(f"Missing keys when loading model: {load_result.missing_keys}")
                if len(load_result.unexpected_keys) > 0:
                    print(f"Unexpected keys when loading model: {load_result.unexpected_keys}")
                    
                if len(load_result.missing_keys) == 0 and len(load_result.unexpected_keys) == 0:
                    print("Model loaded successfully with all keys matching")
                else:
                    print(f"Model loaded with {len(load_result.missing_keys)} missing keys and {len(load_result.unexpected_keys)} unexpected keys")
            else:
                print("No generator ckpt provided, don't load")
            ####### Finetuned model loaded! #######
        else:
            print("Using the original WanModel from: ", config.diffusion_config.generator_ckpt)
            self.generator = WanModel.from_pretrained(config.diffusion_config.generator_ckpt, use_safetensors=True, low_cpu_mem_usage=True)
            print("Pretrained WanModel loaded")
        
        self.generator.eval()


        self.text_encoder = instantiate_from_config(self.config.text_encoder_config)
        self.vae = instantiate_from_config(self.config.vae_config)

        # Step 2: Initialize all bidirectional wan hyperparmeters
        shift = config.get("sample_timestep_shift", 3.0)
        shift = float(shift)
        if self.config.adjust_timestep_shift and 0:
            self.shift = get_lin_function(32760, 73710, 3.0, 5.0)(self.ar_window_size * 1560)
            shift = self.shift
            print(f"Adjusted timestep shift to {self.shift}")
        sampling_steps = config.get("sampling_steps", 50)
        sample_solver = config.get("sample_solver", "unipc")
        guide_scale = config.get("guide_scale", 5.0)
        self.guide_scale = guide_scale
        self.sampling_steps = sampling_steps

        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=shift,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                shift=shift,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
    
        self.scheduler = sample_scheduler
        self.register_buffer("timesteps", timesteps)

        print("AR Wan Inference Pipeline: Sampling steps", sampling_steps, "with sample timestep shift", shift, "with ar window size", self.ar_window_size)
        


    def inference(self, noise: torch.Tensor, text_prompts: List[str], seed_g: torch.Generator, seq_len: int = 32760) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (num_channels, num_frames, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """

        n_prompt = self.sample_neg_prompt
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )  # "prompt_embeds" -> [B, L, D]
        neg_conditional_dict = self.text_encoder(
            text_prompts=n_prompt
        )
        context = conditional_dict['prompt_embeds'] # [B, L, D]
        context_neg = neg_conditional_dict['prompt_embeds']


        # initial point
        latents = [noise]

        timesteps = self.timesteps
        guide_scale = self.guide_scale

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.generator, 'no_sync', noop_no_sync)

        cleaned_latents_list = []
        latents_chunked = rearrange(noise, 'c (nw fw) h w -> c nw fw h w', fw=self.ar_window_size)

        num_total_chunks = latents_chunked.shape[1]

        first_chunk = True

        # Reset the inference state for autoregressive generation
        for name, module in self.generator.named_modules():
            if hasattr(module, 'inference_frame_offset'):
                module.inference_frame_offset = 0
            if hasattr(module, 'cfg_w0'):
                module.cfg_w0 = None
                module.cfg_w1 = None
                module.cfg_w2 = None
            if hasattr(module, 'cur_w0'):
                module.cur_w0 = None
                module.cur_w1 = None
                module.cur_w2 = None
                # print("debug reset inference frame offset", module.inference_frame_offset, module.cur_w0, module.cur_w1, module.cur_w2)
            if hasattr(module, 'kv_cache'):
                module.kv_cache = None
            if hasattr(module, 'kv_cache_cfg'):
                module.kv_cache_cfg = None
            if hasattr(module, 'dw0_momentum'):
                module.dw0_momentum = None
                module.dw1_momentum = None
                module.dw2_momentum = None
                module.cfg_dw0_momentum = None
                module.cfg_dw1_momentum = None
                module.cfg_dw2_momentum = None

        with no_sync(), torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():

            for chunk_idx in range(num_total_chunks):
                latent_model_input = latents_chunked[:, chunk_idx] # [c, fw, h, w]
                first_chunk = (chunk_idx == 0) 

                seq_len = math.prod(latent_model_input.shape[1:]) // 4

                # initialize the scheduler steps
                
                self.scheduler.set_timesteps(self.sampling_steps, device=self.device)
                timesteps = self.scheduler.timesteps
                for t_idx, t in enumerate(tqdm(timesteps)):

                    if chunk_idx == 0:
                        guide_scale = self.guide_scale
                    else:
                        # guide_scale = self.guide_scale * (1 - t_idx / self.sampling_steps)
                        guide_scale = self.guide_scale

                    if (not first_chunk) and t_idx == 0:
                        # repeat the batch dim here
                        cleaned_last_chunk = cleaned_latents_list[-1]
                        latent_model_input_repeated = [cleaned_last_chunk, latent_model_input]
                        
                        context_repeated = [context, context]
                        context_repeated = torch.cat(context_repeated, dim=0)
                        context_neg_repeated = [context_neg, context_neg]
                        context_neg_repeated = torch.cat(context_neg_repeated, dim=0)
                        t_clean = torch.zeros_like(t)
                        timestep_repeated = torch.stack([t_clean, t], dim=0)

                        seq_len_repeated = seq_len #  * 2

                        # print("debug repeat", latent_model_input_repeated[0].shape, latent_model_input_repeated[1].shape, context_repeated[0].shape, context_neg_repeated[0].shape, timestep_repeated.shape)
                    else:
                        # dont repeat the batch dim here
                        pass

                    
                    latent_model_input = [latent_model_input]
                    timestep = [t]

                    timestep = torch.stack(timestep)

                    if (not first_chunk) and t_idx == 0:
                        noise_pred_cond = self.generator(
                            latent_model_input_repeated, t=timestep_repeated, context=context_repeated, seq_len=seq_len_repeated)[0][1]
                        noise_pred_uncond = self.generator(
                            latent_model_input_repeated, t=timestep_repeated, context=context_neg_repeated, seq_len=seq_len_repeated)[0][1]
                        first_chunk = False
                        
                    else:
                        noise_pred_cond = self.generator(
                            latent_model_input, t=timestep, context=context, seq_len=seq_len)[0][0]
                        noise_pred_uncond = self.generator(
                            latent_model_input, t=timestep, context=context_neg, seq_len=seq_len)[0][0]

                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

                    temp_x0 = self.scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latent_model_input[0].unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latent_model_input = temp_x0.squeeze(0) # [ [c, f, h, w]  ] 
                
                    print("latent_model_input shape", latent_model_input.shape, "for chunk", chunk_idx, "current denoising step", t_idx)
                # add the cleaned latent to the list
                cleaned_latents_list.append(latent_model_input)

            latents = torch.cat(cleaned_latents_list, dim=1) # [c, total_f, h, w]
            x0 = latents.unsqueeze(0).transpose(1, 2) # [1, f, c, h, w]
            
            # [b, c, f, h, w]
            videos = self.vae.decode_to_pixel(x0)

        print("x0 shaoe", x0.shape, "videos shape", videos.shape)

        return videos