"""
Mostly copied from https://github.com/tianweiy/CausVid/blob/master/causvid/models/wan/bidirectional_inference.py
Inference pipeline for Wan model with bidirectional latent flow matching. 
"""
from typing import List, Tuple, Literal
import torch
from dataclasses import dataclass
from tqdm import tqdm
from contextlib import contextmanager

from minVid.models.wan.wan_base.modules.model import WanModel
from minVid.utils.config_utils import instantiate_from_config, ObjectParamConfig

# from minVid.models.wan.wan_base.utils.fm_solver import (FlowDPMSolverMultistepScheduler,
#                                get_sampling_sigmas, retrieve_timesteps)
# from minVid.models.wan.wan_base.fm_solvers_unipc import FlowUniPCMultistepScheduler

from .wan_base.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)

from .wan_base.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import random
import sys
import os

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
        sample_timestep_shift: float = 5.0
        sample_solver: Literal["unipc", "dpm++"] = "unipc"
        sampling_steps: int = 50
        guide_scale: float = 5.0


    def __init__(self, config: Config, device: torch.device = torch.device("cuda")):
        super().__init__()
        # Step 1: Initialize all models
        self.config = config
        self.device = device
        print("Initializing WanInferencePipeline", self.config)
        
        if hasattr(config.diffusion_config, "model_config"):
            self.generator = instantiate_from_config(config.diffusion_config.model_config, split_config=True)

            ####### Finetuned model loading #######
            generator_ckpt = config.get("generator_ckpt", None)
            if generator_ckpt is not None:
                print("Loading finetuned generator from", generator_ckpt)
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
                ####### Finetuned model loaded! #######
        else:
            print("Using the original WanModel from: ", config.diffusion_config.generator_ckpt)
            self.generator = WanModel.from_pretrained(config.diffusion_config.generator_ckpt, use_safetensors=True, low_cpu_mem_usage=True)
            print("Pretrained WanModel loaded")
        
        self.generator.eval()


        self.text_encoder = instantiate_from_config(self.config.text_encoder_config)
        self.vae = instantiate_from_config(self.config.vae_config)
        print("Device for generator, text_encoder, vae", self.generator.device, self.text_encoder.device)

        # Step 2: Initialize all bidirectional wan hyperparmeters
        shift = config.get("sample_timestep_shift", 5.0)
        shift = float(shift)
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

        # initialize the scheduler steps

        self.scheduler.set_timesteps(self.sampling_steps, device=self.device)
        timesteps = self.scheduler.timesteps


        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.generator, 'no_sync', noop_no_sync)

        with no_sync(), torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                noise_pred_cond = self.generator(
                    latent_model_input, t=timestep, context=context, seq_len=seq_len)[0][0]
                noise_pred_uncond = self.generator(
                    latent_model_input, t=timestep, context=context_neg, seq_len=seq_len)[0][0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = self.scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)] # [ [c, f, h, w]  ] 

            x0 = latents[0].unsqueeze(0).transpose(1, 2) # [1, f, c, h, w]
            
            # [b, c, f, h, w]
            videos = self.vae.decode_to_pixel(x0)

        print("debug x0 shaoe", x0.shape, "videos shape", videos.shape)

        return videos