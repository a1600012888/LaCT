from minVid.models.wan.wan_base.modules.tokenizers import HuggingfaceTokenizer
from minVid.models.wan.wan_base.modules.model import WanModel
from minVid.models.wan.wan_base.modules.hybrid_model import WanModel as WanModel_hybrid
from minVid.models.wan.wan_base.modules.vae import _video_vae
from minVid.models.wan.wan_base.modules.t5 import umt5_xxl
from minVid.models.wan.flow_match import FlowMatchScheduler
from minVid.scheduler import SchedulerInterface
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import types
from dataclasses import dataclass
import os

from minVid.utils.config_utils import instantiate_from_config, ObjectParamConfig

class WanDiffusionWrapper(nn.Module):
    @dataclass
    class Config:
        generator_ckpt: str
        model_config: ObjectParamConfig
        shift: float = 8.0
        extra_one_step: bool = True
        do_window_attention: bool = False

    def __init__(self, config):
        super().__init__()

        self.model = instantiate_from_config(config.model_config, split_config=True)

        ####### Pretrained model loading #######
        generator_ckpt = config.get("generator_ckpt", None)
        if generator_ckpt is not None and generator_ckpt != "None":
            if generator_ckpt.endswith(".pt"):
                # my trained checkpoint
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
            else:
                # pretrained checkpoint
                # TODO: add support for loading 14B ckpts later.
                if not generator_ckpt.endswith(".safetensors"):
                    generator_ckpt = os.path.join(generator_ckpt, "diffusion_pytorch_model.safetensors")

                print(f"Loading pretrained generator from {generator_ckpt}")
                from safetensors.torch import load_file
                state_dict = load_file(generator_ckpt)
            # Load state dict and capture any missing or unexpected keys
            load_result = self.model.load_state_dict(
                state_dict, strict=False
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
            print("WARNING: No pretrained diffusion model provided, using random initialization!")
        ####### Pretrained model loaded! #######
        self.model.eval()

        self.uniform_timestep = True # Default to True for normal DiT.

        # actually, this is the train scheduler, not necessarily the inference scheduler
        self.scheduler = FlowMatchScheduler(
            shift=config.shift, sigma_min=0.0, extra_one_step=config.extra_one_step
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        self.post_init()

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()

    def set_module_grad(self, module_grad: dict) -> None:
        """
        Adjusts the state of each module in the object.

        Parameters:
        - module_grad (dict): A dictionary where each key is the name of a module (as an attribute of the object), 
          and each value is a bool indicating whether the module's parameters require gradients.

        Functionality:
        For each module name in the dictionary:
        - Updates whether its parameters require gradients based on 'is_trainable'.
        """
        for k, is_trainable in module_grad.items():
            getattr(self, k).requires_grad_(is_trainable)

    def enable_gradient_checkpointing(self) -> None:
        # will enable gradient checkpointing in apply_model in the trainer script
        # self.model.enable_gradient_checkpointing()
        pass

    @staticmethod
    def return_act_ckpt_check_fn():
        """
        Will be used to apply activation checkpointing to the WanAttentionBlock.
        Check if the submodule is a WanAttentionBlock.
        """
        from minVid.models.wan.wan_base.modules.wan_model_warpper import WanAttentionBlock
        def _check_fn(submodule) -> bool:
            return isinstance(submodule, WanAttentionBlock)
        return _check_fn


    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, f, c, h, w]
        xt: the input noisy data with shape [B, f, c, h, w]
        timestep: the timestep with shape [B]

        pred = noise - x0 
        x_t = (1-sigma_t) * x0 + sigma_t * noise 
        we have x0 = x_t - sigma_t * pred 
        # or reverse:  flow_pred = (x_t - x0) / sigma_t
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e 
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t 
        Not used in the current implementation.
        Not used in wan model.
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, 
        convert_to_x0: bool = False,
        seq_len=None,
    ) -> torch.Tensor:
        """
        A method to run diffusion model. 
        Input: 
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings). 
                - prompt_embeds: a tensor with shape [B, L, D] containing the text embeddings.
            - timestep: a tensor with shape [B] 
        Output: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images. 
        We always expect a X0 prediction form for the output.
        """
        prompt_embeds = conditional_dict["prompt_embeds"]

        # print("debug wan input", noisy_image_or_video.shape, prompt_embeds.shape, timestep.shape)
        # [B, F] -> [B]
        if self.uniform_timestep and timestep.ndim == 2:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        # noise - x0 = flow_pred
        # List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        if seq_len is None:
            seq_len = self.seq_len

        flow_pred, extra_info_list = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=input_timestep, context=prompt_embeds,
            seq_len=seq_len
        )
        # if torch.distributed.get_rank() == 0:
        #     from pdb import set_trace; set_trace()
        flow_pred = torch.stack(flow_pred, dim=0) # [B, c, f, h, w]
        flow_pred = flow_pred.permute(0, 2, 1, 3, 4) # [B, f, c, h, w]
        if convert_to_x0:
            ret = self._convert_flow_pred_to_x0(
                flow_pred=flow_pred,
                xt=noisy_image_or_video,
                timestep=timestep
            ) # [B, f, c, h, w]
        else:
            ret = flow_pred

        return ret, extra_info_list
    
    def get_trainable_params(self, attn_only=True, **kwargs):
        return self.model.get_trainable_params(attn_only=attn_only, **kwargs)