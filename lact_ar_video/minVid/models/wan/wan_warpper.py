"""
from https://github.com/tianweiy/CausVid/blob/master/causvid/models/wan/wan_wrapper.py
"""
from minVid.models.wan.wan_base.modules.tokenizers import HuggingfaceTokenizer
from minVid.models.wan.wan_base.modules.model import WanModel
from minVid.models.wan.wan_base.modules.hybrid_model import WanModel as WanModel_hybrid
from minVid.models.wan.wan_base.modules.vae import _video_vae
from minVid.models.wan.wan_base.modules.t5 import umt5_xxl
from minVid.models.wan.flow_match import FlowMatchScheduler
from minVid.scheduler import SchedulerInterface
from typing import List, Tuple, Dict, Optional
from minVid.utils.config_utils import ObjectParamConfig

import torch
import torch.nn as nn
import types
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

            model_dir = os.path.dirname(config.model_path)

            self.tokenizer = HuggingfaceTokenizer(
                name=os.path.join(model_dir, "google/umt5-xxl/"), seq_len=512, clean='whitespace')
        else:
            print(f"Warning: Text Encoder model {config.model_path} does not exist !!!")

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




class WanDiffusionWrapper(nn.Module):
    @dataclass
    class Config:
        generator_ckpt: str
        shift: float = 3.0
        extra_one_step: bool = True
        do_window_attention: bool = False
        # model_config: ObjectParamConfig # not used in this class. 

    def __init__(self, config):
        super().__init__()

        generator_ckpt = config.get("generator_ckpt", "wan_models/Wan2.1-T2V-1.3B/")
        if config.get("do_window_attention", False):
            self.model = WanModel_hybrid.from_pretrained(generator_ckpt)
            print("using hybrid model")
        else:
            print("Loading original WanModel from: ", generator_ckpt)
            self.model = WanModel.from_pretrained(generator_ckpt)
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
        from minVid.models.wan.wan_base.modules.model import WanAttentionBlock
        from minVid.models.wan.wan_base.modules.hybrid_model import WanAttentionBlock as WanAttentionBlock_hybrid

        def _check_fn(submodule) -> bool:
            return isinstance(submodule, WanAttentionBlock) or isinstance(submodule, WanAttentionBlock_hybrid)
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
        convert_to_x0: bool = True
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
        flow_pred, _ = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=input_timestep, context=prompt_embeds,
            seq_len=self.seq_len
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

        return ret, None