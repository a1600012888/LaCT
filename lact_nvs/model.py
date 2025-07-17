# LaCT

import math
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import functional as F

from lact_ttt import TTTOperator

def get_class_by_name(name):
    parts = name.split(".")
    module_name = ".".join(parts[:-1])
    class_name = parts[-1]
    
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
        module.reset_parameters()
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class SelfAttention(nn.Module):
    """
    Self-attention layer
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """

    def __init__(
        self,
        dim,
        head_dim,
        use_qk_norm=True,
        causal=False,
        bias=False,
    ):
        super().__init__()
        assert dim % head_dim == 0
        self.dim = dim
        self.head_dim = head_dim

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)
        self.use_qk_norm = use_qk_norm

        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim)
            self.k_norm = nn.RMSNorm(head_dim)

        self.causal = causal

    def forward(self, x, *args):
        """
        x: (b, l, d)
        """
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b l (qkv nh dh) -> qkv b nh l dh", qkv=3, dh=self.head_dim)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        x = rearrange(x, "b nh l dh -> b l (nh dh)")

        x = self.c_proj(x)
        return x, {}


class MLP(nn.Module):

    def __init__(self, dim, inter_multi=4, bias=False):
        super().__init__()
        intermediate_dim = int(dim * inter_multi)
        self.c_fc = nn.Linear(dim, intermediate_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(intermediate_dim, dim, bias=bias)

    def forward(self, x, *args):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x, {}


class Block(nn.Module):
    def __init__(self, dim, bias, block_config):
        super().__init__()
        module_list = []
        self.length_dim_list = []

        for _, module_config in enumerate(block_config):
            CLASS = get_class_by_name(module_config["type"])
            module = nn.ModuleDict(
                {
                    "ln": LayerNorm(dim, bias=bias),
                    "f": CLASS(dim=dim, bias=bias, **module_config["params"]),
                }
            )

            module_list.append(module)
            self.length_dim_list.append(module_config.get("length_dim", "vl"))

        self.module_list = nn.ModuleList(module_list)

    def forward(self, x, info):
        results = {}
        for module, length_dim in zip(self.module_list, self.length_dim_list):
            residual = x
            x = module["ln"](x)

            if length_dim == "l":
                b, vl, d = x.shape
                l = info["num_img_tokens"]
                x = x.reshape(b * (vl // l), l, d)
                x, result = module["f"](x, info)
                x = x.reshape(b, vl, d)
            else:
                x, result = module["f"](x, info)

            x = residual + x
            results.update(result)
        return x, results



def compute_rays(fxfycxcy, c2w, h, w):
    """Transform target before computing loss
    Args:
        fxfycxcy (torch.tensor): [b, v, 4]
        c2w (torch.tensor): [b, v, 4, 4]
    Returns:
        ray_o: (b, v, 3, h, w)
        ray_d: (b, v, 3, h, w)
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w, device=c2w.device)[None, :].expand(h, -1)  # [h, w]
    idx_y = torch.arange(h, device=c2w.device)[:, None].expand(-1, w)  # [h, w]

    # Reshape for batched matrix multiplication
    idx_x = idx_x.flatten().expand(b * v, -1)           # [b*v, h*w]
    idx_y = idx_y.flatten().expand(b * v, -1)           # [b*v, h*w]

    fxfycxcy = fxfycxcy.reshape(b * v, 4)               # [b*v, 4]
    c2w = c2w.reshape(b * v, 4, 4)                      # [b*v, 4, 4]

    x = (idx_x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]     # [b*v, h*w]
    y = (idx_y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]     # [b*v, h*w]
    z = torch.ones_like(x)                                      # [b*v, h*w]

    ray_d = torch.stack([x, y, z], dim=1)                       # [b*v, 3, h*w]
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d)                    # [b*v, 3, h*w]
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)      # [b*v, 3, h*w]

    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h*w)              # [b*v, 3, h*w]

    ray_o = ray_o.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]
    ray_d = ray_d.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]

    return ray_o, ray_d


class LaCTLVSM(nn.Module):
    def __init__(self, patch_size, dim, layers, block_config):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim

        self.pose_keys = ["ray_o", "ray_d", "o_cross_d"]
        self.posed_image_keys = self.pose_keys + ["normalized_image"]

        self.input_dim = len(self.posed_image_keys) * 3
        self.input_linear = nn.Linear(self.input_dim * (self.patch_size**2), self.dim, bias=False)
        self.input_layernorm = nn.LayerNorm(self.dim, bias=False)
        self.blocks = nn.ModuleList([
            Block(dim=self.dim, bias=False, block_config=block_config)
            for _ in range(layers)
        ])

        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.dim, bias=False),
            nn.Linear(self.dim, (self.patch_size**2) * 3, bias=False),
            nn.Sigmoid(),
        )

        # apply special scaled init to the residual projections, per GPT-2 paper
        self.apply(_init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(len(block_config) * layers))
    
    def forward(self, input_data_dict, target_data_dict):
            # Do not autocast during the data processing
        with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
            batch_size, num_input_views, _, h, w = input_data_dict["image"].size()
            num_target_views = target_data_dict["c2w"].size(1)

            for data_dict in [input_data_dict, target_data_dict]:
                fxfycxcy = data_dict["fxfycxcy"]
                c2w = data_dict["c2w"]

                data_dict["ray_o"], data_dict["ray_d"] = compute_rays(fxfycxcy, c2w, h, w)
                data_dict["o_cross_d"] = torch.cross(data_dict["ray_o"], data_dict["ray_d"], dim=2)
                data_dict["pose_only"] = torch.concat(
                    [data_dict[key] for key in self.pose_keys], dim=2
                )
                
                if "image" in data_dict:
                    data_dict["normalized_image"] = data_dict["image"] * 2.0 - 1.0

                    # Compile the information for posed-image input, and pose-only input.
                    data_dict["posed_image"] = torch.concat(
                        [data_dict[key] for key in self.posed_image_keys], dim=2
                    )
            
            transformer_input = input_data_dict["image"].new_zeros(
                batch_size, num_input_views + num_target_views, self.input_dim, h, w
            )  
            transformer_input[:, :num_input_views, :, :, :] = input_data_dict["posed_image"]
            pose_only_dim = target_data_dict["pose_only"].size(2)
            transformer_input[:, num_input_views:, :pose_only_dim, :, :] = target_data_dict["pose_only"]

        # Running the model
        num_img_tokens = h * w // (self.patch_size**2)
        num_input_tokens = num_input_views * num_img_tokens
        num_target_tokens = num_target_views * num_img_tokens
        ttt_op_order = [
            TTTOperator(start=0, end=num_input_tokens, update=True, apply=False),
            TTTOperator(start=0, end=num_input_tokens + num_target_tokens, update=False, apply=True),
        ]
        info = {
            "ttt_op_order": ttt_op_order,
            "num_img_tokens": num_img_tokens,
        }

        x = rearrange(
            transformer_input,
            "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = self.input_linear(x)
        x = self.input_layernorm(x)
        for block in self.blocks:
            x, _ = block(x, info)
        
        target_x = x[:, -num_target_tokens:]
        target_x = self.image_token_decoder(target_x)
        target_x = rearrange(
            target_x,
            "b (v hh ww) (ph pw c) -> b v c (hh ph) (ww pw)",
            v=num_target_views,
            hh=h // self.patch_size,
            ww=w // self.patch_size,
            ph=self.patch_size,
            pw=self.patch_size,
            c=3,
        )
        return target_x
    
    def reconstruct(self, input_data_dict):
        with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
            batch_size, num_input_views, _, h, w = input_data_dict["image"].size()

            fxfycxcy = input_data_dict["fxfycxcy"]
            c2w = input_data_dict["c2w"]

            input_data_dict["ray_o"], input_data_dict["ray_d"] = compute_rays(fxfycxcy, c2w, h, w)
            input_data_dict["o_cross_d"] = torch.cross(input_data_dict["ray_o"], input_data_dict["ray_d"], dim=2)
            input_data_dict["pose_only"] = torch.concat(
                [input_data_dict[key] for key in self.pose_keys], dim=2
            )
                
            input_data_dict["normalized_image"] = input_data_dict["image"] * 2.0 - 1.0

            # Compile the information for posed-image input, and pose-only input.
            posed_image = torch.concat(
                [input_data_dict[key] for key in self.posed_image_keys], dim=2
            )
            
        # Running the model
        num_img_tokens = h * w // (self.patch_size**2)
        num_input_tokens = num_input_views * num_img_tokens
        ttt_op_order = [
            TTTOperator(start=0, end=num_input_tokens, update=True, apply=True),
        ]
        info = {
            "ttt_op_order": ttt_op_order,
            "num_img_tokens": num_img_tokens,
        }

        x = rearrange(
            posed_image,
            "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = self.input_linear(x)
        x = self.input_layernorm(x)
        states = []
        for block in self.blocks:
            x, state = block(x, info)
            states.append(state)
        return states
    
    def rendering(self, target_data_dict, states, h, w):
        with torch.autocast(device_type="cuda", enabled=False):
            batch_size, num_target_views, _, _ = target_data_dict["c2w"].size()

            fxfycxcy = target_data_dict["fxfycxcy"]
            c2w = target_data_dict["c2w"]

            target_data_dict["ray_o"], target_data_dict["ray_d"] = compute_rays(fxfycxcy, c2w, h, w)
            target_data_dict["o_cross_d"] = torch.cross(target_data_dict["ray_o"], target_data_dict["ray_d"], dim=2)
            target_data_dict["pose_only"] = torch.concat(
                [target_data_dict[key] for key in self.pose_keys], dim=2
            )

            pose_only = target_data_dict["pose_only"].new_zeros(
                batch_size, num_target_views, self.input_dim, h, w
            )  
            pose_only_dim = target_data_dict["pose_only"].size(2)
            pose_only[:, :, :pose_only_dim, :, :] = target_data_dict["pose_only"]
            
        # Running the model for rendering
        num_img_tokens = h * w // (self.patch_size**2)
        num_target_tokens = num_target_views * num_img_tokens
        ttt_op_order = [
            TTTOperator(start=0, end=num_target_tokens, update=False, apply=True),
        ]
        info = {
            "ttt_op_order": ttt_op_order,
            "num_img_tokens": num_img_tokens,
        }

        # Process each target view separately
        all_x = []
        for v in range(num_target_views):
            single_view_pose = pose_only[:, v:v+1]  # b, 1, c, h, w
            
            x = rearrange(
                single_view_pose,
                "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
                ph=self.patch_size,
                pw=self.patch_size,
            )
            x = self.input_linear(x)
            x = self.input_layernorm(x)
            
            # Apply the saved states from reconstruction
            for block, state in zip(self.blocks, states):
                info.update(state)
                x, _ = block(x, info)
            
            all_x.append(x)
        
        # Concatenate all processed views
        x = torch.cat(all_x, dim=1)
            
        # Generate target images
        target_x = self.image_token_decoder(x)
        target_x = rearrange(
            target_x,
            "b (v hh ww) (ph pw c) -> b v c (hh ph) (ww pw)",
            v=num_target_views,
            hh=h // self.patch_size,
            ww=w // self.patch_size,
            ph=self.patch_size,
            pw=self.patch_size,
            c=3,
        )
        
        return target_x

    