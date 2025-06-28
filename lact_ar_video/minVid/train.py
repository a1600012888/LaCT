import argparse
import os
import time
import numpy as np
import random
from collections import defaultdict

import torch
import torch.distributed as dist
import wandb
import yaml
import copy
import math
from omegaconf import OmegaConf
from easydict import EasyDict as edict

from contextlib import nullcontext

from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    ModuleWrapPolicy,
)

from minVid.utils.config_utils import instantiate_from_config, set_nested_key, get_obj_from_str
from minVid.utils.dist_utils import (
    launch_distributed_job, set_seed, print_rank0,
    init_logging_folder, fsdp_wrap, cycle, fsdp_state_dict
)
from minVid.utils.optimizer_scheduler import configure_lr_scheduler
from minVid.utils.io_utils import save_video
from minVid.utils.job_checkpoint_fsdp import (
    save_job, resume_job_dcp, find_latest_checkpoint, clear_old_checkpoints
)
from minVid.utils.ema_param_utils import EMAParams
from transformers.models.t5.modeling_t5 import T5Block

from minVid.data import get_data_module # TODO: change to your own data module
import torch.nn.functional as F
import gc # for garbage collection
import datetime
from functools import partial
import logging

def init_logger():
    rank = int(os.getenv("RANK", 0))       # torchrun injects this
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log message format
        datefmt="%Y-%m-%d %H:%M:%S",  # Define the date format      
        handlers=[
            logging.StreamHandler(),                           # console
        ],
    )
    # Inject the rank into the log record:
    logging.LoggerAdapter(logging.getLogger(), {"rank": rank})
init_logger()


def shard_model_set_ema_set_optimizer(model, config, global_rank, process_info=None):
    """
    config: config object should have the following keys:
    - train: train config.
        - train.fsdp_modules: list of modules to wrap with FSDP.
        - train.fsdp_strategy: strategy to use for FSDP. "full" or "hybrid_full"
        - train.fsdp_wrap_strategy: strategy to use for FSDP wrap. "size" or "module"
        - train.use_ema: whether to use ema.
        - train.ema_weight: ema weight.
        - train.lr_multiplier: multiplier for the learning rate of the newly initialized weights. [optional]
        - train.mixed_precision: default bfloat16.
    - lr: learning rate.
    - weight_decay: weight decay.
    - beta1: beta1 for the AdamW optimizer.
    - beta2: beta2 for the AdamW optimizer.
    Setup the model for training.  Model is already on CPU or meta device. 
    Step-1: extract a dict of the model params, keep track of shape and requires_grad. 
    Step-2: Warp the model with FSDP.
    Step-3: Warp the model with activation checkpointing.
    Step-4: Compile the model [optional]
    Step-5: Setup the optimizer.
    Step-6: Setup the ema.
    Step-7: TODO: support for checkpointing manager and resume! 
    """
    num_total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"=> Number of total parameters: {num_total_params / 1e6}M")
    name_to_orig_shape = {} # for determin weight decay. No weight decay for 1D params.
    name_to_orig_numel = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_to_orig_shape[name] = param.shape
        name_to_orig_numel[name] = param.numel()
    
    # set up the distributed attention if the model has it
    if hasattr(model, "setup_dist_attn"):
        # TODO: add process info
        model.setup_dist_attn(process_info)
    ######## Step-2: Sharding the model with FSDP ########
    # need to setup: 
    # fsdp_modules -> ModuleWrapPolicy
    # fsdp_wrap_policy -> size or module
    # fsdp_strategy -> full, hybrid_full, hybrid_zero2, no_shard
    # mixed_precision -> bfloat16 for params, reduce, buffer
    # use_orig_params -> True or False
    fsdp_modules = []
    for _m in config.train.fsdp_modules:
        _m_class = get_obj_from_str(_m)
        fsdp_modules.append(_m_class)
    fsdp_wrap_policy_dict = {
        "size": partial(size_based_auto_wrap_policy, min_num_params=int(5e7)),
        "module": ModuleWrapPolicy(fsdp_modules),
    }
    fsdp_wrap_policy = fsdp_wrap_policy_dict[config.train.get("fsdp_wrap_policy", "size")]
    fsdp_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    fsdp_strategy = fsdp_strategy[config.train.get("fsdp_strategy", "full")]
   
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    model = FSDP(model, 
                 sharding_strategy=fsdp_strategy, 
                 auto_wrap_policy=fsdp_wrap_policy,
                 mixed_precision=mixed_precision,
                 device_id=torch.cuda.current_device(),
                 use_orig_params=True,
                 limit_all_gathers=True,
                 sync_module_states=False, # Load ckpt on rank 0 and sync to other ranks
                 )
    
    ### Step-3: Warp the model with activation checkpointing ####
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper as ptd_checkpoint_wrapper,
        apply_activation_checkpointing,
    )
    wrapper = partial(ptd_checkpoint_wrapper, preserve_rng_state=False)
    # TODO: current version needs model to implement a return_act_ckpt_check_fn() method 
    # TODO: for future versions, we can implement a wrapper that takes config.trian.act_ckpt_modules 
    # TODO: and apply the checkpointing to the modules specified in the list. 
    _check_fn = model.module.return_act_ckpt_check_fn()
    apply_activation_checkpointing(model, wrapper, check_fn=_check_fn)

    ### Step-4: Compile the model ####
    if config.train.get("compile", False):
        # for larger model set:
        # torch._dynamo.config.accumulated_cache_size_limit = 128
        model = torch.compile(model)

    ### Step-5: Setup the optimizer and lr_scheduler ####   
    # Split parameters into two groups based on their names, 
    # setting one-D params without weight decay, following https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L223
    name_to_trainable_params = {}
    # new_weights means extra weights the ttt-layer added.
    new_weights_params_with_weight_decay = []
    new_weights_params_no_weight_decay = []
    original_params_with_weight_decay = []
    original_params_no_weight_decay = []
    
    wd_params_name_list = [_name for _name, _shape in name_to_orig_shape.items() if len(_shape) > 1]
    def rename_param(name: str) -> str:
        return name.replace("_fsdp_wrapped_module.", "") \
                   .replace("_checkpoint_wrapped_module.", "") \
                   .replace("_orig_mod.", "") \
                   .replace("module.", "")

    num_trainable_params = 0 
    num_trainable_new_weights_params = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        renamed_name = rename_param(name)
        name_to_trainable_params[renamed_name] = param
        if any(key in renamed_name for key in ['self_attn.w0', 'self_attn.w1', 'self_attn.w2', 
                                        'self_attn.qk_scale', 'self_attn.qk_offset',
                                        'self_attn.lr_proj.weight', 'self_attn.lr_proj.bias',
                                        'self_attn.output_norm.weight', "ttt_scale_proj.weight", "ttt_scale_proj.bias"]):
            # new params
            # check if shape is 1D! 
            if renamed_name in wd_params_name_list:
                new_weights_params_with_weight_decay.append(param)
            else:
                new_weights_params_no_weight_decay.append(param)
            logging.info(
                f"Newly created param: Orig param name: {name}, new param name: {renamed_name}. | "
                f"Param shape: {param.shape}, numel: {param.numel()}, dtype: {param.dtype} | "
                f"weight decay? : {renamed_name in wd_params_name_list}"
            )
            num_trainable_new_weights_params += name_to_orig_numel[renamed_name]
        else:
            if renamed_name in wd_params_name_list:
                original_params_with_weight_decay.append(param)
            else:
                original_params_no_weight_decay.append(param)
            logging.info(
                f"Orig param name: {name}, new param name: {renamed_name}. | "
                f"Param shape: {param.shape}, numel: {param.numel()}, dtype: {param.dtype} | "
                f"weight decay? : {renamed_name in wd_params_name_list}"
            )
            num_trainable_params += name_to_orig_numel[renamed_name]
    
    logging.info(f"=> Number of trainable parameters: {num_trainable_params / 1e6}M")
    logging.info(f"=> Number of trainable new weights parameters: {num_trainable_new_weights_params / 1e6}M")
    weight_decay = config.get("weight_decay", 0.0)
    lr_multiplier = config.train.get("lr_multiplier", 1.0)
    # "optional code" added for first stage.
    if config.train.get("first_stage", False):
        weight_decay = 0.05
    optimizer_param_groups = [
        {
            "params": original_params_with_weight_decay,
            "weight_decay": weight_decay,
        },
        {
            "params": original_params_no_weight_decay,
            "weight_decay": 0.0,
        },
        # setup a lr_multipler for the new weights. default is 10.0 => larger lr for new weights.
        {
            "params": new_weights_params_with_weight_decay,
            "lr": lr_multiplier * config.lr,
            "weight_decay": weight_decay,
        },
        {
            "params": new_weights_params_no_weight_decay,
            "lr": lr_multiplier * config.lr,
            "weight_decay": 0.0,
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_param_groups, betas=(config.beta1, config.beta2), lr=config.lr)

    ### Step-6: Setup the ema ####
    use_ema = config.train.get("use_ema", False)
    ema_params = None
    if use_ema: 
        ema_weight = config.train.get("ema_weight", 0.999) # might use 0.999
        name_to_trainable_params = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            renamed_name = rename_param(name)
            name_to_trainable_params[renamed_name] = param
        ema_params = EMAParams(name_to_trainable_params, ema_weight)
        num_ema_params = sum(p.numel() for p in ema_params.name_to_ema_params.values())
        logging.info(f"=> Setting up EMA for trainable params. "
                f"Number of EMA parameters: {num_ema_params / 1e6}M. "
                f"EMA weight: {ema_weight}")
    
    dist.barrier()

    return model, optimizer, ema_params

    

class Trainer:
    def __init__(self, args):

        ########## Step 1: Initialize the distributed training environment ##########
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        global_rank, world_size, local_rank = launch_distributed_job(
            backend="nccl",
        )
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.world_size = world_size

        ########## Step-2: Load the Config. Override the YAML values, Initialize the logging and Wandb. Initialize the seed ##########
        config = OmegaConf.load(args.config)

        # Override the YAML values
        if args.set is not None:
            for key_value in args.set:
                key, value = key_value
                # predefined shortcut keys for editting model config. 
                short2long = {
                    "eff_attn": "model.diffusion_config.model_config.efficient_attn_config.0",
                    "eff_attn1": "model.diffusion_config.model_config.efficient_attn_config.1",
                    "block_config": "model.diffusion_config.model_config",
                }
                for short, long in short2long.items():
                    key = key.replace(short, long)
                key_parts = key.split(".") # split the key into list of strings
                set_nested_key(config, key_parts, value)
                if dist.get_rank() == 0:
                    print(f"Overriding {key} with {value}")

        config = OmegaConf.create(config)
        config = edict(config)
        print_rank0(config)

        # Set the sequence parallel config.
        if config.train.get("sp_size", 1) > 1:
            from minVid.models.wan.wan_base.distributed import sp_support
            sp_support.init_sp_group(sp_size=config.train.sp_size)

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32

        start_seed = config.get("seed", 0)
        set_seed(start_seed + global_rank, deterministic=config.get("deterministic", False))

        config.wandb_name = config.exp_name
        config.output_path = os.path.join(config.output_path, config.wandb_name)
        output_path = os.path.join(config.output_path, f"seed_{config.seed}")
        self.output_path = output_path
        if self.is_main_process:
            # set the wandb and output path
            assert os.path.exists(
                config.api_key_path
            ), f"API key file does not exist: {config.api_key_path}"
            api_keys = edict(yaml.safe_load(open(config.api_key_path, "r")))
            assert api_keys.wandb is not None, "Wandb API key not found in api key file"
            config_copy = copy.deepcopy(config)
            config_copy.wandb_key = api_keys.wandb
            self.wandb_folder = init_logging_folder(config_copy, self.output_path)
            # tell the output_path to all processes

        ########## Step-3: Initialize the model, optimizer, lr_scheduler ##########
        self.model = instantiate_from_config(config.model)
        
        print("Creating trainable params list with attn_only: ", config.train.attn_only, config.train.get("first_stage", False))
        trainable_params_list = self.model.get_trainable_params(attn_only=config.train.attn_only,
                                                                first_stage=config.train.get("first_stage", False))
        for param in self.model.parameters():
            param.requires_grad = False
        for param in trainable_params_list:
            param.requires_grad = True

        
        self.step = 0

        ########## Step-4: DDP and FSDP wrap the model ##########
        # DDP warp for the self.model.generator
        self.model.generator, self.optimizer, self.ema_params = shard_model_set_ema_set_optimizer(
            self.model.generator, config, global_rank
        )
        
        # FSDP warp for the text encoder.  no requires grad params.
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            transformer_module=(
                (T5Block,)
                if config.text_encoder_fsdp_wrap_strategy == "transformer"
                else None
            ),
        )
        self.use_ema = config.train.get("use_ema", False)

        # Print device and dtype information for model components
        if dist.get_rank() == 0:  # Only print from rank 0 to avoid flooding logs
            logging.info("Model component device and dtype information:")
            
            # Check text_encoder
            if hasattr(self.model, "text_encoder"):
                # For FSDP models, we need to check parameters
                text_encoder_device = next(self.model.text_encoder.parameters()).device
                text_encoder_dtype = next(self.model.text_encoder.parameters()).dtype
                logging.info(f"  text_encoder: device={text_encoder_device}, dtype={text_encoder_dtype}")
            
            # Check generator
            if hasattr(self.model, "generator"):
                generator_device = next(self.model.generator.parameters()).device
                generator_dtype = next(self.model.generator.parameters()).dtype
                logging.info(f"  generator: device={generator_device}, dtype={generator_dtype}")
            
            # Check vae
            if hasattr(self.model, "vae"):
                vae_device = next(self.model.vae.parameters()).device
                vae_dtype = next(self.model.vae.parameters()).dtype
                logging.info(f"  vae: device={vae_device}, dtype={vae_dtype}")
        
        self.model.to(self.device)

        ###### Setup lr scheduler ######
        scheduler_type = config.get("lr_scheduler_type", "cosine")
        self.lr_scheduler = configure_lr_scheduler(
            self.optimizer,
            config.max_fwdbwd_passes,
            config.warmup,
            scheduler_type=scheduler_type,
        )

        ########## Step-5: Check for Pretrained DCP: config.train.pretrained_dcp_path ##########
        pretrained_dcp_path = config.train.get("pretrained_dcp_path", None)
        if pretrained_dcp_path is not None and os.path.exists(pretrained_dcp_path):
            print_rank0(f"Loading pretrained DCP from {pretrained_dcp_path}")
            resume_job_dcp(pretrained_dcp_path, self.model.generator, self.device, None, None, self.ema_params)
            self.step = 0
        else:
            print_rank0(f"No pretrained DCP found at {pretrained_dcp_path}")

        ########## Step-6: Check for existing checkpoints and resume if found ##########
        latest_checkpoint, latest_step = find_latest_checkpoint(self.output_path)
        if latest_checkpoint is not None and latest_step > 0:
            print_rank0(f"Found latest checkpoint: {latest_checkpoint}")
            resume_job_dcp(latest_checkpoint, self.model.generator, self.device, self.optimizer, self.lr_scheduler, self.ema_params)
            self.step = latest_step
            print_rank0(f"Resumed training from step {self.step}")
            # set lr_scheduler to the last step
            self.lr_scheduler.last_epoch = latest_step - 1

        self.config = config

        ########## Step-7: Dataset and Dataloader ##########
        if args.debug:
            print_rank0("Debug mode, skipping dataset and dataloader")
            return
    
        # reset seed if it's resuming from a checkpoint
        start_seed = config.get("seed", 0)
        if self.step != 0 or config.train.get("continue_training", False):
            start_seed = start_seed + 1 + self.step
            set_seed(start_seed + global_rank, deterministic=config.get("deterministic", False))
            # use a random data_seed for each rank
            data_seed = int(datetime.datetime.now().timestamp()) + dist.get_rank() * 12345
        else:
            data_seed = start_seed + global_rank

        print(f"Data seed for rank {dist.get_rank()} is {data_seed}. global rank is {global_rank}")

        data_config = config.dataset_train
        # you need to implement your own get_data_module function.
        data_module = get_data_module(
                    data_config, data_seed=data_seed)
        dataloader = data_module.train_dataloader()
        self.dataloader = cycle(dataloader)

        # auto garbage collection does not happen together between multiple nodes
        # should be disabled for training and being replaced with manual gc and empty cache
        # NOTE: when eval, it should be enabled, since there might be lots of things going on.
        self.disable_auto_gc = config.train.get("disable_auto_gc", True)
        self.manual_gc_interval = config.train.get("manual_gc_interval", 250)


    def train_one_step(self, data_batch: dict):
        # data_batch, tensors already on the correct device
        tic = time.time()
        timing_str_list = []

        if self.config.get("profile", False):
            torch.cuda.synchronize()
            timing_str_list.append(f"trainer loop, rank: {self.global_rank}")

        # Determine if this step requires gradient synchronization
        is_sync_step = (self.step + 1) % self.config.grad_accum_steps == 0
        # for full shard model, we need to sync the gradient every step even with grad accum steps
        ctx = nullcontext() #  if is_sync_step else self.model.generator.no_sync()
        
        # Please use `torch.amp.autocast('cuda', args...)` instead.
        with ctx, torch.amp.autocast(
            "cuda",
            enabled=self.config.mixed_precision,
            dtype=self.dtype,
        ):
            # no need of grad scalar for bf16 and fp32, unless using fp16
            result_dict = self.model(data_batch)

        if self.config.get("profile", False):
            torch.cuda.synchronize()
            timing_str_list.append(f"till model forward: {time.time() - tic:06f}")

        loss = result_dict["loss"]
        extra_loss = result_dict.get("extra_loss", 0.0)
        if extra_loss > 0.0:
            loss_bwd = loss + extra_loss
        else:
            loss_bwd = loss

        # don't need to divide by the sp_size. since fsdp will average grad over all processes.
        (loss_bwd / self.config.grad_accum_steps).backward()

        if self.config.get("profile", False):
            torch.cuda.synchronize()
            timing_str_list.append(f"till model backward: {time.time() - tic:06f}")

        self.step += 1
        skip_optimizer_step = False
        if torch.isnan(loss) or torch.isinf(loss):
            skip_optimizer_step = True

        total_grad_norm = 0
        # Only perform optimizer step and gradient clipping at sync steps
        if is_sync_step:
            if not skip_optimizer_step:
                # total_grad_norm = torch.nn.utils.clip_grad_norm_(
                #     self.trainable_params_list, max_norm=self.config.grad_clip_norm
                # ).item()
                total_grad_norm = self.model.generator.clip_grad_norm_(self.config.grad_clip_norm).item()
                if not math.isfinite(total_grad_norm):
                    print(
                        f"WARNING: step {self.step} grad norm is not finite {total_grad_norm}, skipping"
                    )
                    skip_optimizer_step = True

            allowed_gradnorm = self.config.grad_clip_norm * self.config.get(
                    "allowed_gradnorm_factor", 10.0
                )
            if self.step > self.config.get("may_skip_gradnorm_after", 200):
                if total_grad_norm > allowed_gradnorm:
                    print(
                        f"WARNING: step {self.step} grad norm is too high {total_grad_norm}, skipping"
                    )
                    if self.is_main_process:
                        wandb.log({"grad_norm": total_grad_norm}, step=self.step)
                    skip_optimizer_step = True

            if not skip_optimizer_step:
                self.optimizer.step()
                if self.use_ema:
                    self.ema_params.update()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

        if self.config.get("profile", False):
            torch.cuda.synchronize()
            timing_str_list.append(f"till optimizer step: {time.time() - tic:06f}")
            print(",  ".join(timing_str_list))


        ######### gather the loss and time breakdown from all process
        unnormalized_loss = loss.detach()
        timestep = result_dict["t"].mean().detach()
        if self.world_size > 1:
            gathered_unnormalized_loss = torch.zeros(
                [self.world_size, *unnormalized_loss.shape],
                dtype=unnormalized_loss.dtype, device=self.device)
            gathered_timestep = torch.zeros(
                [self.world_size, *timestep.shape],
                dtype=timestep.dtype, device=self.device)

            dist.all_gather_into_tensor(
                gathered_unnormalized_loss, unnormalized_loss)
            dist.all_gather_into_tensor(gathered_timestep, timestep)
        else:
            gathered_unnormalized_loss = unnormalized_loss
            gathered_timestep = timestep

        loss_breakdown = defaultdict(list)
        stats = {}
        gathered_unnormalized_loss_mean = gathered_unnormalized_loss.mean().item()

        for index, t in enumerate(gathered_timestep):
            loss_breakdown[str(int(t.item() // 250) * 250)].append(
                gathered_unnormalized_loss[index].item())

        for key_t in loss_breakdown.keys():
            stats["loss_at_time_" + key_t] = sum(loss_breakdown[key_t]) / \
                len(loss_breakdown[key_t])

        if self.is_main_process and (self.step % self.config.wandb_log_every == 0 or self.step <= 100):
            wandb_loss_dict = {
                "train/loss": gathered_unnormalized_loss_mean,
                "train/grad_norm": total_grad_norm,
                "train/lr": self.optimizer.param_groups[0]['lr'],
                "train/time": time.time() - tic,
                "train/avg_timestep": gathered_timestep.mean().item(),
            }
            if extra_loss > 0.0:
                wandb_loss_dict["train/extra_loss"] = extra_loss
            wandb_loss_dict.update(result_dict["vis_dict"])
            wandb_loss_dict.update(stats)
            wandb.log(wandb_loss_dict, step=self.step)
            print_rank0(f"step {self.step} loss: {loss.item():03f} time: {time.time() - tic:02f}s total grad norm: {total_grad_norm:02f} lr: {self.optimizer.param_groups[0]['lr']:06f} extra loss: {extra_loss:03f}")

            # Log videos at a reduced frequency
            if self.step % (self.config.wandb_log_every * 50) == 0:
                # Get videos from data batch - shape [B, f, c, h, w]
                videos = data_batch["video_rgb"]

                # Convert from range [0, 1] to [0, 255] for wandb
                videos_uint8 = (videos.detach().cpu() * 255).type(torch.uint8)

                # Log a few sample videos (limit to first 2 in batch to save bandwidth)
                video_samples = min(2, videos_uint8.shape[0])
                for i in range(video_samples):
                    video_i = videos_uint8[i]  # [f, c, h, w]
                    caption_i = data_batch["text_prompts"][i] if "text_prompts" in data_batch else f"Sample {i}"
                    wandb.log({
                        f"video_data_{i}": wandb.Video(
                            video_i,
                            fps=16,  # Using the target_fps from the downsample_video_interpolate function
                            caption=caption_i
                        )
                    }, step=self.step)

    def train(self):

        # Disable the auto gc and empty the GPU cache before training.
        if self.disable_auto_gc:
            if dist.get_rank() == 0:
                print_rank0("DistGarbageCollector: Disabling automatic GC.")
            gc.disable()
        torch.cuda.empty_cache()
        self.model.train()
        start_time = time.time()
        while self.step < self.config.max_fwdbwd_passes:
            data_batch = next(self.dataloader)
            text_prompts = data_batch["caption"]
            if self.config.get("profile", False):
                data_time = time.time() - start_time
                data_time = dist.reduce(data_time, "max")
                print_rank0(f"Data loading time: {data_time:06f}s")
                start_time = time.time()

            video_rgbs = data_batch["frames"] 
            video_rgbs = video_rgbs.permute(0, 2, 1, 3, 4) # [B, f, c, h, w], float32, at cpu in range [0, 1]
            data_batch_gpu = {
                "video_rgb": video_rgbs.to(self.device),
                "text_prompts": text_prompts, # list of strings
            }

            self.train_one_step(data_batch_gpu)

            if self.step % self.config.save_every == self.config.save_every - 1:
                # does not save optimizer and lr_scheduler for now
                self.optimizer.zero_grad()
                save_job(
                    self.output_path, 
                    self.step, 
                    self.model.generator, 
                    self.global_rank, 
                    self.local_rank,
                    ema_params=self.ema_params, 
                )
                if self.is_main_process:
                    clear_old_checkpoints(self.output_path, self.config.get("keep_last_iter", 10000))
                dist.barrier()

            if self.step % self.manual_gc_interval == 0:
                gc.collect()
                print_rank0("Manual GC collected for all ranks")

    def debug_train(self):
        # create dummy data batch
        print("debug train")
        bs = self.config.batch_size_per_gpu
        video_rgbs = torch.randn(bs, 81, 3, 480, 832).to(self.device)
        text_prompts = ["a video of a cat"]
        data_batch = {
            "video_rgb": video_rgbs,
            "text_prompts": text_prompts,
        }
        for i in range(100):
            self.train_one_step(data_batch)
            if self.step % self.config.save_every == self.config.save_every - 1:
                # does not save optimizer and lr_scheduler for now
                print(f"saving checkpoint at step {self.step}")
                save_job(self.output_path, self.step, self.model.generator, self.global_rank, ema_params=self.ema_params, upload_to_s3=self.upload_to_s3, s3_folder=self.s3_folder)
                if self.is_main_process:
                    clear_old_checkpoints(self.output_path, self.config.get("keep_last_iter", 10000))
                dist.barrier()

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="RelitLRM training script. override the config")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "--set",
        "-s",
        type=str,
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="New value for the key",  # -s exp_name xxxx
    )
    # debug mode would not create dataset and dataloader, just use dummy data batch for fwd and bwd pass
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run debug mode",
    )
    args = parser.parse_args()


    torch.backends.cuda.matmul.allow_tf32 = True

    trainer = Trainer(args)
    if args.debug:
        trainer.debug_train()
    else:
        trainer.train()
