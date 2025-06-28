"""
This implements rank-0 checkpointing.  (which is not efficient for FSDP with huge models.)
"""
import os

import torch
from easydict import EasyDict as edict

import traceback
from minVid.utils.dist_utils import print_rank0
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from torch.distributed.checkpoint.state_dict import get_state_dict, get_model_state_dict
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
import shutil



def save_dcp(model, ckpt_path, shard_strategy="full"):
    """
    Save the model using torch.distributed.checkpoint
    ckp_path: str, dir path to save the checkpoint
    """
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = model.state_dict()
        if shard_strategy == "full":
            dist_cp.save(
                state_dict=state_dict,
                storage_writer=dist_cp.FileSystemWriter(ckpt_path),
            )
        elif shard_strategy == "hybrid": # hybrid
            raise NotImplementedError("Hybrid strategy not implemented")


def load_dcp(model, ckpt_path, shard_strategy="full", strict=False):
    """
    Load the model using torch.distributed.checkpoint
    ckpt_path: str, dir path where the checkpoint is saved
    shard_strategy: str, either "full" or "hybrid" (matching save strategy)
    strict: bool, whether to strictly enforce that the keys in state_dict match the keys in model
    
    Returns: True if loading succeeds
    """
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        if shard_strategy == "full":
            state_dict = model.state_dict()
            planner = DefaultLoadPlanner(allow_partial_load=True)
            dist_cp.load(
                state_dict=state_dict,
                storage_reader=dist_cp.FileSystemReader(ckpt_path),
                planner=planner,
            )
            # Load the state dict into the model
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            if dist.get_rank() == 0:
                print(f"Loaded {len(state_dict) - len(missing)} tensors from dcp checkpoint "
                    f"(skipped {len(missing)} new and {len(unexpected)} obsolete).")
        elif shard_strategy == "hybrid":
            raise NotImplementedError("Hybrid strategy not implemented")
        else:
            raise ValueError(f"Unknown shard strategy: {shard_strategy}")
    
    # Synchronize all processes to ensure loading is complete
    dist.barrier()
    
    return True

def save_job(output_path, step, model, global_rank, local_rank, optimizer=None, lr_scheduler=None, ema_params=None):
    """
    save_path: str, dir path to save the checkpoint
    step: int, step number of the checkpoint
    model: torch.nn.Module, model to save, might be DDP wrapped
    optimizer: torch.optim.Optimizer, optimizer to save
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler, lr_scheduler to save

    the checkpoint path looks like:
    exp_name/seed_200/checkpoint_model_000009/dcp/
    
    """
    do_ckpt_optimizer = optimizer is not None
    do_ckpt_scheduler = lr_scheduler is not None
    if global_rank == 0:
        print(f"Start checkpointing... Save model: True, Save optimizer: {do_ckpt_optimizer}, Save lr_scheduler: {do_ckpt_scheduler}")
    

    # Create directory with step number in the name
    checkpoint_dir = os.path.join(output_path, f"checkpoint_model_{step:06d}")
    # if global_rank == 0:
    #     os.makedirs(checkpoint_dir, exist_ok=True)

    if ema_params is not None:
        ema_params.cache_model()
        ema_params.copy_to_model()
    
    # make sure any asynchronous CUDA ops are complete before we start touching tensors
    torch.cuda.synchronize()
    dist.barrier()      
    
    if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        if global_rank == 0:
            print(f"WARNING: DistributedDataParallel model found, unwrapping...")
        model = model.module
    else:
        save_dcp(model, os.path.join(checkpoint_dir, "dcp"), shard_strategy="full")
    
    if ema_params is not None:
        ema_params.restore_model_from_cache()
    
    dist.barrier()
    if local_rank == 0:
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    dist.barrier()



def find_latest_checkpoint(output_path):
    """
    Find the latest checkpoint in the output directory.
    
    Returns:
        str or None: Path to the latest checkpoint directory, or None if no checkpoint found.
        int: The step number of the latest checkpoint, or 0 if no checkpoint found.
    """
    if not os.path.exists(output_path):
        return None, 0
    
    # Find all checkpoint directories matching the pattern
    checkpoint_dirs = [d for d in os.listdir(output_path) 
                        if os.path.isdir(os.path.join(output_path, d)) and 
                        d.startswith("checkpoint_model_")]
    
    if not checkpoint_dirs:
        return None, 0
    
    # Extract step numbers from directory names
    checkpoint_steps = []
    for d in checkpoint_dirs:
        try:
            step = int(d.replace("checkpoint_model_", ""))
            checkpoint_dir = os.path.join(output_path, d)
            # Verify that all required files exist in the directory
            if (os.path.exists(os.path.join(checkpoint_dir, "dcp"))):
                checkpoint_steps.append((step, checkpoint_dir))
        except ValueError:
            continue
    
    if not checkpoint_steps:
        return None, 0
    
    # Find the checkpoint with the highest step number
    latest_step, latest_checkpoint = max(checkpoint_steps, key=lambda x: x[0])
    return latest_checkpoint, latest_step



@torch.no_grad()
def resume_job_dcp(checkpoint_path, model, device, optimizer=None, lr_scheduler=None, ema_params=None):
    """
    Resume training from a checkpoint. 
    the checkpoint path looks like:
    exp_name/seed_200/checkpoint_model_000009/  
    the dcp_folder is in the checkpoint_model_000009/dcp/
    
    Args:
        checkpoint_path (str): Path to the checkpoint directory
    """
    # print(f"Resuming training from {checkpoint_path}")
    
    # Extract step from directory name
    dir_name = os.path.basename(checkpoint_path)
    if "checkpoint_model_" in dir_name:
        step = int(dir_name.replace("checkpoint_model_", ""))
        # 1. Load generator model using standard PyTorch loading
        generator_path = os.path.join(checkpoint_path, "dcp")
    else:
        step = 0
        generator_path = checkpoint_path
    

    dist.barrier()
    
    load_dcp(model, generator_path, shard_strategy="full")
    

    print_rank0(
        f"Loaded model from {os.path.abspath(generator_path)}"
    )
    if ema_params is not None:
        print_rank0("Copying EMA parameters from model to ema_params")
        ema_params.copy_from_model()
    
    # 2. Load optimizer state (standard PyTorch format)
    optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
    if optimizer is not None and os.path.exists(optimizer_path):
        optimizer_state = torch.load(optimizer_path, map_location="cpu")
        optimizer.load_state_dict(optimizer_state)
    
        # Move optimizer states to correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    # 3. Load lr_scheduler state (standard PyTorch format)
    scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
    if lr_scheduler is not None and os.path.exists(scheduler_path):
        scheduler_state = torch.load(scheduler_path, map_location="cpu")
        lr_scheduler.load_state_dict(scheduler_state)
    
    # print(f"Successfully resumed training from step {step}")
    
    # Synchronize all processes after loading
    dist.barrier()

    return step


def clear_old_checkpoints(output_path, keep_last_iter=10000):
    """
    Remove older checkpoint directories, keeping only the most recent ones
    within the specified iteration range.
    
    Args:
        keep_last_iter (int): Keep checkpoints from the last N iterations
                                (default: 10000)
    """
    # Only run from the main process to avoid race conditions
    # Safety check - don't run if path is suspicious
    if not output_path or len(output_path.strip()) < 10:
        print("WARNING: Output path seems too short, refusing to delete checkpoints")
        return
    
    print(f"Searching for old checkpoints to remove (keeping last {keep_last_iter} iterations)...")
    
    # Find all checkpoint directories
    if not os.path.exists(output_path):
        return
    
    checkpoint_dirs = []
    for d in os.listdir(output_path):
        if os.path.isdir(os.path.join(output_path, d)) and d.startswith("checkpoint_model_"):
            try:
                step = int(d.replace("checkpoint_model_", ""))
                checkpoint_dirs.append((step, os.path.join(output_path, d)))
            except ValueError:
                continue
    
    # Sort by step number
    checkpoint_dirs.sort(key=lambda x: x[0])
    
    # Define the cutoff step
    cutoff_step = step - keep_last_iter
    
    # Find directories to remove
    dirs_to_remove = []
    for step, dir_path in checkpoint_dirs:
        if step < cutoff_step:
            dirs_to_remove.append((step, dir_path))
    
    # Remove the directories
    for step, dir_path in dirs_to_remove:
        try:
            print(f"Removing old checkpoint: {dir_path} (step {step})")
            import shutil
            shutil.rmtree(dir_path)
        except Exception as e:
            print(f"Error removing directory {dir_path}: {e}")
    
    print(f"Removed {len(dirs_to_remove)} old checkpoint directories.")

