# from https://github.com/tianweiy/CausVid/blob/master/causvid/util.py
import os
import random
from datetime import datetime, timedelta
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (MixedPrecision, ShardingStrategy,
                                    StateDictType)
from torch.distributed.fsdp.wrap import (ModuleWrapPolicy,
                                         size_based_auto_wrap_policy,
                                         transformer_auto_wrap_policy)


def launch_distributed_job(backend: str = "nccl"):
    """
    When launched with torchrun --nproc_per_node=<> --nnodes=<>  --rdzv-endpoint=${master_addr}:${master_port}
    The following environment variables are set:
    LOCAL_RANK: 0 ~ nproc_per_node - 1
    RANK: 0 ~ world_size - 1
    WORLD_SIZE: nproc_per_node * nnodes
    MASTER_ADDR: master_addr
    MASTER_PORT: master_port
    GROUP_RANK: 0 ~ nnodes - 1
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    ddp_local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    ddp_node_rank = int(os.environ["GROUP_RANK"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        init_method=init_method,
        timeout=timedelta(minutes=30),
    )
    torch.cuda.set_device(local_rank)

    print(
        f"Process {rank}/{world_size} is using device or local rank {local_rank}/{ddp_local_world_size} on node {ddp_node_rank}"
    )

    dist.barrier()

    return rank, world_size, local_rank


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def init_logging_folder(args, output_path):
    """
    Initialize the logging folder and wandb run
    Given:
        args:
            output_path: path to save the output
            wandb_host: wandb host
            wandb_key: wandb key
            wandb_entity: wandb entity
            wandb_project: wandb project
            wandb_name: wandb name
    Returns:
        output_path: path to save the output
        wandb_folder: path to save the wandb run
    """
    os.makedirs(output_path, exist_ok=True)

    wandb_id = None
    if os.path.exists(os.path.join(output_path, "wandb_id.txt")):
        with open(os.path.join(output_path, "wandb_id.txt"), "r") as f:
            wandb_id = f.read().strip()
        print(f"Resuming wandb run with id {wandb_id}")
    
    wandb.login(host=args.wandb_host, key=args.wandb_key)
    wandb_save_dir = args.get("wandb_save_dir", "/mnt/localssd/wandb/")
    run = wandb.init(
        config=args,
        resume="allow",
        **{
            "mode": "online",
            "entity": args.wandb_entity,
            "project": args.wandb_project,
        },
        id=wandb_id,
        dir=wandb_save_dir
    )
    wandb.run.log_code(".")
    wandb.run.name = args.wandb_name
    print(f"run dir: {run.dir}")
    wandb_folder = run.dir
    os.makedirs(wandb_folder, exist_ok=True)

    # save wandb id if a new run for later resuming wandb
    if wandb_id is None:
        wandb_id = wandb.run.id
        with open(os.path.join(output_path, "wandb_id.txt"), "w") as f:
            f.write(wandb_id)
    
    print("Wandb setup done")
    
    import omegaconf
    # Convert EasyDict to a regular dictionary
    args_dict = dict(args)
    # Convert to OmegaConf configuration
    conf = omegaconf.OmegaConf.create(args_dict)
    with open(os.path.join(output_path, "config.yaml"), "w") as f:
        OmegaConf.save(conf, f)

    return output_path, wandb_folder


def fsdp_wrap(
    module,
    sharding_strategy="full",
    mixed_precision=False,
    wrap_strategy="size",
    min_num_params=int(5e7),
    transformer_module=None,
    use_orig_params=False, # False => flat params. Slightly faster, efficient and bug free.
):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy, min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        # "hybrid_full": only shard within node. 
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=False,  # Load ckpt on rank 0 and sync to other ranks
        use_orig_params=use_orig_params,
    )
    return module


def cycle(dl):
    while True:
        for data in dl:
            yield data


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def barrier():
    if dist.is_initialized():
        dist.barrier()


def prepare_images_for_saving(
    images_tensor, height, width, grid_size=1, range_type="neg1pos1"
):
    if range_type != "uint8":
        images_tensor = (images_tensor * 0.5 + 0.5).clamp(0, 1) * 255

    images = (
        images_tensor[: grid_size * grid_size]
        .permute(0, 2, 3, 1)
        .detach()
        .cpu()
        .numpy()
        .astype("uint8")
    )
    grid = images.reshape(grid_size, grid_size, height, width, 3)
    grid = np.swapaxes(grid, 1, 2).reshape(grid_size * height, grid_size * width, 3)
    return grid


def print_rank0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)
