from minVid.models.wan.wan_inference_pipeline import WanInferencePipeline
from minVid.models.wan.ar_wan_inference_pipeline import WanInferencePipeline as ARWanInferencePipeline
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import Dataset
import os
from minVid.utils.io_utils import export_to_video
from minVid.utils.config_utils import instantiate_from_config, set_nested_key
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    ModuleWrapPolicy,
)
from torch.distributed.fsdp import StateDictType

from minVid.utils.config_utils import get_obj_from_str
from functools import partial
import torch.distributed.checkpoint as dcp

from torch.distributed.checkpoint import FileSystemReader
import random
import sys
from minVid.utils.dist_utils import launch_distributed_job

class TextDataset(Dataset):
    def __init__(self, data_path, local_rank, world_size):
        self.texts = []
        with open(data_path, "r") as f:
            for line in f:
                self.texts.append(line.strip())
        self.local_rank = local_rank
        self.world_size = world_size
        self.texts = self.texts[self.local_rank::self.world_size]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def setup_model_and_load_weights(config, args, checkpoint_folder, device):

    ########## Step 1: Initialize the Inference Pipeline ##########
    if args.ar:
        print("Using ARWanInferencePipeline")
        pipe = ARWanInferencePipeline(config.model, device=device)
    else:
        pipe = WanInferencePipeline(config.model, device=device)

    # pipe.generator = pipe.generator.to(device)

    ########## Step 2: Wrap the model's generator with FSDP ##########
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
    wrapped_generator = FSDP(pipe.generator,
                 sharding_strategy=fsdp_strategy,
                 auto_wrap_policy=fsdp_wrap_policy,
                 mixed_precision=mixed_precision,
                 device_id=torch.cuda.current_device(),
                 use_orig_params=True,
                 limit_all_gathers=True,
                 sync_module_states=False, # Load ckpt on rank 0 and sync to other ranks
                 )

    ########## Step 3: Load the distributed checkpoint ##########
    dcp_folder = os.path.join(checkpoint_folder, "dcp")
    assert os.path.exists(dcp_folder), f"Distributed checkpoint directory not found at {dcp_folder}"
    print(f"Loading distributed checkpoint from {dcp_folder}")

    # Create a reader for the checkpoint
    reader = FileSystemReader(dcp_folder)
    metadata = reader.read_metadata()

    ckpt_keys = list(metadata.state_dict_metadata.keys())
    # ckpt_keys could be used for future debugging
    print("Parsing checkpoint metadata keys: Few examples are:")
    for key in ckpt_keys[:5]:
        print(key)

    # Load state dict in FSDP format

    def add_model_prefix(name: str) -> str:
        return f"model.{name}"
    def remove_model_prefix(name: str) -> str:
        return name.replace("model.", "")
    def rename_param(name: str) -> str:
        return name.replace("_fsdp_wrapped_module.", "") \
                   .replace("_checkpoint_wrapped_module.", "") \
                   .replace("_orig_mod.", "") \
                   .replace("module.", "")

    state_dict_to_load = {}
    renmaed_state_dict_to_load = {}
    with FSDP.state_dict_type(wrapped_generator, StateDictType.SHARDED_STATE_DICT):
        state_dict_to_load = wrapped_generator.state_dict()

    print("Example of state_dict_to_load keys:")
    for key in list(state_dict_to_load.keys())[:5]:
        print(key)

    for key, value in state_dict_to_load.items():
        new_key = rename_param(key)
        new_key = add_model_prefix(new_key)
        renmaed_state_dict_to_load[new_key] = value

    # Load the state dict
    dcp.load(
        state_dict=renmaed_state_dict_to_load,
        storage_reader=reader,
    )

    print("Example of loaded state_dict keys:")
    for key in list(renmaed_state_dict_to_load.keys())[:5]:
        print(key)

    for key, value in renmaed_state_dict_to_load.items():
        state_dict_to_load[remove_model_prefix(key)] = value

    with FSDP.state_dict_type(wrapped_generator, StateDictType.SHARDED_STATE_DICT):
        wrapped_generator.load_state_dict(state_dict_to_load)
    print(f"Loaded the checkpoint from distributed checkpoint at {dcp_folder}")
    pipe.generator = wrapped_generator
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--checkpoint_folder", type=str, default=None)
    parser.add_argument("--output_folder", type=str, default="../output_videos_moviegen/")
    parser.add_argument("--prompt_file_path", type=str, default="sample_dataset/MovieGenVideoBench.txt")
    parser.add_argument("--num_videos", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ar", action="store_true", default=False)
    parser.add_argument("--ten_secs", action="store_true", default=False)
    parser.add_argument("--eight_secs", action="store_true", default=False)
    parser.add_argument("--twenty_secs", action="store_true", default=False)
    parser.add_argument(
            "--set",
            "-s",
            type=str,
            action="append",
            nargs=2,
            metavar=("KEY", "VALUE"),
            help="New value for the key",  # -s exp_name xxxx
        )

    args = parser.parse_args()

    if args.ar:
        print("called with --ar, using ARWanInferencePipeline")

    config = OmegaConf.load(args.config)

    # Override the YAML values
    if args.set is not None:
        for key_value in args.set:
            key, value = key_value
            short2long = {
                "eff_attn": "model.diffusion_config.model_config.efficient_attn_config.0",
                "eff_attn1": "model.diffusion_config.model_config.efficient_attn_config.1",
                "block_config": "model.diffusion_config.model_config",
            }
            for short, long in short2long.items():
                key = key.replace(short, long)
            key_parts = key.split(".") # split the key into list of strings
            set_nested_key(config, key_parts, value)

            print(f"Overriding {key} with {value}")

    config = OmegaConf.create(config)
    ##### Parse the checkpoint_folder to comeup with a new name of the output subfolder.
    # checkpoint_folder is like  ../experiments/bs64_lr5e-6_ts3_uniform/seed_42/checkpoint_model_001999/
    # the new name of the output subfolder is bs64_lr5e-6_ts3_uniform_model_001999
    if args.checkpoint_folder is not None:
        exp_name = args.checkpoint_folder.split("/")[2]
        model_iter_name = args.checkpoint_folder.strip("/").split("/")[-1]
        output_folder = os.path.join(args.output_folder, f"{exp_name}/{model_iter_name}")
        args.output_folder = output_folder

    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # Set GPU to use before doing anything else
    global_rank, world_size, local_rank = launch_distributed_job(
        backend="nccl",
    )
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.set_grad_enabled(False)

    pipe = setup_model_and_load_weights(config, args, args.checkpoint_folder, device)

    pipe = pipe.to(device)
    pipe.eval()

    ########## Step 4: Inference ##########

    seed = args.seed + local_rank
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    dataset = TextDataset(args.prompt_file_path, local_rank, world_size)

    generate_shape = (16, 21, 60, 104)
    # generate_shape = (16, 21, 90, 156)
    # generate_shape = (16, 41, 60, 104)

    if args.ten_secs:
        # 10 seconds video
        generate_shape = (16, 42, 60, 104)
    elif args.twenty_secs:
        # 20 seconds video
        generate_shape = (16, 84, 60, 104)
    elif args.eight_secs:
        # 8.5 seconds video
        generate_shape = (16, 36, 60, 104)


    import math
    seq_len = math.prod(generate_shape[1:]) // 4
    print("seq_len", seq_len)
    # generate_shape = (16, 41, 60, 104)
    for index in tqdm(range(args.num_videos)):
        prompt = dataset[index]
        video = pipe.inference(
            noise=torch.randn(
                generate_shape, generator=seed_g,
                dtype=torch.float32, device=device
            ),
            text_prompts=[prompt],
            seed_g=seed_g,
            seq_len=seq_len
        )[0].permute(1, 2, 3, 0).cpu().numpy() # [c, f, h, w]

        video = video * 0.5 + 0.5 # [c, f, h, w]

        save_index = index * world_size + local_rank

        export_to_video(
            video, os.path.join(args.output_folder, f"output_{save_index:03d}.mp4"), fps=16)
        print(f"At Rank {local_rank}, Exported video {save_index} to {args.output_folder}/output_{save_index:03d}.mp4")

    print(f"Finished inference for {args.num_videos} videos at rank {local_rank}")

    dist.barrier()

    # delete the process group
    dist.destroy_process_group()

    if local_rank == 0:
        # Generate HTML file
        cmd = f"python inference_scripts/get_html.py {args.output_folder}"
        os.system(cmd)

        exp_name = output_folder.strip("/").split("/")[-3]
        if args.ten_secs:
            exp_name = f"{exp_name}_10secs"
        elif args.twenty_secs:
            exp_name = f"{exp_name}_20secs"
        # also push to huggingface
        cmd = f"python inference_scripts/push_to_huggingface.py {args.output_folder} --name {exp_name}"
        os.system(cmd)

if __name__ == "__main__":
    ### example command:
    # bash dist_inference.sh /sensei-fs/users/tianyuanz/projects/VideoTTT/experiments_v1/v1_debug_10sec_rep1_mf0.3_2_fullft_ar_5k_tttonorm_only_tttmix_sliding_window_4680_hybrid_bs64_lr5e-6_10x_wd0_ema0.995_ts3_block_swiglu_head768_inter_2x_weight_norm_ttscale_learn_qkl2norm_noattn_logitn/seed_200/config.yaml --checkpoint_folder /sensei-fs/users/tianyuanz/projects/VideoTTT/experiments_v1/v1_debug_10sec_rep1_mf0.3_2_fullft_ar_5k_tttonorm_only_tttmix_sliding_window_4680_hybrid_bs64_lr5e-6_10x_wd0_ema0.995_ts3_block_swiglu_head768_inter_2x_weight_norm_ttscale_learn_qkl2norm_noattn_logitn/seed_200/checkpoint_model_001999/ --ten_secs  --output_folder ../output_videos_debug_ar_new/ --ar
    main()
