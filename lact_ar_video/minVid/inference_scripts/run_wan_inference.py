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
import random
import sys

class TextDataset(Dataset):
    def __init__(self, data_path):
        self.texts = []
        with open(data_path, "r") as f:
            for line in f:
                self.texts.append(line.strip())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="Path to YAML configuration file")
parser.add_argument("--checkpoint_folder", type=str, default=None)
parser.add_argument("--output_folder", type=str, default="../output_videos/")
parser.add_argument("--prompt_file_path", type=str, default="sample_dataset/high_motion_prompts_50.txt")
parser.add_argument("--num_videos", type=int, default=35)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ar", action="store_true", default=False)
parser.add_argument("--debug", action="store_true", default=False)
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


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.set_grad_enabled(False)
device = torch.device("cuda")

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

model_config = config.model
if args.checkpoint_folder is not None:
    checkpoint_path = os.path.join(args.checkpoint_folder, "model.pt")
    model_config.generator_ckpt = checkpoint_path
else:
    if "14b" in args.config:
        checkpoint_path = "/mnt/localssd/minVid/ckpt/wan/Wan2.1-T2V-14B/"
    else:
        checkpoint_path = "/mnt/localssd/minVid/ckpt/wan/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
    model_config.generator_ckpt = checkpoint_path

if args.ar:
    print("Using ARWanInferencePipeline")
    pipe = ARWanInferencePipeline(model_config, device=device)
else:
    pipe = WanInferencePipeline(model_config, device=device)
pipe.eval()
pipe = pipe.to(device="cuda", dtype=torch.bfloat16)

# compile the generator
# pipe.generator = torch.compile(pipe.generator)

# seed for inference
seed = args.seed
seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
seed_g = torch.Generator(device=device)
seed_g.manual_seed(seed)

dataset = TextDataset(args.prompt_file_path)

os.makedirs(args.output_folder, exist_ok=True)


# for index in tqdm(range(len(dataset))):
generate_shape = (16, 21, 60, 104)
# generate_shape = (16, 21, 90, 156)
# generate_shape = (16, 41, 60, 104)

if args.debug:
    generate_shape = (16, 42, 60, 104)


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

    export_to_video(
        video, os.path.join(args.output_folder, f"output_{index:03d}.mp4"), fps=16)


# Generate HTML file
import os
cmd = f"python inference_scripts/get_html.py {args.output_folder}"
os.system(cmd)

exp_name = output_folder.strip("/").split("/")[-2]

# also push to huggingface
cmd = f"python inference_scripts/push_to_huggingface.py {args.output_folder} --name {exp_name}"
os.system(cmd)
