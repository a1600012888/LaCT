# LaCT Novel View Synthesis

Code and model release for [LaCT](https://tianyuanzhang.com/projects/ttt-done-right/) (Large-Chunk TTT) novel view synthesis.
The basic architecture was based on [LVSM](https://haian-jin.github.io/projects/LVSM/).
The checkpoints are pre-trained on [Objaverse](https://objaverse.allenai.org/) for obj-level and 
[DL3DV](https://dl3dv-10k.github.io/DL3DV-10K/) for scene level.

## TODO
- [x] LaCT NVS codebase
- [x] Object-level model checkpoints
- [x] Scene-level model checkpoints
- [x] Object-level inference example
- [x] Scene-level inference example
- [x] Training code and script
- [ ] Training data

## Environment Setup
Install the python dependencies:
```
pip install -r requirement.txt
```

Install `ffmpeg` to save rendering results as mp4 video:
```
sudo apt install ffmpeg
```
If ffmpeg can not be installed, changing `*_turntable.mp4` to `*_turntable.gif` in code and it will save in gif (but the size of video is larger).

## Download Pre-trained checkpoints
The weight is stored on [huggingface/airsplay/lact_nvs](https://huggingface.co/airsplay/lact_nvs).
You can also follow the script below to download the weights directly.
If the server's IP has trouble with wget, trying downloading with `huggingface-hub`.

*Object Checkpoints*
```
mkdir -p weight

# 256 Res checkpoint
wget https://huggingface.co/airsplay/lact_nvs/resolve/main/obj_res256.pt -O weight/obj_res256.pt

# 512 Res checkpoint
wget https://huggingface.co/airsplay/lact_nvs/resolve/main/obj_res512.pt -O weight/obj_res512.pt

# 1024 Res checkpoint
wget https://huggingface.co/airsplay/lact_nvs/resolve/main/obj_res1024.pt -O weight/obj_res1024.pt
```


*Scene checkpoints*

Checkpoints with config `config/lact_l24_d768_ttt2x.yaml`. These models trained with 32 inputs.
```
mkdir -p weight 

# 128x128 Resolution checkpoint
wget https://huggingface.co/airsplay/lact_nvs/resolve/main/scene_res128x128.pt -O weight/scene_res128x128.pt

# 256x256 Resolution checkpoint  
wget https://huggingface.co/airsplay/lact_nvs/resolve/main/scene_res256x256.pt -O weight/scene_res256x256.pt

# 512x512 Resolution checkpoint
wget https://huggingface.co/airsplay/lact_nvs/resolve/main/scene_res512x512.pt -O weight/scene_res512x512.pt
```


Checkpoints with config `config/lact_l24_d768_ttt4x.yaml`. These models are trained longer and trained with 64 inputs.
```
mkdir -p weight 

# 72x128 Resolution checkpoint
wget https://huggingface.co/airsplay/lact_nvs/resolve/main/scene_res72x128.pt -O weight/scene_res72x128.pt

# 144x256 Resolution checkpoint
wget https://huggingface.co/airsplay/lact_nvs/resolve/main/scene_res144x256.pt -O weight/scene_res144x256.pt

# 288x512 Resolution checkpoint
wget https://huggingface.co/airsplay/lact_nvs/resolve/main/scene_res288x512.pt -O weight/scene_res288x512.pt

# 536x960 Resolution checkpoint
wget https://huggingface.co/airsplay/lact_nvs/resolve/main/scene_res536x960.pt -O weight/scene_res536x960.pt
```


## Inference


### Object-level inference: 
Run inference with example 512-resolution data in [data_example](/data_example/).
```bash
# Resolution 256:
python inference.py --load weight/obj_res256.pt --image_size 256 256 --data_path data_example/gso_sample_data_path.json

# Resolution 512:
python inference.py --load weight/obj_res512.pt --image_size 512 512 --data_path data_example/gso_sample_data_path.json
```
The command will output example videos like (at 512 resolution):
<p align="center">
  <img src="data_example/gso_character_inference_demo.gif" alt="Example Inference Result">
</p>

Note: the checkpoints are the same as the paper while the code and data are rewritten. For best inference performance, a uniform view selection is preferred. The current example takes random view selections to demonstrate robustness.


### Scene-level inference
First download the DL3DV benchmark (i.e., test; not used in training) data samples:
```bash
python data_preprocess/dl3dv_eval_download.py --odir ./data_example/dl3dv_benchmark --subset hash --only_level4 --hash 032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7

python data_preprocess/dl3dv_eval_download.py --odir ./data_example/dl3dv_benchmark --subset hash --only_level4 --hash 0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695

python data_preprocess/dl3dv_eval_download.py --odir ./data_example/dl3dv_benchmark --subset hash --only_level4 --hash 341b4ff3dfd3d377d7167bd81f443bedafbff003bf04881b99760fc0aeb69510
```
Can grub more scenes with hashid listed [here](https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark/tree/main). The above scenes are just randomly picked from the above website.

After downloading the above samples, run
```bash
python data_preprocess/dl3dv_format_converter.py
```
to convert to a unified format. Thanks to [Ziwen Chen](https://chenziwe.com/) for the undistort code!

Run inference with example 512-resolution data in [data_example](/data_example/).
```bash
# Resolution 256 x 256, Num input views 64:
python inference.py \
--load weight/scene_res256x256.pt \
--config config/lact_l24_d768_ttt2x.yaml \
--image_size 256 256 \
--scene_inference \
--num_all_views 136 --num_input_views 128 \
--data_path data_example/dl3dv_sample_data_path.json 

# Resolution 72 x 128, Num input views 256:
python inference.py \
--load weight/scene_res72x128.pt \
--config config/lact_l24_d768_ttt4x.yaml \
--image_size 72 128 \
--scene_inference \
--num_all_views 300 --num_input_views 256 \
--data_path data_example/dl3dv_sample_data_path.json 

# Resolution 536 x 960, Num input views 32:
python inference.py \
--load weight/scene_res536x960.pt \
--config config/lact_l24_d768_ttt4x.yaml \
--image_size 536 960 \
--scene_inference \
--num_all_views 52  --num_input_views 48 \
--data_path data_example/dl3dv_sample_data_path.json 
```

Comparing to object-level inference, scene-level inference uses option `--scene_inference`. As scene poses can be unbounded, this normalizes the camera pose to be more regular. We normalize train+test together for simplicity in this codebase; feel free to normalize train only. This option also changes the video's camera trajetories (now would be interpolation over input). 

Note:
1. The paper's results used the kmeans centroids of cameras for better coverage of the scene. Please refer to Ziwen's release [here](https://github.com/arthurhero/Long-LRM). This release just used random samples.
2. The paper's video are rendered with a more smooth interpolation. 

## Training Script 

We still working on providing an example training data.
The current training code is for a reference.

Note:
1. The TTT model needs the option `--compile` to reach the best performance (about 1.4x~1.5x speedup). However, compilation for the first training step can take about 30s~2min. Thus we recommend removing it for debugging the code. 
2. Please follow train.py to add activation checkpointing; o.w., it can hurt the compilation of backward.

From scratch:
```
torchrun \
--nproc_per_node=8 \
--standalone \
train.py --config config/lact_l14_d768_ttt2x.yaml --actckpt
```



