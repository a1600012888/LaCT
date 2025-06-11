# LaCT Novel View Synthesis

Code and model release for [LaCT](https://tianyuanzhang.com/projects/ttt-done-right/) (Large-Chunk TTT) novel view synthesis.
The basic architecture was based on [LVSM](https://haian-jin.github.io/projects/LVSM/).
The checkpoints are pre-trained on [Objaverse](https://objaverse.allenai.org/) for obj-level and 
[DL3DV](https://dl3dv-10k.github.io/DL3DV-10K/) for scene level.

## TODO
- [x] LaCT NVS codebase
- [x] Object-level model checkpoints
- [ ] Scene-level model checkpoints
- [x] Object-level inference example
- [ ] Scene-level inference example
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

```
mkdir -p weight

# 256 Res checkpoint
wget https://huggingface.co/airsplay/lact_nvs/resolve/main/obj_res256.pt -O weight/obj_res256.pt

# 512 Res checkpoint
wget https://huggingface.co/airsplay/lact_nvs/resolve/main/obj_res512.pt -O weight/obj_res512.pt

# 1024 Res checkpoint
wget https://huggingface.co/airsplay/lact_nvs/resolve/main/obj_res1024.pt -O weight/obj_res1024.pt
```

## Inference
Run inference with example 512-resolution data in [data_example](/data_example/).
The command will output example videos like (at 512 resolution):
<p align="center">
  <img src="data_example/gso_character_inference_demo.gif" alt="Example Inference Result">
</p>


Objects: 
```bash
# Resolution 256:
python inference.py --load weight/obj_res256.pt --image_size 256 256 --data_path data_example/gso_sample_data_path.json

# Resolution 512:
python inference.py --load weight/obj_res512.pt --image_size 512 512 --data_path data_example/gso_sample_data_path.json
```



Note: the checkpoints are the same as the paper while the code and data are rewritten. For best inference performance, a uniform view selection is preferred. The current example takes random view selections to demonstrate robustness.


## Training Script 

We still working on providing an example training data.
The current training code is for a reference.

From scratch
```
torchrun \
--nproc_per_node=8 \
--standalone \
train.py --config config/lact_l14_d768_ttt2x.yaml --actckpt
```



