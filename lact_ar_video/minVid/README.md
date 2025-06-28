
# LaCT for AutoRegressive Video Generation

This repository contains the official code for [LaCT (Large-Chunk TTT)](https://tianyuanzhang.com/projects/ttt-done-right/) for autoregressive video generation, built upon the [Wan T2V codebase](https://github.com/Wan-Video/Wan2.1).


## Key Features
- **Autoregressive Video Generation** with teacher-forcing training.
- Support for repeating the noise chunk multiple times to increase token utilization.
- Timestep shifting with logit-normal weighting during training.
- Memory efficient with FSDP, distributed ckpt and activation ckpt. 
- Sequence parallelization for attention layers and LaCT. 


### Environment
Here is the environment for the [WanT2V](https://github.com/Wan-Video/Wan2.1/blob/main/requirements.txt)

We use this 
```
pip install -r requirement.txt
```
Also, please install minVid by
```
cd lact_ar_video
pip install -e .
```

Note: Since this code utilizes FSDP 2.0, it’s recommended to use a recent version of PyTorch with optimal support for FSDP 2.0.

### How to train

Implement your dataset and dataloader in the `data/` directory. Specifically, define your own `get_data_module` function in `data/`, which returns a data loader.

Example usage in train.py:
```
from minVid.data import get_data_module

data_config = config.dataset_train
data_module = get_data_module(data_config, data_seed=data_seed)  # data_seed is an integer
dataloader = data_module.train_dataloader()

data_batch = next(dataloader)
# data_batch["frames"] => [batch_size, 3, num_frames, height, width], RGB videos normalized in [0, 1]
# data_batch["caption"] => list of caption strings per video clip
```

Review and adjust the configuration file (e.g., `configs/ar/lact_ar_chunk3f_1.3B.yaml`), ensuring the pretrained Wan checkpoint path, dataloader configuration, output directories, and your wandb account settings are correctly set. Include your wandb key in `./api_keys.yaml`.

Launch training with:
```
bash launch_train.sh configs/ar/lact_ar_chunk3f_1.3B.yaml -s exp_name <wandb-job-name>
```

To overwrite additional parameters, use the `-s `flag, examples:
```
-s train.lr=0.00001 -s train.use_ema=False -s train.sp_size 2
```

By setting `train.sp_size` larger than 1 will enable sequence parallel automatically. Note current implementation only supports `train.sp_size` is dividable by the number of heads. 

### How to generate videos
There are two provided inference scripts:

Distributed Inference: (`run_wan_inference_distributed.py`)
```
bash dist_inference.sh <config-path> --checkpoint_folder <path-to-checkpoint-folder> --output_folder <path-to-output-folder> --ar
```


Single Process Inference: (`run_wan_inference.py`)
```
python3 inference_scripts/run_wan_inference.py <config-path> --checkpoint_folder <path-to-checkpoint-folder> --output_folder <path-to-output-folder> --ar
```

You can change `--prompt_file_path`  to use different text prompt files when generating the videos.  
The `--ar` flag will use inference code for auto-regressive video gen.


### Acknowledgement 

This codebase built upon [CausVid](https://github.com/tianweiy/CausVid) and [WanT2V](https://github.com/Wan-Video/Wan2.1)
