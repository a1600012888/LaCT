# Doc: https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/optimizer_schedules#transformers.get_constant_schedule_with_warmup

import inspect

import torch
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from minVid.utils.dist_utils import print_rank0


def configure_optimizer(model, weight_decay, learning_rate, betas):
    # start with all of the candidate parameters
    all_param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in all_param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params, nodecay_params = [], []
    decay_names, nodecay_names = [], []
    for n, p in param_dict.items():
        # if nerf_mlp uses tinycudann, then its mlp parameters are 1D due to flattening
        if p.dim() >= 2:
            decay_params.append(p)
            decay_names.append(n)
        else:
            nodecay_params.append(p)
            nodecay_names.append(n)

    print_rank0(
        f"Decay params ({len(decay_params)}): {decay_names}\n"
        f"No Decay params ({len(nodecay_params)}): {nodecay_names}"
    )

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and next(model.parameters()).is_cuda
    # print(f"Using fused AdamW? {use_fused}")
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )

    return optimizer, param_dict, all_param_dict


def configure_lr_scheduler(
    optimizer, total_train_steps, warm_up_steps, scheduler_type="cosine"
):
    if scheduler_type == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=total_train_steps,
        )
    elif scheduler_type == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=total_train_steps,
        )
    elif scheduler_type == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up_steps,
        )
    else:
        raise ValueError(f"Not support LR scheduler type {scheduler_type}.")

    return lr_scheduler
