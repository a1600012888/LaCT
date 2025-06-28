# Copyright (c) 2025 Hao Tan and Tianyuan Zhang
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
"""
Sequence parallel support helper functions. 
"""

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather


_SP_GROUP = None


def init_sp_group(sp_size):
    global _SP_GROUP

    sp_group, sp_groups = dist.new_subgroups(group_size=sp_size)
    _SP_GROUP = sp_group


def is_sp():
    return _SP_GROUP is not None


def get_sp_rank():
    return dist.get_rank(_SP_GROUP)


def get_sp_world_size():
    return dist.get_world_size(_SP_GROUP)


def sp_broadcast(x: torch.Tensor):
    if dist.get_rank(_SP_GROUP) != 0:
        pass

    # Note: in torch 2.7, there is group_src that you can just specify the group's rank.
    #       we are still at torch 2.5, thus need to use the global rank.
    dist.broadcast(x, src=dist.get_global_rank(_SP_GROUP, 0), group=_SP_GROUP)

    return x


def slice_tensor(x, dim, start, end):
    slices = [slice(None)] * x.dim()  # create slices for all dims
    slices[dim] = slice(start, end)   # set slice for the desired dim
    return x[tuple(slices)]


def pad0_tensor(x, dim, pad_len):
    if pad_len <= 0:
        return x  # no padding needed

    # Create a zero tensor of the same dtype and device as x
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    pad_tensor = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)

    # Concatenate along the specified dimension
    return torch.cat([x, pad_tensor], dim=dim)


def gather_scatter(x, gather_dim=2, scatter_dim=1, process_group=None):
    # [b, N, s, d]
    all_x = all_gather(x, group=process_group)
    local_rank = process_group.rank()

    scatter_size = process_group.size()
    scatter_len = x.size(scatter_dim)
    assert scatter_len % scatter_size == 0, f"scatter_len {scatter_len} is not divisible by scatter_size {scatter_size}"
    scatter_stride = scatter_len // scatter_size

    # [b, n, s, d]
    all_x = [
        slice_tensor(x, scatter_dim, local_rank * scatter_stride, (local_rank + 1) * scatter_stride)
        for x in all_x
    ]

    # [b, n, S, d]
    all_x = torch.cat(all_x, dim=gather_dim)

    return all_x


def sp_gather_scatter(x, gather_dim=2, scatter_dim=1):
    return gather_scatter(x, gather_dim=gather_dim, scatter_dim=scatter_dim, process_group=_SP_GROUP)


def local_scatter(x, scatter_dim=1, process_group=None):
    local_rank = process_group.rank()

    scatter_size = process_group.size()
    scatter_len = x.size(scatter_dim)
    assert scatter_len % scatter_size == 0, f"scatter_len {scatter_len} is not divisible by scatter_size {scatter_size}"
    scatter_stride = scatter_len // scatter_size

    # [b, n, s, d]
    all_x = slice_tensor(x, scatter_dim, local_rank * scatter_stride, (local_rank + 1) * scatter_stride)

    return all_x


def sp_local_scatter(x, scatter_dim=1):
    return local_scatter(x, scatter_dim=scatter_dim, process_group=_SP_GROUP)


def sp_input_broadcast_scatter(x, scatter_dim=1):
    """
    Note:
        1. need all rank have the same size and type of x.
        2. it will auto-pad and shard among the scatter_dim given the sp world size.
    """

    x = sp_broadcast(x)

    # [b, n, S, d]
    sp_world_size = get_sp_world_size()

    # If the input can not be divided by the sp size, pad it
    if x.size(scatter_dim) % sp_world_size != 0:
        pad_len = sp_world_size - (x.size(scatter_dim) % sp_world_size)
        x = pad0_tensor(x, scatter_dim, pad_len)

    # [b, n, s, d]
    x = sp_local_scatter(x, scatter_dim=scatter_dim)
    return x


def sp_all_gather(x, gather_dim=2, length=-1):
    """
    Note:
        1. need all rank have the same size and type of x.
    """

    # [b, n, S, d]
    x = all_gather(x, group=_SP_GROUP)      # list of x's
    x = torch.cat(x, dim=gather_dim)

    if length != -1:
        x = slice_tensor(x, gather_dim, 0, length)

    return x


if __name__ == "__main__":
    # export NCCL_DEBUG=0; torchrun --nproc_per_node=8 --rdzv_id=100 --rdzv_endpoint=localhost:29400 sp_support.py
    dist.init_process_group(backend="nccl")

    init_sp_group(sp_size=2)

    print(f"rank: {dist.get_rank()}, sp rank: {get_sp_rank()}")

    # Set default device to local rank
    local_rank = dist.get_rank() % 8
    torch.cuda.set_device(local_rank)

    a = torch.ones(1, 8, 1).cuda() * dist.get_rank()

    a.requires_grad = True

    b = sp_gather_scatter(a, gather_dim=2, scatter_dim=1)

    (dist.get_rank() * b).sum().backward()

    # print(b)
    # print(f"rank: {dist.get_rank()} grad", a.grad)

    x = torch.ones(1, 7, 1).cuda() * dist.get_rank()
    x = sp_input_broadcast_scatter(x, scatter_dim=1)
    print(f"rank: {dist.get_rank()} input scatter {x}")



