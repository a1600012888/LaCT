import math
import torch
import triton
import triton.language as tl
import itertools
from torch.library import triton_op, wrap_triton
from typing import Tuple


def get_autotune_configs(
    block_M_list=(64, 128),
    block_N_list=(64, 128, 256),
    block_K_list=(32, 64),
    num_stages_list=(2, 3),
    threads_list=(128, 256),
):
    configs = []
    for BM, BN, BK, stages, threads in itertools.product(
        block_M_list, block_N_list, block_K_list, num_stages_list, threads_list
    ):
        configs.append(
            triton.Config(
                {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
                num_warps=threads // 32,
                num_stages=stages,
            )
        )
    return configs


########################################################
# W0_W2: [B, 2M, K]
# W1: [B, K, M]
# X: [B, N, K]
# V: [B, N, K]
# -- Outputs --
# DY0_DY2: [B, 2M, N]
# Hidden: [B, M, N]
# This kernel supports dynamic shapes for B and N(sequence length) if you remove B and N from the autotune key.
########################################################


@triton.autotune(configs=get_autotune_configs(), key=["B", "M", "N", "K"])
@triton.jit
def _swiglu_three_bmm_kernel(
    w0_w2_ptr,
    w1_ptr,
    x_ptr,
    v_ptr,
    dy0_dy2_ptr,
    hidden_ptr,
    B,
    M: tl.constexpr,
    N,
    K: tl.constexpr,
    # strides for W0_W2 [B, 2M, K]
    s_w0w2_b,
    s_w0w2_m,
    s_w0w2_k,
    # strides for W1 [B, K, M]
    s_w1_b,
    s_w1_k,
    s_w1_m,
    # strides for X and V [B, N, K]
    s_x_b,
    s_x_n,
    s_x_k,
    # strides for outputs [B, 2M, N]
    s_dy0_dy2_b,
    s_dy0_dy2_m,
    s_dy0_dy2_n,
    # strides for Hidden [B, M, N]
    s_h_b,
    s_h_m,
    s_h_n,
    out_dtype: tl.constexpr,
    # meta
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for ragged tiles
    mask_m = offs_m < M
    mask_n = offs_n < N

    # FP32 accumulators
    acc_y0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_y2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_dh = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Bases for this batch
    w0_batch = w0_w2_ptr + pid_b * s_w0w2_b
    w2_batch = w0_w2_ptr + pid_b * s_w0w2_b + M * s_w0w2_m
    w1_batch = w1_ptr + pid_b * s_w1_b
    x_batch = x_ptr + pid_b * s_x_b
    v_batch = v_ptr + pid_b * s_x_b

    out_dtype_tl = (
        tl.float16
        if out_dtype == "fp16"
        else tl.bfloat16 if out_dtype == "bf16" else tl.float32
    )

    # Loop over K
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for ko in range(0, num_k_tiles):
        k0 = ko * BLOCK_K
        k_ids = k0 + offs_k
        mask_k = k_ids < K

        # A0 = W0[offs_m, k_ids] -> [M, K]
        a0_ptrs = w0_batch + (offs_m[:, None] * s_w0w2_m + k_ids[None, :] * s_w0w2_k)
        a0 = tl.load(a0_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # A2 = W2[offs_m, k_ids] -> [M, K]
        a2_ptrs = w2_batch + (offs_m[:, None] * s_w0w2_m + k_ids[None, :] * s_w0w2_k)
        a2 = tl.load(a2_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # A1 = (W1^T)[offs_m, k_ids] = W1[k_ids, offs_m]  -> [M, K]
        a1_ptrs = w1_batch + (k_ids[None, :] * s_w1_k + offs_m[:, None] * s_w1_m)
        a1 = tl.load(a1_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Bx = X^T[k_ids, offs_n] = X[offs_n, k_ids] -> [K, N]
        bx_ptrs = x_batch + (offs_n[None, :] * s_x_n + k_ids[:, None] * s_x_k)
        bx = tl.load(bx_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Bv = V^T[k_ids, offs_n] -> [K, N]
        bv_ptrs = v_batch + (offs_n[None, :] * s_x_n + k_ids[:, None] * s_x_k)
        bv = tl.load(bv_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # another way is to don't transpose the pointer, transpose the data in the register.
        # # A1 = W1[k_ids, offs_m] -> [K, M]
        # a1_ptrs = w1_batch + (k_ids[:, None] * s_w1_k + offs_m[None, :] * s_w1_m)
        # a1 = tl.load(a1_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)

        # Three GEMMs (FP32 accumulate)
        acc_y0 += tl.dot(a0, bx, out_dtype=tl.float32)
        acc_y2 += tl.dot(a2, bx, out_dtype=tl.float32)
        acc_dh += tl.dot(a1, bv, out_dtype=tl.float32)

    # --- Epilogue on fragments (FP32 math) ---
    # Hidden = swish(y0) * y2
    # DY0 = sigmoid(y0) * y2 * dh * (1 + y0 * (1 - sigmoid(y0)))
    # DY2 = swish(y0) * dh
    y0 = acc_y0
    y2 = acc_y2
    dh = acc_dh

    sigma = tl.sigmoid(y0)
    swish = sigma * y0

    hidden_tile = swish * y2
    dy0_tile = sigma * y2 * dh * (1.0 + y0 * (1.0 - sigma))
    dy2_tile = swish * dh

    # Store with Casting
    # [B, 2M, N]
    dy0_ptrs = (
        dy0_dy2_ptr
        + pid_b * s_dy0_dy2_b
        + (offs_m[:, None] * s_dy0_dy2_m + offs_n[None, :] * s_dy0_dy2_n)
    )
    dy2_ptrs = (
        dy0_dy2_ptr
        + pid_b * s_dy0_dy2_b
        + M * s_dy0_dy2_m
        + (offs_m[:, None] * s_dy0_dy2_m + offs_n[None, :] * s_dy0_dy2_n)
    )
    hid_ptrs = (
        hidden_ptr + pid_b * s_h_b + (offs_m[:, None] * s_h_m + offs_n[None, :] * s_h_n)
    )

    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(dy0_ptrs, dy0_tile.to(out_dtype_tl), mask=mask_out)
    tl.store(dy2_ptrs, dy2_tile.to(out_dtype_tl), mask=mask_out)
    tl.store(hid_ptrs, hidden_tile.to(out_dtype_tl), mask=mask_out)


def swiglu_backward_three_bmm_triton(W0_W2, W1, X, V):
    """
    Outs:
        Hidden: [B, M, N] in other words [B, Hidden, num_tokens]
        DY0_DY2: [B, 2M, N] in other words [B, 2 * Hidden, num_tokens]
    """
    B, M_times_2, K = W0_W2.shape
    M = M_times_2 // 2
    Bx, N, Kx = X.shape
    assert W1.shape == (B, K, M)
    assert V.shape == (B, N, K)
    assert W0_W2.dtype == W1.dtype == X.dtype == V.dtype == torch.bfloat16
    # this cause graph break in bwd pass, please make sure its contiguous when you call this function.
    # assert (
    #     W0_W2.is_contiguous()
    #     and W1.is_contiguous()
    #     and X.is_contiguous()
    #     and V.is_contiguous()
    # )

    # compute strides assuming contigous inputs
    s_w0w2_b, s_w0w2_m, s_w0w2_k = K * M_times_2, K, 1
    s_w1_b, s_w1_k, s_w1_m = K * M, M, 1
    s_x_b, s_x_n, s_x_k = K * N, K, 1

    s_dy0_dy2_b, s_dy0_dy2_m, s_dy0_dy2_n = M_times_2 * N, N, 1
    s_h_b, s_h_m, s_h_n = M * N, N, 1

    # Allocate outputs (compute dtype)
    Hidden = torch.empty((B, M, N), device=X.device, dtype=X.dtype)
    DY0_DY2 = torch.empty((B, M_times_2, N), device=X.device, dtype=X.dtype)

    out_dtype_str = (
        "float32"
        if X.dtype == torch.float32
        else "bf16" if X.dtype == torch.bfloat16 else "fp16"
    )

    # 3D grid (M-tiles, N-tiles, Batches)
    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), B)

    _swiglu_three_bmm_kernel[grid](
        W0_W2,
        W1,
        X,
        V,
        DY0_DY2,
        Hidden,
        B,
        M,
        N,
        K,
        # strides (element strides, not bytes)
        s_w0w2_b,
        s_w0w2_m,
        s_w0w2_k,
        s_w1_b,
        s_w1_k,
        s_w1_m,
        s_x_b,
        s_x_n,
        s_x_k,
        s_dy0_dy2_b,
        s_dy0_dy2_m,
        s_dy0_dy2_n,
        s_h_b,
        s_h_m,
        s_h_n,
        out_dtype=out_dtype_str,
    )

    return DY0_DY2, Hidden


### Warpping around with triton_op
@triton_op("lact::triton_swiglu_bwd", mutates_args={})
def swiglu_backward_three_bmm_triton_op(
    W0_W2: torch.Tensor, W1: torch.Tensor, X: torch.Tensor, V: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, M_times_2, K = W0_W2.shape
    M = M_times_2 // 2
    Bx, N, Kx = X.shape
    assert W1.shape == (B, K, M)
    assert V.shape == (B, N, K)
    assert W0_W2.dtype == W1.dtype == X.dtype == V.dtype == torch.bfloat16
    # this cause graph break in bwd pass, please make sure its contiguous when you call this function.
    # assert (
    #     W0_W2.is_contiguous()
    #     and W1.is_contiguous()
    #     and X.is_contiguous()
    #     and V.is_contiguous()
    # )

    # choose a compute dtype (default: X.dtype)

    # compute strides assuming contigous inputs
    s_w0w2_b, s_w0w2_m, s_w0w2_k = K * M_times_2, K, 1
    s_w1_b, s_w1_k, s_w1_m = K * M, M, 1
    s_x_b, s_x_n, s_x_k = K * N, K, 1

    s_dy0_dy2_b, s_dy0_dy2_m, s_dy0_dy2_n = M_times_2 * N, N, 1
    s_h_b, s_h_m, s_h_n = M * N, N, 1

    # Allocate outputs (compute dtype)
    Hidden = torch.empty((B, M, N), device=X.device, dtype=X.dtype)
    DY0_DY2 = torch.empty((B, M_times_2, N), device=X.device, dtype=X.dtype)

    out_dtype_str = (
        "float32"
        if X.dtype == torch.float32
        else "bf16" if X.dtype == torch.bfloat16 else "fp16"
    )

    # 3D grid (M-tiles, N-tiles, Batches)
    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), B)

    wrap_triton(_swiglu_three_bmm_kernel)[grid](
        W0_W2,
        W1,
        X,
        V,
        DY0_DY2,
        Hidden,
        B,
        M,
        N,
        K,
        # strides (element strides, not bytes)
        s_w0w2_b,
        s_w0w2_m,
        s_w0w2_k,
        s_w1_b,
        s_w1_k,
        s_w1_m,
        s_x_b,
        s_x_n,
        s_x_k,
        s_dy0_dy2_b,
        s_dy0_dy2_m,
        s_dy0_dy2_n,
        s_h_b,
        s_h_m,
        s_h_n,
        out_dtype=out_dtype_str,
    )

    return DY0_DY2, Hidden


def swiglu_backward_three_bmm_ref(W0_W2, W1, X, V):
    """
    Reference implementation in PyTorch (for correctness checks).

    Shapes / layouts (contiguous, unless noted):
      W0: [B, M, K]
      W1: [B, K, M]   (note transpose in math)
      W2: [B, M, K]
      X : [B, N, K]
      V : [B, N, K]
    Returns:
      Hidden: [B, M, N] = swish(Y0) * Y2
      DY0   : [B, M, N]
      DY2   : [B, M, N]
    """

    # Cast to a "compute dtype" (bf16 or f16 is fine; we keep accum in fp32).
    # Mimics "cast-on-copy" behavior from your TileLang kernel.
    compute_dtype = X.dtype
    W0, W2 = W0_W2.chunk(2, dim=1)

    # GEMMs (accumulate in fp32 for accuracy)
    # Y0 = W0 @ X^T
    Y0 = torch.matmul(W0, X.transpose(-1, -2))
    # Y2 = W2 @ X^T
    Y2 = torch.matmul(W2, X.transpose(-1, -2))
    # DH = (W1^T) @ V^T
    DH = torch.matmul(W1.transpose(-1, -2), V.transpose(-1, -2))

    # Epilogue
    sigma = torch.sigmoid(Y0)
    swish = sigma * Y0
    Hidden = swish * Y2
    DY0 = sigma * Y2 * DH * (1.0 + Y0 * (1.0 - sigma))
    DY2 = swish * DH

    DY0_DY2 = torch.cat([DY0, DY2], dim=1)
    return DY0_DY2, Hidden


def make_inputs(B, M, K, N, dtype=torch.bfloat16, device="cuda"):
    W0_W2 = torch.randn(B, 2 * M, K, device=device, dtype=dtype)
    W1 = torch.randn(B, K, M, device=device, dtype=dtype)
    X = torch.randn(B, N, K, device=device, dtype=dtype)
    V = torch.randn(B, N, K, device=device, dtype=dtype)
    return W0_W2, W1, X, V


@torch.no_grad()
def check_correctness(B=2, M=256, N=192, K=320, dtype=torch.bfloat16, device="cuda"):
    torch.manual_seed(0)
    from .benchmark import report_error

    inputs = make_inputs(B, M, K, N, dtype, device)
    ref_DY0_DY2, ref_H = swiglu_backward_three_bmm_ref(*inputs)

    fp32_inputs = [_.to(torch.float32) for _ in inputs]
    tt_DY0_DY2, tt_H = swiglu_backward_three_bmm_triton(*fp32_inputs)

    report_error(ref_H, tt_H, "Hidden")
    report_error(ref_DY0_DY2, tt_DY0_DY2, "DY0_DY2")


if __name__ == "__main__":
    check_correctness()
