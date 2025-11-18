import math
import time
import torch
import triton
import triton.language as tl
import itertools


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


@triton.autotune(configs=get_autotune_configs(), key=["B", "M", "N", "K"])
@triton.jit
def _swiglu_three_bmm_with_lr_kernel(
    w0_w2_ptr,
    w1_ptr,
    x_ptr,
    v_ptr,
    lr0_ptr,  # scales DY0
    lr1_ptr,  # scales Hidden
    lr2_ptr,  # scales DY2
    # outputs
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
    # strides for W1 [B, K, M]  (note axes)
    s_w1_b,
    s_w1_k,
    s_w1_m,
    # strides for X and V [B, N, K]
    s_x_b,
    s_x_n,
    s_x_k,
    # NEW: strides for lr* [B, N]
    s_lr_b,
    s_lr_n,
    # strides for dy0_dy2 [B, 2M, N]
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

    # --- K loop ---
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

    # --- Load per-(B, N) scaling vectors (once per tile), cast to fp32 ---
    lr0_batch = lr0_ptr + pid_b * s_lr_b
    lr1_batch = lr1_ptr + pid_b * s_lr_b
    lr2_batch = lr2_ptr + pid_b * s_lr_b

    lr0_vec = tl.load(lr0_batch + offs_n * s_lr_n, mask=mask_n, other=0.0).to(
        tl.float32
    )
    lr1_vec = tl.load(lr1_batch + offs_n * s_lr_n, mask=mask_n, other=0.0).to(
        tl.float32
    )
    lr2_vec = tl.load(lr2_batch + offs_n * s_lr_n, mask=mask_n, other=0.0).to(
        tl.float32
    )

    # Broadcast to [BLOCK_M, BLOCK_N] and scale.
    dy0_tile *= lr0_vec[None, :]
    dy2_tile *= lr2_vec[None, :]
    hidden_tile *= lr1_vec[None, :]

    # Store with Casting
    out_dtype_tl = (
        tl.float16
        if out_dtype == "fp16"
        else tl.bfloat16 if out_dtype == "bf16" else tl.float32
    )

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


def swiglu_backward_three_bmm_with_lr_triton(W0_W2, W1, X, V, lr0, lr1, lr2):
    """
    Args:
        W0: [B, M, K] - [B, Hidden, D]
        W1: [B, K, M] - [B, D, Hidden]
        W2: [B, M, K] - [B, Hidden, D]
        X: [B, N, K] - [B, num_tokens, D]
        V: [B, N, K] - [B, num_tokens, D]
        lr0: [B, N] - [B, num_tokens]  bf16 or fp32
        lr1: [B, N] - [B, num_tokens]  bf16 or fp32
        lr2: [B, N] - [B, num_tokens]  bf16 or fp32
    Returns:
        DY0: [B, M, N] in other words [B, Hidden, num_tokens]
        DY2: [B, M, N] in other words [B, Hidden, num_tokens]
        Hidden: [B, M, N] in other words [B, Hidden, num_tokens]
    """
    B, M_times_2, K = W0_W2.shape
    M = M_times_2 // 2
    Bx, N, Kx = X.shape
    assert W1.shape == (B, K, M)
    assert V.shape == (B, N, K)
    assert (
        W0_W2.dtype == torch.bfloat16 and V.dtype == torch.bfloat16
    ), "W0_W2 and V must be bf16"
    assert (
        W0_W2.is_contiguous()
        and W1.is_contiguous()
        and X.is_contiguous()
        and V.is_contiguous()
    )
    assert lr0.shape == (B, N) and lr1.shape == (B, N) and lr2.shape == (B, N)

    # compute strides assuming contigous inputs
    s_w0w2_b, s_w0w2_m, s_w0w2_k = K * M_times_2, K, 1
    s_w1_b, s_w1_k, s_w1_m = K * M, M, 1
    s_x_b, s_x_n, s_x_k = K * N, K, 1
    s_lr_b, s_lr_n = lr0.stride(0), lr0.stride(1)

    s_dy0_dy2_b, s_dy0_dy2_m, s_dy0_dy2_n = M_times_2 * N, N, 1
    s_h_b, s_h_m, s_h_n = M * N, N, 1

    # Allocate outputs (compute dtype)
    Hidden = torch.empty((B, M, N), device=X.device, dtype=X.dtype)
    DY0_DY2 = torch.empty((B, M_times_2, N), device=X.device, dtype=X.dtype)

    # Make the store dtype match the destination tensors (robust if W0.dtype != X.dtype)
    out_dtype_str = (
        "float32"
        if Hidden.dtype == torch.float32
        else "bf16" if Hidden.dtype == torch.bfloat16 else "fp16"
    )

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), B)

    _swiglu_three_bmm_with_lr_kernel[grid](
        W0_W2,
        W1,
        X,
        V,
        lr0,
        lr1,
        lr2,
        DY0_DY2,
        Hidden,
        B,
        M,
        N,
        K,
        # strides (element strides)
        s_w0w2_b,
        s_w0w2_m,
        s_w0w2_k,
        s_w1_b,
        s_w1_k,
        s_w1_m,
        s_x_b,
        s_x_n,
        s_x_k,
        s_lr_b,
        s_lr_n,
        s_dy0_dy2_b,
        s_dy0_dy2_m,
        s_dy0_dy2_n,
        s_h_b,
        s_h_m,
        s_h_n,
        out_dtype=out_dtype_str,
    )

    return DY0_DY2, Hidden


@torch.no_grad()
def ref_func(W0_W2, W1, X, dOut, lr0, lr1, lr2):
    """
    Shapes:
      W0: [B, M, K]
      W1: [B, K, M]
      W2: [B, M, K]
      X : [B, N, K]
      dOut (a.k.a. V): [B, N, K]
      lr0, lr1, lr2: [B, N]
    Returns (for convenience): DY0, DY2, Hidden
    """
    W0, W2 = W0_W2.chunk(2, dim=1)
    Y0 = torch.bmm(W0, X.transpose(1, 2))  # [B, M, N]
    Y2 = torch.bmm(W2, X.transpose(1, 2))  # [B, M, N]
    Hidden = torch.nn.functional.silu(Y0) * Y2
    DHidden = torch.bmm(W1.transpose(1, 2), dOut.transpose(1, 2))  # [B, M, N]
    DY0 = DHidden * Y2 * torch.sigmoid(Y0) * (1 + Y0 * (1 - torch.sigmoid(Y0)))
    DY2 = DHidden * torch.nn.functional.silu(Y0)

    # Column-wise scalings: match kernel
    DY0 = DY0 * lr0.unsqueeze(1)  # [B, 1, N]
    DY2 = DY2 * lr2.unsqueeze(1)  # [B, 1, N]
    Hidden = Hidden * lr1.unsqueeze(1)
    return torch.cat([DY0, DY2], dim=1), Hidden


def make_inputs(B, H, D, L, lr_dtype=torch.bfloat16):
    """
    W0, W1: [B, K, M]
    X0, X1: [B, K, N]

    W2:     [B, M, N]
    X2:     [B, K, M]
    """
    device = torch.device("cuda", torch.cuda.current_device())
    W0_W2 = torch.randn(
        B, 2 * H, D, device=device, dtype=torch.bfloat16, requires_grad=True
    )

    W1 = torch.randn(B, D, H, device=device, dtype=torch.bfloat16, requires_grad=True)

    X = torch.randn(B, L, D, device=device, dtype=torch.bfloat16, requires_grad=True)

    dOut = torch.randn(B, L, D, device=device, dtype=torch.bfloat16, requires_grad=True)

    lr0 = torch.randn(B, L, device=device, dtype=lr_dtype, requires_grad=True)
    lr1 = torch.randn(B, L, device=device, dtype=lr_dtype, requires_grad=True)
    lr2 = torch.randn(B, L, device=device, dtype=lr_dtype, requires_grad=True)

    return W0_W2, W1, X, dOut, lr0, lr1, lr2


def check_correctness():

    device = torch.device("cuda", torch.cuda.current_device())

    from .benchmark import report_error

    inps = make_inputs(4, 2048, 1024, 8192, lr_dtype=torch.float32)
    DY0_DY2, Hidden = swiglu_backward_three_bmm_with_lr_triton(*inps)

    fp32_inps = [_.to(torch.float32) for _ in inps]
    DY0_DY2_ref, Hidden_ref = ref_func(*fp32_inps)

    # print(torch.allclose(DY0, DY0_ref))
    # print(torch.allclose(DY2, DY2_ref))
    # print(torch.allclose(Hidden, Hidden_ref))

    report_error(DY0_DY2_ref, DY0_DY2, "DY0_DY2")
    report_error(Hidden_ref, Hidden, "Hidden")


if __name__ == "__main__":
    check_correctness()
