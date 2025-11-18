import torch
import triton
import triton.language as tl
import itertools


def two_mm(W0, X0, W1, X1, A_transpose=True, B_transpose=True):
    if A_transpose and B_transpose:
        return W0.transpose(1, 2) @ X0.transpose(1, 2) + W1.transpose(
            1, 2
        ) @ X1.transpose(1, 2)
    if A_transpose and not B_transpose:
        return W0.transpose(1, 2) @ X0 + W1.transpose(1, 2) @ X1
    raise NotImplementedError(
        "Only the two variants with A_transpose=True are implemented here."
    )


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
# O = W1.transpose(1, 2) @ X1T.transpose(1, 2) + W0.transpose(1, 2) @ X0T.transpose(1, 2)
# W0 and W1 of shape [B, K, M]
# X0T and X1T of shape [B, N, K]
# O of shape [B, M, N]
########################################################
@triton.autotune(configs=get_autotune_configs(), key=["B", "M", "N", "K"])
@triton.jit
def fused_two_mm_wT_xT_kernel(
    W0,
    W1,
    X0T,
    X1T,
    O,
    B,
    M,
    N,
    K,
    # W0 strides: [B, K, M]
    stride_w0_b,
    stride_w0_k,
    stride_w0_m,
    # W1 strides: [B, K, M]
    stride_w1_b,
    stride_w1_k,
    stride_w1_m,
    # X0T strides: [B, N, K]
    stride_x0t_b,
    stride_x0t_n,
    stride_x0t_k,
    # X1T strides: [B, N, K]
    stride_x1t_b,
    stride_x1t_n,
    stride_x1t_k,
    # O strides: [B, M, N]
    stride_o_b,
    stride_o_m,
    stride_o_n,
    OUT_IS_BF16: tl.constexpr,
    OUT_IS_FP16: tl.constexpr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    tl.multiple_of(offs_k, BLOCK_K)
    tl.max_contiguous(offs_m, BLOCK_M)
    tl.max_contiguous(offs_n, BLOCK_N)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        # A tiles: W^T -> treat W as (M,K) via strides (W is stored [K, M])
        a0_ptrs = (
            W0
            + pid_b * stride_w0_b
            + offs_m[:, None] * stride_w0_m
            + k[None, :] * stride_w0_k
        )

        # B tiles: X stored as (N,K) -> we need (K,N) by swapping strides
        b0_ptrs = (
            X0T
            + pid_b * stride_x0t_b
            + k[:, None] * stride_x0t_k
            + offs_n[None, :] * stride_x0t_n
        )

        mask_a = (offs_m[:, None] < M) & (k[None, :] < K)
        mask_b = (k[:, None] < K) & (offs_n[None, :] < N)

        a0 = tl.load(a0_ptrs, mask=mask_a, other=0.0)
        b0 = tl.load(b0_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a0, b0)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        # A tiles: W^T -> treat W as (M,K) via strides (W is stored [K, M])

        a0_ptrs = (
            W1
            + pid_b * stride_w1_b
            + offs_m[:, None] * stride_w1_m
            + k[None, :] * stride_w1_k
        )

        # B tiles: X stored as (N,K) -> we need (K,N) by swapping strides

        b0_ptrs = (
            X1T
            + pid_b * stride_x1t_b
            + k[:, None] * stride_x1t_k
            + offs_n[None, :] * stride_x1t_n
        )

        mask_a = (offs_m[:, None] < M) & (k[None, :] < K)
        mask_b = (k[:, None] < K) & (offs_n[None, :] < N)

        a0 = tl.load(a0_ptrs, mask=mask_a, other=0.0)
        b0 = tl.load(b0_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a0, b0)

    o_ptrs = (
        O
        + pid_b * stride_o_b
        + offs_m[:, None] * stride_o_m
        + offs_n[None, :] * stride_o_n
    )
    mask_o = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if OUT_IS_BF16:
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_o)
    elif OUT_IS_FP16:
        tl.store(o_ptrs, acc.to(tl.float16), mask=mask_o)
    else:
        tl.store(o_ptrs, acc, mask=mask_o)


########################################################
# O = W1.transpose(1, 2) @ X1 + W0.transpose(1, 2) @ X0
# W0 and W1 of shape [B, K, M]
# X0 and X1 of shape [B, K, N]
# O of shape [B, M, N]
########################################################
@triton.autotune(configs=get_autotune_configs(), key=["B", "M", "N", "K"])
@triton.jit
def fused_two_mm_wT_x_kernel(
    W0,
    W1,
    X0,
    X1,
    O,
    B,
    M,
    N,
    K,
    # W0 strides: [B, K, M]
    stride_w0_b,
    stride_w0_k,
    stride_w0_m,
    # W1 strides: [B, K, M]
    stride_w1_b,
    stride_w1_k,
    stride_w1_m,
    # X0 strides: [B, K, N]
    stride_x0_b,
    stride_x0_k,
    stride_x0_n,
    # X1 strides: [B, K, N]
    stride_x1_b,
    stride_x1_k,
    stride_x1_n,
    # O strides: [B, M, N]
    stride_o_b,
    stride_o_m,
    stride_o_n,
    OUT_IS_BF16: tl.constexpr,
    OUT_IS_FP16: tl.constexpr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    tl.multiple_of(offs_k, BLOCK_K)
    tl.max_contiguous(offs_m, BLOCK_M)
    tl.max_contiguous(offs_n, BLOCK_N)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        # A tiles: W^T -> treat W as (M,K) via strides (W is stored [K, M])
        a0_ptrs = (
            W0
            + pid_b * stride_w0_b
            + offs_m[:, None] * stride_w0_m
            + k[None, :] * stride_w0_k
        )

        # B tiles: X stored as (K,N)
        b0_ptrs = (
            X0
            + pid_b * stride_x0_b
            + k[:, None] * stride_x0_k
            + offs_n[None, :] * stride_x0_n
        )

        mask_a = (offs_m[:, None] < M) & (k[None, :] < K)
        mask_b = (k[:, None] < K) & (offs_n[None, :] < N)

        a0 = tl.load(a0_ptrs, mask=mask_a, other=0.0)
        b0 = tl.load(b0_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a0, b0)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        # A tiles: W^T -> treat W as (M,K) via strides (W is stored [K, M])

        a0_ptrs = (
            W1
            + pid_b * stride_w1_b
            + offs_m[:, None] * stride_w1_m
            + k[None, :] * stride_w1_k
        )

        b0_ptrs = (
            X1
            + pid_b * stride_x1_b
            + k[:, None] * stride_x1_k
            + offs_n[None, :] * stride_x1_n
        )

        mask_a = (offs_m[:, None] < M) & (k[None, :] < K)
        mask_b = (k[:, None] < K) & (offs_n[None, :] < N)

        a0 = tl.load(a0_ptrs, mask=mask_a, other=0.0)
        b0 = tl.load(b0_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a0, b0)

    o_ptrs = (
        O
        + pid_b * stride_o_b
        + offs_m[:, None] * stride_o_m
        + offs_n[None, :] * stride_o_n
    )
    mask_o = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if OUT_IS_BF16:
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_o)
    elif OUT_IS_FP16:
        tl.store(o_ptrs, acc.to(tl.float16), mask=mask_o)
    else:
        tl.store(o_ptrs, acc, mask=mask_o)


@triton.autotune(configs=get_autotune_configs(), key=["B", "M", "N", "K"])
@triton.jit
def fused_four_mm_wT_x_kernel(
    W0,
    W1,
    W2,
    W3,
    X0,
    X1,
    X2,
    X3,
    O,
    B,
    M,
    N,
    K,
    # W0 strides: [B, K, M]
    stride_w0_b,
    stride_w0_k,
    stride_w0_m,
    # W1 strides: [B, K, M]
    stride_w1_b,
    stride_w1_k,
    stride_w1_m,
    # W2 strides: [B, K, M]
    stride_w2_b,
    stride_w2_k,
    stride_w2_m,
    # W3 strides: [B, K, M]
    stride_w3_b,
    stride_w3_k,
    stride_w3_m,
    # X0 strides: [B, K, N]
    stride_x0_b,
    stride_x0_k,
    stride_x0_n,
    # X1 strides: [B, K, N]
    stride_x1_b,
    stride_x1_k,
    stride_x1_n,
    # X2 strides: [B, K, N]
    stride_x2_b,
    stride_x2_k,
    stride_x2_n,
    # X3 strides: [B, K, N]
    stride_x3_b,
    stride_x3_k,
    stride_x3_n,
    # O strides: [B, M, N]
    stride_o_b,
    stride_o_m,
    stride_o_n,
    OUT_IS_BF16: tl.constexpr,
    OUT_IS_FP16: tl.constexpr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    tl.multiple_of(offs_k, BLOCK_K)
    tl.max_contiguous(offs_m, BLOCK_M)
    tl.max_contiguous(offs_n, BLOCK_N)

    # O = W0.T @ X0
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        # A tiles: W^T -> treat W as (M,K) via strides (W is stored [K, M])
        a0_ptrs = (
            W0
            + pid_b * stride_w0_b
            + offs_m[:, None] * stride_w0_m
            + k[None, :] * stride_w0_k
        )

        # B tiles: X stored as (K,N)
        b0_ptrs = (
            X0
            + pid_b * stride_x0_b
            + k[:, None] * stride_x0_k
            + offs_n[None, :] * stride_x0_n
        )

        mask_a = (offs_m[:, None] < M) & (k[None, :] < K)
        mask_b = (k[:, None] < K) & (offs_n[None, :] < N)

        a0 = tl.load(a0_ptrs, mask=mask_a, other=0.0)
        b0 = tl.load(b0_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a0, b0)

    # O += W1.T @ X1
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        # A tiles: W^T -> treat W as (M,K) via strides (W is stored [K, M])

        a0_ptrs = (
            W1
            + pid_b * stride_w1_b
            + offs_m[:, None] * stride_w1_m
            + k[None, :] * stride_w1_k
        )

        b0_ptrs = (
            X1
            + pid_b * stride_x1_b
            + k[:, None] * stride_x1_k
            + offs_n[None, :] * stride_x1_n
        )

        mask_a = (offs_m[:, None] < M) & (k[None, :] < K)
        mask_b = (k[:, None] < K) & (offs_n[None, :] < N)

        a0 = tl.load(a0_ptrs, mask=mask_a, other=0.0)
        b0 = tl.load(b0_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a0, b0)

    # O += W2.T @ X2
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        # A tiles: W^T -> treat W as (M,K) via strides (W is stored [K, M])
        a0_ptrs = (
            W2
            + pid_b * stride_w2_b
            + offs_m[:, None] * stride_w2_m
            + k[None, :] * stride_w2_k
        )

        # B tiles: X stored as (K,N)
        b0_ptrs = (
            X2
            + pid_b * stride_x2_b
            + k[:, None] * stride_x2_k
            + offs_n[None, :] * stride_x2_n
        )

        mask_a = (offs_m[:, None] < M) & (k[None, :] < K)
        mask_b = (k[:, None] < K) & (offs_n[None, :] < N)

        a0 = tl.load(a0_ptrs, mask=mask_a, other=0.0)
        b0 = tl.load(b0_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a0, b0)

    # O += W3.T @ X3
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        # A tiles: W^T -> treat W as (M,K) via strides (W is stored [K, M])
        a0_ptrs = (
            W3
            + pid_b * stride_w3_b
            + offs_m[:, None] * stride_w3_m
            + k[None, :] * stride_w3_k
        )

        b0_ptrs = (
            X3
            + pid_b * stride_x3_b
            + k[:, None] * stride_x3_k
            + offs_n[None, :] * stride_x3_n
        )

        mask_a = (offs_m[:, None] < M) & (k[None, :] < K)
        mask_b = (k[:, None] < K) & (offs_n[None, :] < N)

        a0 = tl.load(a0_ptrs, mask=mask_a, other=0.0)
        b0 = tl.load(b0_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a0, b0)

    o_ptrs = (
        O
        + pid_b * stride_o_b
        + offs_m[:, None] * stride_o_m
        + offs_n[None, :] * stride_o_n
    )
    mask_o = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if OUT_IS_BF16:
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_o)
    elif OUT_IS_FP16:
        tl.store(o_ptrs, acc.to(tl.float16), mask=mask_o)
    else:
        tl.store(o_ptrs, acc, mask=mask_o)


# -----------------------------
# Python launchers
# -----------------------------
def _out_dtype_flags(t: torch.Tensor):
    return t.dtype == torch.bfloat16, t.dtype == torch.float16


def fused_two_mm_same_out_wT_xT_triton(W0, X0T, W1, X1T):
    """
    W0, W1: [B, K, M]  (bf16/fp16/fp32)
    X0T, X1T: [B, N, K]
    Returns O: [B, M, N] with O = W0^T @ X0T.T + W1^T @ X1T.T
    """
    assert W0.ndim == W1.ndim == X0T.ndim == X1T.ndim == 3
    B, K, M = W0.shape
    Bx, N, Kx = X0T.shape
    assert (Bx == B) and (Kx == K), f"Bx: {Bx}, B: {B}, Kx: {Kx}, K: {K}"
    assert W1.shape == (B, K, M), f"W1.shape: {W1.shape}, B: {B}, K: {K}, M: {M}"
    assert X1T.shape == (B, N, K), f"X1T.shape: {X1T.shape}, B: {B}, N: {N}, K: {K}"
    W0 = W0.contiguous()
    W1 = W1.contiguous()
    X0T = X0T.contiguous()
    X1T = X1T.contiguous()

    device = W0.device
    out = torch.empty((B, M, N), device=device, dtype=X0T.dtype)
    out_is_bf16, out_is_fp16 = _out_dtype_flags(out)

    s_w_b, s_w_k, s_w_m = K * M, M, 1
    s_x_b, s_x_n, s_x_k = K * N, K, 1
    s_o_b, s_o_m, s_o_n = M * N, N, 1

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), B)

    fused_two_mm_wT_xT_kernel[grid](
        W0,
        W1,
        X0T,
        X1T,
        out,
        B,
        M,
        N,
        K,
        s_w_b,
        s_w_k,
        s_w_m,
        s_w_b,
        s_w_k,
        s_w_m,
        s_x_b,
        s_x_n,
        s_x_k,
        s_x_b,
        s_x_n,
        s_x_k,
        s_o_b,
        s_o_m,
        s_o_n,
        out_is_bf16,
        out_is_fp16,
    )
    return out


def fused_two_mm_same_out_wT_x_triton(W0, X0, W1, X1, out=None):
    """
    W0, W1: [B, K, M]
    X0, X1: [B, K, N]
    Returns O: [B, M, N] with O = W0^T @ X0 + W1^T @ X1
    """
    assert W0.ndim == W1.ndim == X0.ndim == X1.ndim == 3
    B, K, M = W0.shape
    Bx, Kx, N = X0.shape
    assert (Bx == B) and (Kx == K)
    assert W1.shape == (B, K, M)
    assert X1.shape == (B, K, N)

    W0 = W0.contiguous()
    W1 = W1.contiguous()
    X0 = X0.contiguous()
    X1 = X1.contiguous()
    if out is not None:
        out = out.contiguous()

    device = W0.device
    if out is None:
        out = torch.empty((B, M, N), device=device, dtype=X0.dtype)
    out_is_bf16, out_is_fp16 = _out_dtype_flags(out)

    # compute strides assuming contigous inputs

    s_w_b, s_w_k, s_w_m = K * M, M, 1
    s_x_b, s_x_k, s_x_n = K * N, N, 1
    s_o_b, s_o_m, s_o_n = M * N, N, 1

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), B)

    fused_two_mm_wT_x_kernel[grid](
        W0,
        W1,
        X0,
        X1,
        out,
        B,
        M,
        N,
        K,
        s_w_b,
        s_w_k,
        s_w_m,
        s_w_b,
        s_w_k,
        s_w_m,
        s_x_b,
        s_x_k,
        s_x_n,
        s_x_b,
        s_x_k,
        s_x_n,
        s_o_b,
        s_o_m,
        s_o_n,
        out_is_bf16,
        out_is_fp16,
    )
    return out


def fused_two_mm_same_out_interface(W0, X0, W1, X1, A_transpose=True, B_transpose=True):
    assert A_transpose, "Only A_transpose=True is implemented"
    if B_transpose:
        return fused_two_mm_same_out_wT_xT_triton(W0, X0, W1, X1)
    else:
        return fused_two_mm_same_out_wT_x_triton(W0, X0, W1, X1)


def fused_four_mm_same_out_interface(
    W0, X0, W1, X1, W2, X2, W3, X3, A_transpose=True, B_transpose=False
):
    assert (
        A_transpose and not B_transpose
    ), "Only A_transpose=True and B_transpose=False is implemented"
    B, K, M = W0.shape
    Bx, Kx, N = X0.shape
    assert (Bx == B) and (Kx == K)
    assert W1.shape == (B, K, M)
    assert X1.shape == (B, K, N)
    assert W2.shape == (B, K, M)
    assert X2.shape == (B, K, N)
    assert W3.shape == (B, K, M)
    assert X3.shape == (B, K, N)

    # assume contiguous inputs
    s_w_b, s_w_k, s_w_m = K * M, M, 1
    s_x_b, s_x_k, s_x_n = K * N, N, 1
    s_o_b, s_o_m, s_o_n = M * N, N, 1

    device = W0.device
    out = torch.empty((B, M, N), device=device, dtype=X0.dtype)
    out_is_bf16, out_is_fp16 = _out_dtype_flags(X0)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), B)

    fused_four_mm_wT_x_kernel[grid](
        W0,
        W1,
        W2,
        W3,
        X0,
        X1,
        X2,
        X3,
        out,
        B,
        M,
        N,
        K,
        s_w_b,
        s_w_k,
        s_w_m,
        s_w_b,
        s_w_k,
        s_w_m,
        s_w_b,
        s_w_k,
        s_w_m,
        s_w_b,
        s_w_k,
        s_w_m,
        s_x_b,
        s_x_k,
        s_x_n,
        s_x_b,
        s_x_k,
        s_x_n,
        s_x_b,
        s_x_k,
        s_x_n,
        s_x_b,
        s_x_k,
        s_x_n,
        s_o_b,
        s_o_m,
        s_o_n,
        out_is_bf16,
        out_is_fp16,
    )
    return out


# -----------------------------
# Correctness checks and benchmark code below
# -----------------------------


def correctness_check_wT_x(
    device="cuda",
):
    from .benchmark import report_error

    device = torch.device("cuda", torch.cuda.current_device())

    def make_inputs(B, M, N, K, dtype=torch.bfloat16):
        return (
            torch.randn(B, K, M, device=device, dtype=dtype),
            torch.randn(B, K, N, device=device, dtype=dtype),
            torch.randn(B, K, M, device=device, dtype=dtype),
            torch.randn(B, K, N, device=device, dtype=dtype),
        )

    def ref_func(W0, X0, W1, X1):
        return two_mm(W0, X0, W1, X1, A_transpose=True, B_transpose=False)

    shape_list = [[1, 512, 512, 512], [2, 2048, 1024, 4096], [1, 678, 724, 996]]
    for B, M, N, K in shape_list:
        _inputs = make_inputs(B, M, N, K)
        O_triton = fused_two_mm_same_out_wT_x_triton(*_inputs)
        fp32_inputs = [_inp.float() for _inp in _inputs]
        O_ref = ref_func(*fp32_inputs)
        report_error(O_ref, O_triton, "O")


def correctness_check_wT_xT():
    """
    Validates fused_two_mm_same_out_wT_xT_triton against a float32 PyTorch reference.
    """
    from .benchmark import report_error

    device = torch.device("cuda", torch.cuda.current_device())

    def make_inputs(B, M, N, K, dtype=torch.bfloat16):
        return (
            torch.randn(B, K, M, device=device, dtype=dtype),
            torch.randn(B, N, K, device=device, dtype=dtype),
            torch.randn(B, K, M, device=device, dtype=dtype),
            torch.randn(B, N, K, device=device, dtype=dtype),
        )

    def ref_func(W0, X0T, W1, X1T):
        return two_mm(W0, X0T, W1, X1T, A_transpose=True, B_transpose=True)

    shape_list = [[1, 512, 512, 512], [2, 2048, 1024, 4096], [1, 678, 724, 996]]
    for B, M, N, K in shape_list:
        _inputs = make_inputs(B, M, N, K)
        O_triton = fused_two_mm_same_out_wT_xT_triton(*_inputs)
        fp32_inputs = [_inp.float() for _inp in _inputs]
        O_ref = ref_func(*fp32_inputs)
        report_error(O_ref, O_triton, "O")


def check_correctness_four_mm():
    """
    Validates fused_four_mm_same_out_interface against a float32 PyTorch reference.
    """
    from .benchmark import report_error

    device = torch.device("cuda", torch.cuda.current_device())

    def make_inputs(B, M, N, K, dtype=torch.bfloat16):
        return (
            torch.randn(B, K, M, device=device, dtype=dtype),
            torch.randn(B, K, N, device=device, dtype=dtype),
            torch.randn(B, K, M, device=device, dtype=dtype),
            torch.randn(B, K, N, device=device, dtype=dtype),
            torch.randn(B, K, M, device=device, dtype=dtype),
            torch.randn(B, K, N, device=device, dtype=dtype),
            torch.randn(B, K, M, device=device, dtype=dtype),
            torch.randn(B, K, N, device=device, dtype=dtype),
        )

    def ref_func(W0, X0, W1, X1, W2, X2, W3, X3):
        O = (
            W0.transpose(1, 2) @ X0
            + W1.transpose(1, 2) @ X1
            + W2.transpose(1, 2) @ X2
            + W3.transpose(1, 2) @ X3
        )
        return O

    shape_list = [[1, 512, 512, 512], [2, 2048, 1024, 4096], [1, 678, 724, 996]]
    for B, M, N, K in shape_list:
        _inputs = make_inputs(B, M, N, K)
        O_triton = fused_four_mm_same_out_interface(*_inputs)
        fp32_inputs = [_inp.float() for _inp in _inputs]
        O_ref = ref_func(*fp32_inputs)
        print(f"B={B}, M={M}, N={N}, K={K} results")
        report_error(O_ref, O_triton, "O")


if __name__ == "__main__":

    correctness_check_wT_x()
    check_correctness_four_mm()
    correctness_check_wT_xT()
