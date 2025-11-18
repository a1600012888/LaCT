# fused_two_mm_swiglu_triton.py
import math
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


########################################################
# W0_W2: [B, 2M, K]
# X: [B, N, K]
# O: [B, M, N]

# This kernel supports dynamic shapes for B and N(sequence length) if you remove B and N from the autotune key.
########################################################


@triton.autotune(
    configs=get_autotune_configs(),
    key=["B", "M", "N", "K"],
)
@triton.jit
def _fused_two_mm_swiglu_kernel(
    W0_W2,
    X,
    O,
    B,
    M: tl.constexpr,
    N,
    K: tl.constexpr,  # mark the reduce axis as constexpr
    stride_w_b,  # = 2M * K
    stride_w_m,  # = K
    stride_w_k,
    stride_x_b,
    stride_x_n,
    stride_x_k,
    stride_o_b,
    stride_o_m,
    stride_o_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 3D launch grid: (m-tiles, n-tiles, batch)
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]

    # Pointers base for this batch
    w0_batch = W0_W2 + pid_b * stride_w_b
    w2_batch = W0_W2 + pid_b * stride_w_b + M * stride_w_m
    x_batch = X + pid_b * stride_x_b
    o_batch = O + pid_b * stride_o_b

    # Accumulators in fp32
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + offs_k  # [BLOCK_K]
        k_mask = k_ids < K

        # Load tiles: W0/W2 shape -> (BLOCK_M, BLOCK_K), X shape -> (BLOCK_N, BLOCK_K)
        w0_ptrs = (
            w0_batch + (offs_m[:, None] * stride_w_m) + (k_ids[None, :] * stride_w_k)
        )
        w2_ptrs = (
            w2_batch + (offs_m[:, None] * stride_w_m) + (k_ids[None, :] * stride_w_k)
        )
        x_ptrs = (
            x_batch + (offs_n[:, None] * stride_x_n) + (k_ids[None, :] * stride_x_k)
        )

        w_mask = (offs_m[:, None] < M) & k_mask[None, :]
        x_mask = (offs_n[:, None] < N) & k_mask[None, :]

        w0 = tl.load(w0_ptrs, mask=w_mask, other=0).to(tl.bfloat16)
        w2 = tl.load(w2_ptrs, mask=w_mask, other=0).to(tl.bfloat16)
        x = tl.load(x_ptrs, mask=x_mask, other=0).to(tl.bfloat16)  # (BLOCK_N, BLOCK_K)

        # (M,K) x (K,N): we trans(x) to (BLOCK_K, BLOCK_N)
        acc0 += tl.dot(w0, tl.trans(x), out_dtype=tl.float32)
        acc2 += tl.dot(w2, tl.trans(x), out_dtype=tl.float32)

    # Apply SiLU in fp32 and fuse multiply
    y0 = acc0  # fp32
    y2 = acc2  # fp32
    # SiLU(x) = x * sigmoid(x)
    out = y2 * (y0 * tl.sigmoid(y0))

    # Store to bf16
    o_ptrs = o_batch + (offs_m[:, None] * stride_o_m) + (offs_n[None, :] * stride_o_n)
    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=o_mask)


def fused_two_mm_swiglu_triton(
    W0_W2: torch.Tensor,
    X: torch.Tensor,
):
    """
    Wraps the Triton kernel. Shapes:
      W0, W2: [B, M, K]  (bf16)
      X     : [B, N, K]  (bf16)
      returns O: [B, M, N] (bf16) where O = SiLU(W0 @ X^T) * (W2 @ X^T)
    """
    assert W0_W2.dtype == torch.bfloat16 and X.dtype == torch.bfloat16
    assert W0_W2.is_contiguous() and X.is_contiguous(), "W0_W2 and X must be contiguous"

    B, M_times_2, K = W0_W2.shape
    Bx, N, Kx = X.shape
    assert Bx == B and Kx == K, "X must be [B, N, K] with matching B,K."

    M = M_times_2 // 2

    O = torch.empty((B, M, N), device=X.device, dtype=torch.bfloat16)

    # Strides (PyTorch: element strides)
    stride_w_b, stride_w_m, stride_w_k = W0_W2.stride()
    stride_x_b, stride_x_n, stride_x_k = X.stride()
    stride_o_b, stride_o_m, stride_o_n = O.stride()

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), B)

    _fused_two_mm_swiglu_kernel[grid](
        W0_W2,
        X,
        O,
        B,
        M,
        N,
        K,
        stride_w_b,
        stride_w_m,
        stride_w_k,
        stride_x_b,
        stride_x_n,
        stride_x_k,
        stride_o_b,
        stride_o_m,
        stride_o_n,
        # BLOCK_M=BLOCK_M,
        # BLOCK_N=BLOCK_N,
        # BLOCK_K=BLOCK_K,
        # num_warps=num_warps,
        # num_stages=num_stages,
    )
    return O


# -------------------------
# Test harness vs PyTorch
# -------------------------
@torch.compile
def _reference_pytorch(W0_W2, X):
    # Compute in fp32 and cast to bf16 to match kernel's final cast
    W0, W2 = W0_W2.chunk(2, dim=1)
    Y0 = torch.matmul(W0, X.transpose(-1, -2))  # [B, M, N]
    Y2 = torch.matmul(W2, X.transpose(-1, -2))  # [B, M, N]
    O = torch.nn.functional.silu(Y0) * Y2
    return O


def make_inputs_ffn(B, M, K, N, require_grad=True):
    W0_W2 = torch.randn(
        B, 2 * M, K, device="cuda", dtype=torch.bfloat16, requires_grad=require_grad
    )
    K_input = torch.randn(
        B, N, K, device="cuda", dtype=torch.bfloat16, requires_grad=require_grad
    )

    return W0_W2, K_input


def check_correctness():
    from .benchmark import report_error

    torch.manual_seed(0)
    device = "cuda"

    # Example sizes; feel free to change
    B, H, D, L = 4, 1536, 3072, 16384

    inputs = make_inputs_ffn(B, H, D, L)

    # Triton
    O_triton = fused_two_mm_swiglu_triton(*inputs)

    # Reference in fp32
    fp32_inputs = [_.to(torch.float32) for _ in inputs]
    O_ref = _reference_pytorch(*fp32_inputs)

    report_error(O_ref, O_triton, "triton_swiglu_kernel")


if __name__ == "__main__":
    check_correctness()
