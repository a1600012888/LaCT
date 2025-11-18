"""
Modified from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

BT_LIST = [8, 16, 32, 64, 128]
NUM_WARPS_AUTOTUNE = [1, 2, 4, 8, 16, 32]

triton_dtype_from_torch_dtype = {
    torch.bfloat16: 0,
    torch.float32: 1,
}


#### L2 norm with 2D tiles for small head-dim tensors ####


@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for num_warps in NUM_WARPS_AUTOTUNE
        for BT in BT_LIST
    ],
    key=["D"],
)
@triton.jit
def l2norm_add_fwd_kernel_2d(
    x,  # [B, T, D]
    x_add,
    y,
    tgt_scale,
    rstd,
    eps,
    tgt_dtype: tl.constexpr,
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
):
    # 2D Block tiling:
    # Grid y-axis is T // BT, Grid x-axis is B
    i_b = tl.program_id(0)
    i_t = tl.program_id(1)

    # Adjust pointers for batch offset
    batch_offset = i_b * T * D
    scale_batch_offset = i_b * T

    p_x = tl.make_block_ptr(
        x + batch_offset, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
    )
    p_x_add = tl.make_block_ptr(
        x_add + batch_offset, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
    )
    p_y = tl.make_block_ptr(
        y + batch_offset, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
    )
    p_rstd = tl.make_block_ptr(
        rstd + scale_batch_offset, (T,), (1,), (i_t * BT,), (BT,), (0,)
    )
    p_tgt_scale = tl.make_block_ptr(
        tgt_scale + scale_batch_offset, (T,), (1,), (i_t * BT,), (BT,), (0,)
    )

    # Load and Element-wise Add
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_x_add = tl.load(p_x_add, boundary_check=(0, 1)).to(tl.float32)
    b_tgt_scale = tl.load(p_tgt_scale, boundary_check=(0,)).to(tl.float32)

    b_sum = b_x + b_x_add

    # L2 Norm Calculation
    # Variance = sum(x^2). Rstd = 1/sqrt(var + eps)
    b_var = tl.sum(b_sum * b_sum, 1)
    b_rstd = 1 / tl.sqrt(b_var + eps)

    # Normalize and Scale
    # b_rstd is (BT,), needs broadcasting to (BT, BD) -> b_rstd[:, None]
    b_y = b_sum * b_rstd[:, None] * b_tgt_scale[:, None]

    # Dtype conversion for output
    tgt_dtype_tl = (
        tl.bfloat16
        if tgt_dtype == 0
        else (tl.float32 if tgt_dtype == 1 else tl.float16)
    )

    tl.store(p_y, b_y.to(tgt_dtype_tl), boundary_check=(0, 1))
    tl.store(p_rstd, b_rstd, boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
        for BT in BT_LIST
    ],
    key=["D"],
)
@triton.jit
def l2norm_add_bwd_kernel_2d(
    y,
    rstd,
    tgt_scale,
    # output
    dy,
    dx,
    dx_add,
    dtgt_scale,
    eps,
    x_dtype: tl.constexpr,  # 0 for bf16, 1 for fp32
    x_add_dtype: tl.constexpr,  # 0 for bf16, 1 for fp32
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
):

    x_add_dtype_tl = tl.bfloat16 if x_add_dtype == 0 else tl.float32
    x_dtype_tl = tl.bfloat16 if x_dtype == 0 else tl.float32
    B_stride = T * D
    i_b = tl.program_id(0)
    i_t = tl.program_id(1)

    # pointer for 2D tiles.
    p_y = tl.make_block_ptr(
        y + i_b * B_stride, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
    )
    p_dy = tl.make_block_ptr(
        dy + i_b * B_stride, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
    )
    p_dx = tl.make_block_ptr(
        dx + i_b * B_stride, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
    )
    p_dx_add = tl.make_block_ptr(
        dx_add + i_b * B_stride, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)
    )

    # pointer for 1D scales
    p_rstd = tl.make_block_ptr(rstd + i_b * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    p_tgt_scale = tl.make_block_ptr(
        tgt_scale + i_b * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
    )
    p_dtgt_scale = tl.make_block_ptr(
        dtgt_scale + i_b * T, (T,), (1,), (i_t * BT,), (BT,), (0,)
    )

    # by is normalized y * tgt_scale.
    b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
    b_rstd = tl.load(p_rstd, boundary_check=(0,)).to(tl.float32)
    b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
    b_tgt_scale = tl.load(p_tgt_scale, boundary_check=(0,)).to(tl.float32)

    sum_dy_y_normalized_div_tgt_scale = tl.sum(b_dy * b_y / b_tgt_scale[:, None], 1)
    b_dx = (
        b_dy * b_rstd[:, None] * b_tgt_scale[:, None]
        - sum_dy_y_normalized_div_tgt_scale[:, None] * b_y * b_rstd[:, None]
    )

    tl.store(p_dx, b_dx.to(x_dtype_tl), boundary_check=(0, 1))
    tl.store(p_dx_add, b_dx.to(x_add_dtype_tl), boundary_check=(0, 1))
    tl.store(
        p_dtgt_scale,
        sum_dy_y_normalized_div_tgt_scale.to(p_dtgt_scale.dtype.element_ty),
        boundary_check=(0,),
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D"],
)
@triton.jit
def l2norm_add_fwd_kernel1(
    x,
    x_add,
    y,
    tgt_scale,
    rstd,
    eps,
    tgt_dtype: tl.constexpr,  # 0 for bf16, 1 for fp32
    D,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    x_add += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D
    tgt_dtype_tl = tl.bfloat16 if tgt_dtype == 0 else tl.float32

    tgt_scale_f = tl.load(tgt_scale + i_t).to(tl.float32)

    b_x = tl.load(x + cols, mask=mask, other=0.0)
    orig_dtype = b_x.dtype
    b_x = b_x.to(tl.float32)
    b_x_add = tl.load(x_add + cols, mask=mask, other=0.0).to(tl.float32)
    b_x = b_x + b_x_add

    # Note, eps is inside the sqrt.
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x) + eps)
    b_y = b_x * b_rstd * tgt_scale_f
    tl.store(
        y + cols, b_y.to(tgt_dtype_tl), mask=mask
    )  # save the output to original dtype
    tl.store(rstd + i_t, b_rstd)  # this is float32


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D"],
)
@triton.jit
def l2norm_bwd_kernel1(
    y,
    rstd,
    tgt_scale,
    dy,
    # output
    dx,
    dx_add,
    dtgt_scale,
    eps,
    x_dtype: tl.constexpr,  # 0 for bf16, 1 for fp32
    x_add_dtype: tl.constexpr,  # 0 for bf16, 1 for fp32
    D,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    y += i_t * D
    dx += i_t * D
    dy += i_t * D
    dx_add += i_t * D

    x_add_dtype_tl = tl.bfloat16 if x_add_dtype == 0 else tl.float32
    x_dtype_tl = tl.bfloat16 if x_dtype == 0 else tl.float32

    cols = tl.arange(0, BD)
    mask = cols < D

    b_rstd = tl.load(rstd + i_t).to(tl.float32)
    b_tgt_scale = tl.load(tgt_scale + i_t).to(tl.float32)

    # b_y is the  x / x.norm()
    b_y = tl.load(y + cols, mask=mask, other=0.0)
    b_y = b_y.to(tl.float32)
    b_dy = tl.load(dy + cols, mask=mask, other=0.0).to(tl.float32)

    sum_dy_y_normalized_div_tgt_scale = tl.sum(b_dy * b_y / b_tgt_scale)
    b_dx = (
        b_dy * b_rstd * b_tgt_scale - sum_dy_y_normalized_div_tgt_scale * b_y * b_rstd
    )

    dtgt_scale_f = sum_dy_y_normalized_div_tgt_scale

    tl.store(dtgt_scale + i_t, dtgt_scale_f)
    tl.store(dx + cols, b_dx.to(x_dtype_tl), mask=mask)
    tl.store(dx_add + cols, b_dx.to(x_add_dtype_tl), mask=mask)


def l2_norm_add_fwd(
    x: torch.Tensor,  # [B, D1, D2]
    x_add: torch.Tensor,  # [B, D1, D2]
    tgt_scale: torch.Tensor,  # [B, D1]
    tgt_dtype: torch.dtype = torch.bfloat16,
    eps: float = 1e-5,
):
    y = torch.empty_like(x, dtype=tgt_dtype)
    rstd = torch.empty_like(tgt_scale, dtype=torch.float32)

    B, nD, D = x.shape

    block_size_d = triton.next_power_of_2(D)

    N = B * nD
    tgt_dtype_int_code = triton_dtype_from_torch_dtype[tgt_dtype]

    if D <= 512:
        # use 2D tiles
        def grid(meta):
            return (
                B,
                triton.cdiv(nD, meta["BT"]),
            )

        l2norm_add_fwd_kernel_2d[grid](
            x,
            x_add,
            y,
            tgt_scale,
            rstd,
            eps,
            tgt_dtype_int_code,
            B,
            nD,
            D,
            block_size_d,
        )

    else:
        # use 1D tiles
        grid = (N,)
        l2norm_add_fwd_kernel1[grid](
            x, x_add, y, tgt_scale, rstd, eps, tgt_dtype_int_code, D, block_size_d
        )
    return y, rstd


def l2_norm_bwd(
    dy: torch.Tensor,  # [B, D1, D2]
    y: torch.Tensor,  # [B, D1, D2]
    tgt_scale: torch.Tensor,  # [B, D1]
    rstd: torch.Tensor,  # [B, D1]
    x_dtype,
    x_add_dtype,
    eps: float = 1e-5,
):
    B, nD, D = y.shape

    block_size_d = triton.next_power_of_2(D)

    dx = torch.empty_like(dy, dtype=x_dtype)
    d_x_add = torch.empty_like(dy, dtype=x_add_dtype)
    dtgt_scale = torch.empty_like(tgt_scale)

    x_dtype_int_code = triton_dtype_from_torch_dtype[x_dtype]
    x_add_dtype_int_code = triton_dtype_from_torch_dtype[x_add_dtype]
    N = B * nD

    if D <= 512:

        def grid(meta):
            return (
                B,
                triton.cdiv(nD, meta["BT"]),
            )

        l2norm_add_bwd_kernel_2d[grid](
            y,
            rstd,
            tgt_scale,
            dy,
            dx,
            d_x_add,
            dtgt_scale,
            eps,
            x_dtype_int_code,
            x_add_dtype_int_code,
            B,
            nD,
            D,
            block_size_d,
        )
    else:
        # use 1D tiles
        grid = (N,)
        l2norm_bwd_kernel1[grid](
            y,
            rstd,
            tgt_scale,
            dy,
            dx,
            d_x_add,
            dtgt_scale,
            eps,
            x_dtype_int_code,
            x_add_dtype_int_code,
            D,
            block_size_d,
        )
    return dx, d_x_add, dtgt_scale


class L2NormAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_add, tgt_scale, eps=1e-5, tgt_dtype=None):
        """
        x: [B, D1, D2]
        x_add: [B, D1, D2]
        tgt_scale: [B, D1]
        eps: float
        tgt_dtype: if None, will use the same dtype as x
        """
        ctx.eps = eps
        ctx.x_dtype = x.dtype
        ctx.x_add_dtype = x_add.dtype

        if tgt_dtype is None:
            tgt_dtype = x.dtype

        tgt_scale = tgt_scale.squeeze(dim=-1)

        y, rstd = l2_norm_add_fwd(x, x_add, tgt_scale, tgt_dtype, eps)
        ctx.save_for_backward(y, rstd, tgt_scale)
        return y

    @staticmethod
    def backward(ctx, dy):
        y, rstd, tgt_scale = ctx.saved_tensors
        dx, d_x_add, dtgt_scale = l2_norm_bwd(
            dy.contiguous(), y, tgt_scale, rstd, ctx.x_dtype, ctx.x_add_dtype, ctx.eps
        )

        return dx, d_x_add, dtgt_scale, None, None


def l2_norm_add_fused(x, x_add, tgt_scale, eps=1e-5, tgt_dtype=None):
    """
    x: [B, T, D]
    x_add: [B, T, D]
    tgt_scale: [B, T]
    eps: float
    tgt_dtype: if None, will use the same dtype as x
    """
    return L2NormAddFunction.apply(x, x_add, tgt_scale, eps, tgt_dtype)


def reference_l2_norm_add_fused(
    x: torch.Tensor,  # [B, D1, D2]
    x_add: torch.Tensor,  # [B, D1, D2]
    tgt_scale: torch.Tensor,  # [B, D1]
    eps: float = 1e-5,
):
    y = x.to(torch.float32) + x_add.to(torch.float32)
    rstd = 1 / torch.sqrt(torch.sum(y * y, dim=2, keepdim=True) + eps)
    y = y * rstd * tgt_scale.unsqueeze(dim=-1)

    return y


def make_inputs(B, T, D):
    x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
    x_add = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
    tgt_scale = torch.randn(B, T, device="cuda", dtype=torch.float32)
    return x, x_add, tgt_scale


def check_correctness(B=4, T=2048, D=384):
    from benchmark import report_error

    x, x_add, tgt_scale = make_inputs(B, T, D)
    y = reference_l2_norm_add_fused(x, x_add, tgt_scale)
    y_triton = l2_norm_add_fused(x, x_add, tgt_scale)

    report_error(y, y_triton, "y")


if __name__ == "__main__":
    check_correctness()
