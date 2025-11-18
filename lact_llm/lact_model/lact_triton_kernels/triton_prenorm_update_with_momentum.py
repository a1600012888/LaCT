import torch
import torch.nn.functional as F
import triton
import triton.language as tl

NUM_WARPS_AUTOTUNE = [1, 2, 4, 8, 16, 32]


#### L2 norm with momentum update used in Test-Time Training
def reference_l2_norm_add_fused_with_momentum(
    x: torch.Tensor,  # [B, D1, D2], fp32 recommeded
    dx: torch.Tensor,
    dx_momentum: torch.Tensor,  # [B, D1, D2]
    momentum_mu: torch.Tensor,  # [B], fp32
    tgt_scale: torch.Tensor,  # [B, D1]
    eps: float = 1e-5,
):
    """
    Args:
        x: [B, D1, D2], must be fp32
        dx: [B, D1, D2], bf16
        dx_momentum: [B, D1, 1], fp32 recommended
        momentum_mu: float
        tgt_scale: [B, D1], fp32
        eps: float
    Returns:
        x_updated: same as x, recommended to be fp32
        m: [B, D1, D2], fp32 recommended, depends on dtype of dx_momentum
        x_updated_normalized: [B, D1, D2], bf16
    """
    m = dx + dx_momentum * momentum_mu.unsqueeze(dim=-1).unsqueeze(
        dim=-1
    )  # this should be fp32
    x_updated = x + m  # this must be fp32

    x_updated_sq_sum = torch.sqrt(torch.sum(x_updated**2, dim=-1, keepdim=True) + eps)
    x_updated_normalized = x_updated / x_updated_sq_sum * tgt_scale.unsqueeze(dim=-1)
    return x_updated, m, x_updated_normalized.to(torch.bfloat16)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D"],
)
@triton.jit
def prenorm_update_with_momentum_and_l2_norm_kernel(
    x_main_ptr,  # [B, n, D2], fp32
    dx_ptr,  # [B, n, D2], bf16
    dx_momentum_ptr,  # [B, n, D2], fp32
    momentum_mu_ptr,  # [B], fp32
    tgt_scale_ptr,  # [B, n], fp32
    eps,  # float
    # output
    y_main_ptr,  # [B, n, D2], same dtype as x_main
    updated_momentum_ptr,  # [B, n, D2], fp32
    y_normalized_ptr,  # [B, n, D2], bf16
    rstd_ptr,  # [B, n], fp32
    n,
    D,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_b = i_t // n
    # if output_dtype == 0:
    #     y_normalized_save_dtype = tl.bfloat16
    # else:
    #     y_normalized_save_dtype = tl.float32

    ### Step 1: update dx and dx_momentum
    x_main_ptr += i_t * D
    dx_ptr += i_t * D
    dx_momentum_ptr += i_t * D

    y_main_ptr += i_t * D
    updated_momentum_ptr += i_t * D
    y_normalized_ptr += i_t * D
    cols = tl.arange(0, BD)
    mask = cols < D

    tgt_scale_f = tl.load(tgt_scale_ptr + i_t).to(tl.float32)

    x_main = tl.load(x_main_ptr + cols, mask=mask, other=0.0)
    x_main_dtype = x_main.dtype
    x_main = x_main.to(tl.float32)

    dx = tl.load(dx_ptr + cols, mask=mask, other=0.0)
    y_normalized_save_dtype = dx.dtype
    dx = dx.to(tl.float32)

    dx_momentum = tl.load(dx_momentum_ptr + cols, mask=mask, other=0.0)
    momentum_dtype = dx_momentum.dtype
    dx_momentum = dx_momentum.to(tl.float32)

    momentum_mu = tl.load(momentum_mu_ptr + i_b).to(tl.float32)

    ## Step 2: update dx_momentum
    updated_momentum = dx_momentum * momentum_mu + dx

    ## Step 3: update x
    x_main = x_main + updated_momentum

    ## Step 4: L2_norm

    # \sqrt(\sum_{i=0}^{D2} x_i^2 + eps)
    rstd = 1 / (tl.sqrt(tl.sum(x_main * x_main) + eps))
    y_normalized = x_main * rstd * tgt_scale_f

    ### Step-5: cast dtype and save output

    tl.store(y_main_ptr + cols, x_main.to(x_main_dtype), mask=mask)
    tl.store(rstd_ptr + i_t, rstd.to(tl.float32))  # this is float32
    tl.store(
        y_normalized_ptr + cols, y_normalized.to(y_normalized_save_dtype), mask=mask
    )
    tl.store(
        updated_momentum_ptr + cols, updated_momentum.to(momentum_dtype), mask=mask
    )


def prenorm_update_with_momentum_and_l2_norm(
    x_main: torch.Tensor,  # [B, n, D2], fp32
    dx: torch.Tensor,  # [B, n, D2], bf16
    dx_momentum: torch.Tensor,  # [B, n, D2], fp32
    momentum_mu: torch.Tensor,  # [B], fp32
    tgt_scale: torch.Tensor,  # [B, n], fp32
    eps: float = 1e-5,
):
    """
    Args:
        x: [B, n, D2], fp32
        dx: [B, n, D2], bf16
        dx_momentum: [B, n, D2], fp32
        momentum_mu: [B], fp32
        tgt_scale: [B, n], fp32
        eps: float
    Returns:
        y: [B, n, D2], fp32
        updated_momentum: [B, n, D2], fp32
        y_normalized: [B, n, D2], bf16
        rstd: [B, n], fp32
    """
    y_main = torch.empty_like(x_main)

    updated_momentum = torch.empty_like(dx_momentum)

    y_normalized = torch.empty_like(dx)
    rstd = torch.empty_like(tgt_scale)

    B, n, D = x_main.shape

    block_size_d = triton.next_power_of_2(D)

    grid = (B * n,)
    prenorm_update_with_momentum_and_l2_norm_kernel[grid](
        x_main,
        dx,
        dx_momentum,
        momentum_mu,
        tgt_scale,
        eps,
        y_main,
        updated_momentum,
        y_normalized,
        rstd,
        n,
        D,
        block_size_d,
    )
    return y_main, updated_momentum, y_normalized, rstd


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D"],
)
@triton.jit
def prenorm_update_with_momentum_and_l2_norm_backward_kernel(
    y_normalized_ptr,  # [B, n, D2], fp32
    dx_momentum_ptr,  # [B, n, D2], fp32
    momentum_mu_ptr,  # [B], fp32
    rstd_ptr,  # [B, n], fp32
    tgt_scale_ptr,  # [B, n], fp32
    dy_main_ptr,  # [B, n, D2], fp32
    dy_momentum_ptr,  # [B, n, D2], fp32
    dy_normalized_ptr,  # [B, n, D2], bf16
    # output
    grad_x_main_ptr,  # [B, n, D2], fp32
    grad_dx_ptr,  # [B, n, D2], fp32
    grad_dx_momentum_ptr,  # [B, n, D2], fp32
    grad_mu_ptr,  # [B], fp32
    grad_target_scale_ptr,  # [B, n], fp32
    n,
    D,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_b = i_t // n

    y_normalized_ptr += i_t * D
    dx_momentum_ptr += i_t * D

    # gradients
    dy_main_ptr += i_t * D
    dy_momentum_ptr += i_t * D
    dy_normalized_ptr += i_t * D

    # for output
    grad_x_main_ptr += i_t * D
    grad_dx_ptr += i_t * D
    grad_dx_momentum_ptr += i_t * D
    grad_target_scale_ptr += i_t

    cols = tl.arange(0, BD)
    mask = cols < D

    b_rstd = tl.load(rstd_ptr + i_t).to(tl.float32)
    b_tgt_scale = tl.load(tgt_scale_ptr + i_t)
    tgt_scale_dtype_tl = b_tgt_scale.dtype
    b_tgt_scale = b_tgt_scale.to(tl.float32)

    # b_y is the  x / x.norm()
    y_normalized = tl.load(y_normalized_ptr + cols, mask=mask, other=0.0)
    normalized_dtype = y_normalized.dtype
    y_normalized = y_normalized.to(tl.float32)

    # Now, backward pass through the final L2 Normalization

    dy_normalized = tl.load(dy_normalized_ptr + cols, mask=mask, other=0.0).to(
        tl.float32
    )

    sum_dy_y_normalized_div_tgt_scale = tl.sum(
        dy_normalized * y_normalized / b_tgt_scale
    )
    grad_y_before_l2_norm = (
        dy_normalized * b_rstd * b_tgt_scale
        - sum_dy_y_normalized_div_tgt_scale * y_normalized * b_rstd
    )

    dtgt_scale_f = sum_dy_y_normalized_div_tgt_scale

    dy_main = tl.load(dy_main_ptr + cols, mask=mask, other=0.0)
    y_main_dtype = dy_main.dtype
    dy_main = dy_main.to(tl.float32)

    grad_x_main = dy_main + grad_y_before_l2_norm

    tl.store(grad_target_scale_ptr, dtgt_scale_f.to(tgt_scale_dtype_tl))
    # in fwd pass: x_main_updated = x_main + updated_momentum
    # thus, grad_dx = grad_x_main.to(dx_dtype)
    tl.store(grad_x_main_ptr + cols, grad_x_main.to(y_main_dtype), mask=mask)

    # Now, backward pass through the momentum update
    # in fwd pass: updated_momentum = momentum_mu * momentum + dx
    b_mu = tl.load(momentum_mu_ptr + i_b)
    mu_dtype_tl = b_mu.dtype
    b_mu = b_mu.to(tl.float32)

    dy_updated_momentum = tl.load(dy_momentum_ptr + cols, mask=mask, other=0.0)
    momentum_dtype = dy_updated_momentum.dtype
    dy_updated_momentum = dy_updated_momentum.to(tl.float32) + grad_x_main
    tl.store(grad_dx_ptr + cols, dy_updated_momentum.to(normalized_dtype), mask=mask)

    grad_momentum = dy_updated_momentum * b_mu
    tl.store(grad_dx_momentum_ptr + cols, grad_momentum.to(momentum_dtype), mask=mask)

    b_momentum = tl.load(dx_momentum_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    grad_mu = tl.sum(dy_updated_momentum * b_momentum, axis=0).to(
        tl.float32
    )  # a scalar.
    # tl.atomic_add(grad_mu_ptr + i_b, grad_mu)
    # Let's try not using atomic_add
    tl.store(grad_mu_ptr + i_t, grad_mu)


def prenorm_update_with_momentum_and_l2_norm_backward(
    y_normalized: torch.Tensor,  # [B, n, D2], fp32
    dx_momentum: torch.Tensor,  # [B, n, D2], fp32
    momentum_mu: torch.Tensor,  # [B], fp32
    rstd: torch.Tensor,  # [B, n], fp32
    tgt_scale: torch.Tensor,  # [B, n], fp32
    dy_main: torch.Tensor,  # [B, n, D2], fp32
    dy_momentum: torch.Tensor,  # [B, n, D2], fp32
    dy_normalized: torch.Tensor,  # [B, n, D2], bf16
):
    B, n, D = dy_main.shape
    grad_x_main = torch.empty_like(dy_main)

    grad_dx = torch.empty_like(dy_normalized)

    grad_dx_momentum = torch.empty_like(dx_momentum)

    grad_mu = momentum_mu.new_zeros(B, n).to(torch.float32)

    grad_tgt_scale = torch.empty_like(tgt_scale)

    B, n, D = dy_main.shape

    block_size_d = triton.next_power_of_2(D)

    grid = (B * n,)

    prenorm_update_with_momentum_and_l2_norm_backward_kernel[grid](
        y_normalized,
        dx_momentum,
        momentum_mu,
        rstd,
        tgt_scale,
        dy_main,
        dy_momentum,
        dy_normalized,
        grad_x_main,
        grad_dx,
        grad_dx_momentum,
        grad_mu,
        grad_tgt_scale,
        n,
        D,
        block_size_d,
    )
    return (
        grad_x_main,
        grad_dx,
        grad_dx_momentum,
        grad_mu.sum(dim=1).to(momentum_mu.dtype),
        grad_tgt_scale,
    )


class PrenormUpdateWithMomentumAndL2NormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_main, dx, dx_momentum, momentum_mu, tgt_scale, eps=1e-5):

        assert tgt_scale.ndim == 2
        ctx.x_main_dtype = x_main.dtype

        y_main, updated_momentum, y_normalized, rstd = (
            prenorm_update_with_momentum_and_l2_norm(
                x_main, dx, dx_momentum, momentum_mu, tgt_scale, eps
            )
        )
        ctx.save_for_backward(y_normalized, dx_momentum, momentum_mu, tgt_scale, rstd)
        return y_main, updated_momentum, y_normalized

    @staticmethod
    def backward(ctx, dy_main, dy_momentum, dy_normalized):
        y_normalized, dx_momentum, momentum_mu, tgt_scale, rstd = ctx.saved_tensors

        dy_main = dy_main.contiguous()
        dy_momentum = dy_momentum.contiguous()
        dy_normalized = dy_normalized.contiguous()

        grad_x_main, grad_dx, grad_dx_momentum, grad_mu, grad_tgt_scale = (
            prenorm_update_with_momentum_and_l2_norm_backward(
                y_normalized,
                dx_momentum,
                momentum_mu,
                rstd,
                tgt_scale,
                dy_main,
                dy_momentum,
                dy_normalized,
            )
        )
        return (
            grad_x_main,
            grad_dx,
            grad_dx_momentum,
            grad_mu,
            grad_tgt_scale,
            None,
        )


def fused_prenorm_update_with_momentum_and_l2_norm(
    x_main: torch.Tensor,
    dx: torch.Tensor,
    dx_momentum: torch.Tensor,
    momentum_mu: torch.Tensor,
    tgt_scale: torch.Tensor,
    eps: float = 1e-5,
):
    """
    x_main: [B, n, D2], fp32 or bf16, recommended to be fp32
    dx: [B, n, D2], bf16
    dx_momentum: [B, n, D2], same dtype as x_main
    momentum_mu: [B], fp32
    tgt_scale: [B, n], fp32
    eps: float
    Returns:
        y_main: [B, n, D2], same dtype as x_main
        updated_momentum: [B, n, D2], fp32
        y_normalized: [B, n, D2], bf16
    """
    return PrenormUpdateWithMomentumAndL2NormFunction.apply(
        x_main, dx, dx_momentum, momentum_mu, tgt_scale, eps
    )


def test_correctness_of_prenorm_update_with_momentum_and_l2_norm():

    def make_inputs(
        B,
        n,
        D,
        x_main_dtype=torch.float32,
        dx_dtype=torch.bfloat16,
        dx_momentum_dtype=torch.float32,
        momentum_mu_dtype=torch.float32,
        tgt_scale_dtype=torch.float32,
        requires_grad=False,
        eps=1e-5,
    ):
        x_main = torch.randn(
            B, n, D, device="cuda", dtype=x_main_dtype, requires_grad=requires_grad
        )
        dx = torch.randn(
            B, n, D, device="cuda", dtype=dx_dtype, requires_grad=requires_grad
        )
        dx_momentum = torch.randn(
            B, n, D, device="cuda", dtype=dx_momentum_dtype, requires_grad=requires_grad
        )
        momentum_mu = torch.randn(
            B, device="cuda", dtype=momentum_mu_dtype, requires_grad=requires_grad
        )
        tgt_scale = torch.randn(
            B, n, device="cuda", dtype=tgt_scale_dtype, requires_grad=False
        )
        tgt_scale = torch.abs(tgt_scale) + 1e-5
        tgt_scale = tgt_scale.requires_grad_(requires_grad)
        return x_main, dx, dx_momentum, momentum_mu, tgt_scale, eps

    B, n, D = 16, 1024, 768
    _inputs = make_inputs(
        B,
        n,
        D,
        x_main_dtype=torch.float32,
        dx_dtype=torch.bfloat16,
        dx_momentum_dtype=torch.float32,
        momentum_mu_dtype=torch.float32,
        tgt_scale_dtype=torch.float32,
        eps=1e-5,
    )

    y_main, updated_momentum, y_normalized = (
        PrenormUpdateWithMomentumAndL2NormFunction.apply(*_inputs)
    )

    ref_y_main, ref_updated_momentum, ref_y_normalized = (
        reference_l2_norm_add_fused_with_momentum(*_inputs)
    )

    from benchmark import report_error

    report_error(ref_y_main, y_main, "y_main")
    report_error(ref_updated_momentum, updated_momentum, "updated_momentum")
    report_error(ref_y_normalized, y_normalized, "y_normalized")
    print("=> Done testing correctness")

    ## bwd
    x_main, dx, dx_momentum, momentum_mu, tgt_scale, eps = make_inputs(
        B,
        n,
        D,
        x_main_dtype=torch.float32,
        dx_dtype=torch.bfloat16,
        dx_momentum_dtype=torch.float32,
        momentum_mu_dtype=torch.float32,
        tgt_scale_dtype=torch.float32,
        requires_grad=True,
    )
    y_main, updated_momentum, y_normalized = (
        PrenormUpdateWithMomentumAndL2NormFunction.apply(
            x_main, dx, dx_momentum, momentum_mu, tgt_scale, eps
        )
    )

    loss = y_main.sum() + updated_momentum.sum() + y_normalized.sum()
    loss.backward()

    x_main_grad = x_main.grad.clone().detach()
    dx_grad = dx.grad.clone().detach()
    dx_momentum_grad = dx_momentum.grad.clone().detach()
    momentum_mu_grad = momentum_mu.grad.clone().detach()
    tgt_scale_grad = tgt_scale.grad.clone().detach()

    # clean all grads
    x_main.grad.zero_()
    dx.grad.zero_()
    dx_momentum.grad.zero_()
    momentum_mu.grad.zero_()
    tgt_scale.grad.zero_()
    x_main = x_main.detach().clone().requires_grad_(True)
    dx = dx.detach().clone().requires_grad_(True)
    dx_momentum = dx_momentum.detach().clone().requires_grad_(True)
    momentum_mu = momentum_mu.detach().clone().requires_grad_(True)
    tgt_scale = tgt_scale.detach().clone().requires_grad_(True)

    ref_y_main, ref_updated_momentum, ref_y_normalized = (
        reference_l2_norm_add_fused_with_momentum(
            x_main, dx, dx_momentum, momentum_mu, tgt_scale, eps
        )
    )
    loss = ref_y_main.sum() + ref_updated_momentum.sum() + ref_y_normalized.sum()
    loss.backward()

    ref_x_main_grad = x_main.grad.clone().detach()
    ref_dx_grad = dx.grad.clone().detach()
    ref_dx_momentum_grad = dx_momentum.grad.clone().detach()
    ref_momentum_mu_grad = momentum_mu.grad.clone().detach()
    ref_tgt_scale_grad = tgt_scale.grad.clone().detach()

    report_error(x_main_grad, ref_x_main_grad, "x_main_grad")
    report_error(dx_grad, ref_dx_grad, "dx_grad")
    report_error(dx_momentum_grad, ref_dx_momentum_grad, "dx_momentum_grad")
    report_error(momentum_mu_grad, ref_momentum_mu_grad, "momentum_mu_grad")
    report_error(tgt_scale_grad, ref_tgt_scale_grad, "tgt_scale_grad")
    print("=> Done testing correctness")


if __name__ == "__main__":
    test_correctness_of_prenorm_update_with_momentum_and_l2_norm()
