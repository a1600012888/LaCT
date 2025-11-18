import torch

from torch.autograd.function import once_differentiable

try:
    from triton_swiglu_bwd_with_lr import swiglu_backward_three_bmm_with_lr_triton

    from triton_fused_matmul_kernels import (
        fused_two_mm_same_out_interface,
    )
    from triton_pointwise_kernels import triton_swiglu_bwd_bwd_fused_cat_inp_out
except ImportError:
    from .triton_swiglu_bwd_with_lr import swiglu_backward_three_bmm_with_lr_triton

    from .triton_fused_matmul_kernels import (
        fused_two_mm_same_out_interface,
    )
    from .triton_pointwise_kernels import triton_swiglu_bwd_bwd_fused_cat_inp_out


class FusedLactSwiGLUFFNBwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, W0_W2, W1, K, V, lr0, lr1, lr2):
        """
        Args:
            W0_W2:    [B, 2M, K] or [B, 2 * Hidden, D]
            W1:        [B, K, M] or [B, D, Hidden]
            K, V:      [M, N, K] or [B, num_Tokens, D]
            lr0, lr1, lr2:    [B, N]

        Outs:
            Hidden: [B, N, K] or [B, num_tokens, Hidden]
            dW0_W2: [B, 2 * Hidden, D]
            dW1: [B, D, Hidden]
        Total FLOPS: 12 * B * Hidden * D * num_tokens
        """
        #### without this triton kernel, we will materize Y0, Y2, Dhidden;  DY0_with_LR0, DY2_with_LR2, Hidden_with_LR1;
        #### 3 + 3 + 3.   read, write.
        DY0_DY2, Hidden = swiglu_backward_three_bmm_with_lr_triton(
            W0_W2,
            W1,
            K,
            V,
            lr0,
            lr1,
            lr2,
        )

        # groupping below two GEMM togeather can futher reduce launching overhead.
        # [B, 2 * Hidden, num_tokens] @ [B, num_tokens, D] -> [B, 2 * Hidden, D]
        DW0_DW2 = torch.bmm(DY0_DY2, K)
        # [B, D, Hidden] = [B, D, num_tokens] @ [B, num_tokens, Hidden]
        DW1 = torch.bmm(V.transpose(1, 2), Hidden.transpose(1, 2))

        # we don't need to save DY0, DY2, and Hidden, because we will compute them again in the backward pass.
        ctx.save_for_backward(W0_W2, W1, K, V, lr0, lr1, lr2)

        return DW0_DW2, DW1

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dw0_dw2, grad_dw1):
        """
        Args:
            grad_dw0_dw2: [B, 2 * Hidden, D]
            grad_dw1: [B, D, Hidden]
        Outs:
            grad_W0: [B, Hidden, D]
            grad_W1: [B, D, Hidden]
            grad_W2: [B, Hidden, D]
            grad_K: [B, D, num_tokens]
            grad_V: [B, num_tokens, D]
            grad_lr0: [B, 1]
            grad_lr1: [B, 1]
            grad_lr2: [B, 1]

        Total FLOPS: 24 * B * Hidden * D * num_tokens + 6 * B * Hidden * D * num_tokens
        # 24 for backward matmuls, and 6 for forward recomputation.
        """

        W0_W2, W1, K, V, lr0, lr1, lr2 = ctx.saved_tensors

        # -> [B, 2 * Hidden, num_tokens]
        Y0_Y2 = torch.bmm(W0_W2, K.transpose(1, 2))

        DHidden = torch.bmm(W1.transpose(1, 2), V.transpose(1, 2))
        grad_Hidden_with_lr1 = torch.bmm(grad_dw1.transpose(1, 2), V.transpose(1, 2))

        # [B, Hidden, num_tokens] = [B, Hidden, D] @ [B, num_tokens, D].T
        # grad_DY0_with_lr0, grad_DY2_with_lr2 = torch.ops.lact.two_mm_same_inp(
        #     grad_dw0.contiguous(), grad_dw2.contiguous(), K.contiguous(), False, True
        # )

        # [B, 2 * Hidden, D]
        grad_DY0_with_lr0_and_grad_DY2_with_lr2 = torch.bmm(
            grad_dw0_dw2, K.transpose(1, 2)
        )

        #### Next, we do tones of element-wise ops.
        #### These element-wise ops are compiled with torch.compile. one graph?
        (
            grad_DHidden,  # [B, Hidden, num_tokens]
            grad_Y0_Y2,  # [B, 2 * Hidden, num_tokens]
            grad_lr0,  # [B, L]
            grad_lr1,  # [B, L]
            grad_lr2,  # [B, L]
            DY0_with_lr0_and_DY2_with_lr2,  # [B, 2 * Hidden, L]
            Hidden_with_lr1,  # [B, Hidden, L]
        ) = triton_swiglu_bwd_bwd_fused_cat_inp_out(
            # ) = pytorch_swiglu_bwd_bwd_fused_cat_inp_out(
            DHidden,  # [B, Hidden, num_tokens]
            Y0_Y2,  # [B, 2 * Hidden, num_tokens]
            lr0,  # [B, L]
            lr1,  # [B, L]
            lr2,  # [B, L]
            grad_DY0_with_lr0_and_grad_DY2_with_lr2,  # [B, 2 * Hidden, num_tokens]
            grad_Hidden_with_lr1,  # [B, Hidden, num_tokens]
        )

        grad_K = fused_two_mm_same_out_interface(
            DY0_with_lr0_and_DY2_with_lr2,  # [B, 2 * Hidden, num_tokens]
            grad_dw0_dw2.contiguous(),  # [B, 2 * Hidden, D]
            grad_Y0_Y2,
            W0_W2,
            A_transpose=True,
            B_transpose=False,
        )

        # grad_V = two_mm_same_out_interface_v2(
        grad_V = fused_two_mm_same_out_interface(
            grad_DHidden,
            W1,
            Hidden_with_lr1,
            grad_dw1.contiguous(),
            A_transpose=True,
            B_transpose=True,
        )

        #### For below three matmuls, occupancy is the key, cause their dimension might be small.

        # [B, D, num_tokens] @ [B, num_tokens, Hidden].T -> [B, D, Hidden]
        grad_W1 = torch.bmm(V.transpose(1, 2), grad_DHidden.transpose(1, 2))

        # [B, 2 * Hidden, D] @ [B, D, num_tokens] -> [B, 2 * Hidden, num_tokens]
        grad_W0_W2 = torch.bmm(grad_Y0_Y2, K)

        return (
            grad_W0_W2,
            grad_W1,
            grad_K,
            grad_V,
            grad_lr0,
            grad_lr1,
            grad_lr2,
        )


def fused_lact_swiglu_ffn_fast_weight_grads(W0_W2, W1, K, V, lr0, lr1, lr2):
    """
    Args:
        W0_W2:    [B, 2 * Hidden, D]
        W1:       [B, D, Hidden]
        K, V:     [B, num_Tokens, D]
        lr0, lr1, lr2:    [B, N]

    Outs:
        Hidden: [B, num_tokens, Hidden]
        dW0_W2: [B, 2 * Hidden, D]
        dW1: [B, D, Hidden]
    Total FLOPS: 12 * B * Hidden * D * num_tokens
    This function computes the test-time training gradients for W0, W1, W2.
    The pytorch reference implementation is shwon in reference_lact_swiglu_ffn_fast_weight_grads, implemented below
    """
    return FusedLactSwiGLUFFNBwd.apply(W0_W2, W1, K, V, lr0, lr1, lr2)


@torch.compile()
def pytorch_swiglu_bwd_bwd_fused_cat_inp_out(
    dh: torch.Tensor,  # [b, d, l]
    x0_x2: torch.Tensor,  # [b, 2* d, l]
    lr0: torch.Tensor,  # [b, l]
    lr1: torch.Tensor,  # [b, l]
    lr2: torch.Tensor,  # [b, l]
    grad_dx0_dx2: torch.Tensor,  # [b, 2 * d, l]
    grad_hidden_lr1: torch.Tensor,  # [b, d, l]
):
    """
    In previous fwd pass:
    dx0 = lr0 * dh * x2 * sigma * (1 + x0 * (1 - sigma))
    dx2 = lr2 * dh * silu(x0)
    hidden_lr1 = lr1 * x2 * silu(x0)

    In this backward pass:
    grad_dh = grad_dx0 * lr0 * x2 * sigma * (1 + x0 * (1 - sigma)) + grad_dx2 * lr2 * silu(x0)

    grad_x2 = grad_dx0 * lr0 * dh * sigma * (1 + x0 * (1 - sigma)) + grad_hidden_lr1 * lr1 * sigma * x0
    # for grad_x0, a little bit tricky,
    - grad_sigma = grad_dx0 * lr0 * dh * x2 * (1 + x0 - 2 sigma * x0)
    - grad_x0_naive  = grad_dx2 * lr2 * dh * sigma * (1 + x0 * (1 - sigma)) +  grad_dx0 * lr0 * dh * x2 * sigma * (1 - sigma) + grad_hidden_lr1 * lr1 * x2 * dsilu_x0_multiplier
    grad_x0 = grad_x0_naive + grad_sigma * sigma * (1 - sigma)

    # then sum of the last dimension (the d dimension!)
    grad_lr0 = grad_dx0 * dh * x2 * sigma * (1 + x0 * (1 - sigma)) # need to sum over all the d of the same l
    grad_lr2 = grad_dx2 * dh * silu(x0)
    grad_lr1 = grad_hidden_lr1 * x2 * sigma * x0

    """
    lr0 = lr0.unsqueeze(dim=1)
    lr1 = lr1.unsqueeze(dim=1)
    lr2 = lr2.unsqueeze(dim=1)

    x0, x2 = x0_x2.chunk(2, dim=1)
    grad_dx0, grad_dx2 = grad_dx0_dx2.chunk(2, dim=1)

    sigma = torch.sigmoid(x0)
    silu_x0 = torch.nn.functional.silu(x0)
    silu_bp_multiplier = sigma * (1 + x0 * (1 - sigma))
    grad_dh = grad_dx0 * lr0 * x2 * silu_bp_multiplier + grad_dx2 * lr2 * silu_x0
    grad_x2 = grad_dx0 * lr0 * dh * silu_bp_multiplier + grad_hidden_lr1 * lr1 * silu_x0

    grad_sigma = grad_dx0 * lr0 * dh * x2 * (1 + x0 - 2 * sigma * x0)
    grad_x0_naive = (
        grad_dx2 * lr2 * dh + grad_hidden_lr1 * lr1 * x2
    ) * silu_bp_multiplier + grad_dx0 * lr0 * dh * x2 * sigma * (1 - sigma)
    grad_x0 = grad_x0_naive + grad_sigma * sigma * (1 - sigma)
    grad_lr0 = grad_dx0 * dh * x2 * silu_bp_multiplier
    grad_lr1 = grad_hidden_lr1 * x2 * silu_x0
    grad_lr2 = grad_dx2 * dh * silu_x0

    grad_lr0 = grad_lr0.sum(dim=1, keepdim=False)
    grad_lr1 = grad_lr1.sum(dim=1, keepdim=False)
    grad_lr2 = grad_lr2.sum(dim=1, keepdim=False)

    # also for the first order backward:

    dx2 = silu_x0 * dh
    dx0 = dh * x2 * silu_bp_multiplier

    dx0 = dx0 * lr0
    dx2 = dx2 * lr2

    hidden_lr1 = lr1 * x2 * silu_x0

    grad_x0_x2 = torch.cat([grad_x0, grad_x2], dim=1)
    dx0_x2 = torch.cat([dx0, dx2], dim=1)

    x_dtype = x0.dtype

    # grad_dh_and_hidden_lr1 = torch.cat([grad_dh, hidden_lr1], dim=1)

    return (
        grad_dh.to(x_dtype),
        grad_x0_x2.to(x_dtype),
        grad_lr0,
        grad_lr1,
        grad_lr2,
        dx0_x2.to(x_dtype),
        hidden_lr1.to(x_dtype),
    )


########################################################
# Pytorch Reference implementation
########################################################


@torch.compile
def reference_lact_swiglu_ffn_fast_weight_grads(W0_W2, W1, K, V, lr0, lr1, lr2):
    """
    Args:
        W0, W2: [B, M, K] or [B, Hidden, D]
        W1:     [B, K, M] or [B, D, Hidden]
        X:      [M, N, K] or [B, num_Tokens, D]
    """
    W0, W2 = W0_W2.chunk(2, dim=1)
    Y0 = torch.bmm(W0, K.transpose(1, 2))
    Y2 = torch.bmm(W2, K.transpose(1, 2))

    DHidden = torch.bmm(W1.transpose(1, 2), V.transpose(1, 2))

    # DY0_with_lr0, DY2_with_lr2, Hidden_with_lr1 = ref_pytorch_swiglu_bwd(
    #     DHidden, Y0, Y2, lr0, lr1, lr2
    # )
    ### Element-wise ops
    lr0 = lr0.unsqueeze(dim=1)
    lr1 = lr1.unsqueeze(dim=1)
    lr2 = lr2.unsqueeze(dim=1)

    x0_sigmoid = torch.sigmoid(Y0)

    dx2 = x0_sigmoid * Y0 * DHidden
    dx0 = DHidden * Y2 * x0_sigmoid * (1 + Y0 * (1 - x0_sigmoid))

    DY0_with_lr0 = dx0 * lr0
    DY2_with_lr2 = dx2 * lr2

    Hidden_with_lr1 = lr1 * Y2 * torch.nn.functional.silu(Y0)
    ### Element-wise ops done.

    DY0_with_lr0_and_DY2_with_lr2 = torch.cat([DY0_with_lr0, DY2_with_lr2], dim=1)

    DW0_DW2 = torch.bmm(DY0_with_lr0_and_DY2_with_lr2, K)
    DW1 = torch.bmm(V.transpose(1, 2), Hidden_with_lr1.transpose(1, 2))

    return DW0_DW2, DW1
