import torch


try:
    from triton_swiglu_bwd_kernels import (
        swiglu_backward_three_bmm_triton,
        swiglu_backward_three_bmm_triton_op,
    )
    from triton_swiglu_kernels import fused_two_mm_swiglu_triton
except ImportError:
    from .triton_swiglu_bwd_kernels import (
        swiglu_backward_three_bmm_triton,
        swiglu_backward_three_bmm_triton_op,
    )
    from .triton_swiglu_kernels import fused_two_mm_swiglu_triton
from torch.autograd.function import once_differentiable


class FusedSwiGLUFFNFwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, W0_W2, W1, X):
        """
        Args:
            W0_W2: [B, 2 * Hidden, D]
            W1:     [B, K, M] or [B, D, Hidden]
            X:      [M, N, K] or [B, num_Tokens, D]
        Outs:
            Hidden: [B, N, K] or [B, num_tokens, Hidden]

        W1 @ [SiLU(W0 @ X.T) * (W2 @ X.T)]
        """

        # [B, Hidden, num_tokens]
        #### Without this triton kernel, we will materize Y2, SiLU(Y0) * Y2.
        #### 2 + 1 read and write.
        # Here we only have one write.
        Hidden = fused_two_mm_swiglu_triton(W0_W2, X)

        # -> [B, num_tokens, D]
        # output = torch.bmm(W1, Hidden).transpose(1, 2)
        output = torch.bmm(Hidden.transpose(1, 2), W1.transpose(1, 2))
        ctx.save_for_backward(W0_W2, W1, X)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: [B, num_tokens, D]
        Outs:
            grad_W0_W2: [B, 2 * Hidden, D]
            grad_W1: [B, D, Hidden]
            grad_X: [B, D, num_tokens]
        """
        W0_W2, W1, X = ctx.saved_tensors
        # [B, 2 * Hidden, num_tokens]
        DY0_DY2, Hidden = swiglu_backward_three_bmm_triton(
            # DY0_DY2, Hidden = swiglu_backward_three_bmm_triton_op(
            W0_W2,
            W1,
            X,
            grad_out.contiguous(),
        )

        # [B, D, num_tokens] @ [B, num_tokens, Hidden] -> [B, D, Hidden]
        grad_W1 = torch.bmm(grad_out.transpose(1, 2), Hidden.transpose(1, 2))

        # [B, 2 * Hidden, num_tokens] @ [B, num_tokens, D] -> [B, 2 * Hidden, D]
        grad_W0_W2 = torch.bmm(DY0_DY2, X)

        # [B, 2 * Hidden, num_tokens].T @ [B, 2 * Hidden, D] -> [B, 2 * Hidden, D]
        grad_X = torch.bmm(DY0_DY2.transpose(1, 2), W0_W2)

        return (grad_W0_W2, grad_W1, grad_X)


def fused_swiglu_ffn_fwd(W0_W2, W1, X):
    """
    Args:
        W0_W2: [B, 2 * Hidden, D]
        W1:     [B, D, Hidden]
        X:      [B, num_Tokens, D]
    Outs:
        Hidden: [B, num_tokens, Hidden]
    This function implements the following formula:
    output = W1 @ [SiLU(W0 @ X.T) * (W2 @ X.T)].T

    During backward pass, it recomputes W0@X.T and W2@X.T.

    The triton kernel does two optimizations:
    1. in fwd pass, it fuse SiLU(W0 @ X.T) * (W2 @ X.T).  The epilogues after two GEMM.
    2. in bwd pass, it fuse epilogues after three GEMM, W0 @ X.T, W2 @ X.T, and W1 @ Grad_Out.T.
    """
    return FusedSwiGLUFFNFwd.apply(W0_W2, W1, X)


############################################################
# Pytorch Reference Code Below
############################################################


@torch.compile
def reference_swiglu_ffn_fwd(W0_W2, W1, X):
    """
    Args:
        W0, W2: [B, M, K] or [B, Hidden, D]
        W1:     [B, K, M] or [B, D, Hidden]
        X:      [M, N, K] or [B, num_Tokens, D]
    """
    W0, W2 = W0_W2.chunk(2, dim=1)
    Y0 = torch.bmm(W0, X.transpose(1, 2))
    Y2 = torch.bmm(W2, X.transpose(1, 2))
    Hidden = torch.nn.functional.silu(Y0) * Y2
    return torch.bmm(W1, Hidden).transpose(1, 2)


if __name__ == "__main__":
    pass
