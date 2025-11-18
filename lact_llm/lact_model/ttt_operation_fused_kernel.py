import torch
import torch.nn.functional as F
from einops import rearrange

try:

    from .lact_triton_kernels.lact_swiglu_ffn import fused_swiglu_ffn_fwd
    from .lact_triton_kernels.lact_fw_grad import (
        fused_lact_swiglu_ffn_fast_weight_grads,
    )
    from .lact_triton_kernels.triton_prenorm_update_with_momentum import (
        fused_prenorm_update_with_momentum_and_l2_norm,
    )

    from .lact_triton_kernels.l2norm_triton_kernels import l2_norm_add_fused
except ImportError:

    from lact_triton_kernels.lact_swiglu_ffn import fused_swiglu_ffn_fwd
    from lact_triton_kernels.lact_fw_grad import fused_lact_swiglu_ffn_fast_weight_grads
    from lact_triton_kernels.triton_prenorm_update_with_momentum import (
        fused_prenorm_update_with_momentum_and_l2_norm,
    )

    from lact_triton_kernels.l2norm_triton_kernels import l2_norm_add_fused


@torch.compile()
def zeropower_via_newtonschulz5(G):
    """
    This is an updated version of the zeropower_via_newtonschulz5 function in here:
    https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L26
    The code is modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49, which contains the original muon implementation.
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Args:
        G: [b, d, d']
    Returns:
        X: [b, d, d']
    FLOPS:  When d=d', Total FLOPS=30 * b * d^3
    """
    assert len(G.shape) == 3
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


@torch.compile()
def postnorm_block_causal_lact_swiglu_fused_kernel_triton(
    w0: torch.Tensor,  # [b, dh, d], fp32 or b16
    w1: torch.Tensor,  # [b, d, dh], fp32 or b16
    w2: torch.Tensor,  # [b, dh, d], fp32 or b16
    q: torch.Tensor,  # [b, l, d], bf16
    k: torch.Tensor,  # [b, l, d], bf16
    v: torch.Tensor,  # [b, l, d], bf16
    lr0: torch.Tensor,  # [b, l, 1], fp32
    lr1: torch.Tensor,  # [b, l, 1], fp32
    lr2: torch.Tensor,  # [b, l, 1], fp32
    chunk_size: int = 2048,  # test-time training chunk size
    use_muon: bool = False,
    momentum: torch.Tensor = None,  # [b, s, 1], fp32 or bf16
):
    """
    Block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))

    About precision:
        w0, w1, w2 are recommended to be fp32.
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.

    FLOPS:
        (assume dk=dv denoted as D, hidden dimension of swiglu-mlp is H, ignore muon, ignore last chunk)
        Forward pass with key: 4 * D * H * L * B
        Backward pass: 8 * D * H * L * B
        Forward with Query: 6 * D * H * L * B
        Total: 18 * D * H * L * B
    Outputs:
        o: [b, l, dv]
    """

    w0_w2 = torch.cat([w0, w2], dim=1).contiguous()
    w0_w2_norm = w0_w2.norm(dim=2, keepdim=False)
    w1_norm = w1.norm(dim=2, keepdim=False)

    if momentum is not None:
        # same dtype as w0_w2_main and w1_main, recommended to be fp32
        dw0_dw2_momentum = torch.zeros_like(w0_w2)
        dw1_momentum = torch.zeros_like(w1)

    q_original_length = q.shape[1]
    ### Padding the inputs to make the length a multiple of chunk_size
    k = F.pad(k, (0, 0, 0, -k.shape[1] % chunk_size))
    v = F.pad(v, (0, 0, 0, -v.shape[1] % chunk_size))
    q = F.pad(q, (0, 0, 0, -q.shape[1] % chunk_size))
    lr0 = F.pad(lr0, (0, 0, 0, -lr0.shape[1] % chunk_size))
    lr1 = F.pad(lr1, (0, 0, 0, -lr1.shape[1] % chunk_size))
    lr2 = F.pad(lr2, (0, 0, 0, -lr2.shape[1] % chunk_size))
    if momentum is not None:
        momentum = F.pad(momentum, (0, 0, 0, -momentum.shape[1] % chunk_size))
    num_chunks = (q.shape[1] + chunk_size - 1) // chunk_size

    k = rearrange(k, "b (n c) d -> n b c d", n=num_chunks)
    v = rearrange(v, "b (n c) d -> n b c d", n=num_chunks)
    q = rearrange(q, "b (n c) d -> n b c d", n=num_chunks)
    lr0 = rearrange(lr0, "b (n c) d -> n b (c d)", n=num_chunks, d=1)
    lr1 = rearrange(lr1, "b (n c) d -> n b (c d)", n=num_chunks, d=1)
    lr2 = rearrange(lr2, "b (n c) d -> n b (c d)", n=num_chunks, d=1)
    if momentum is not None:
        momentum = rearrange(momentum, "b (n c) 1 -> n b c 1", n=num_chunks)

    output = torch.zeros_like(q)

    e_index = 0
    seq_len = k.shape[1]
    for chunk_idx in range(num_chunks - 1):

        # [b, l, dk]
        ki = k[chunk_idx].contiguous()  # bf16
        # [b, l, dv]
        vi = v[chunk_idx].contiguous()  # bf16
        # [b, dh, l]
        qi = q[chunk_idx].contiguous()
        # [b, l, d/1] fp32
        lr1i = lr1[chunk_idx].contiguous()  # [b, l, d/1] fp32
        lr2i = lr2[chunk_idx].contiguous()  # [b, l, d/1] fp32
        lr0i = lr0[chunk_idx].contiguous()  # [b, l, d/1] fp32

        # apply first, perform swiglu ffn forward pass with the qi.
        w0_w2_bf16 = w0_w2.to(torch.bfloat16)
        w1_bf16 = w1.to(torch.bfloat16)
        output[chunk_idx] = fused_swiglu_ffn_fwd(w0_w2_bf16, w1_bf16, qi)

        # then, compute test-time training gradients for w0, w1, w2. under negative dot product loss.
        dw0_w2, dw1 = fused_lact_swiglu_ffn_fast_weight_grads(
            w0_w2_bf16, w1_bf16, ki, vi, lr0i, lr1i, lr2i
        )

        if momentum is not None:
            m_i = momentum[chunk_idx].contiguous()
            m_i = m_i.mean(dim=1, keepdim=True)  # [b, 1, 1]

            dw0_w2 = dw0_w2 + dw0_dw2_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw0_dw2_momentum = dw0_w2
            dw1_momentum = dw1

        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0_w2 = zeropower_via_newtonschulz5(dw0_w2)

        w0_w2 = l2_norm_add_fused(w0_w2, dw0_w2, w0_w2_norm, eps=1e-5)
        w1 = l2_norm_add_fused(w1, dw1, w1_norm, eps=1e-5)

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.

    qi = q[-1].contiguous()

    output[-1] = fused_swiglu_ffn_fwd(
        w0_w2.to(torch.bfloat16), w1.to(torch.bfloat16), qi
    )

    output = rearrange(output, "n b c d -> b (n c) d")

    return output[:, :q_original_length]


@torch.compile()
def prenorm_block_causal_lact_swiglu_fused_kernel_triton(
    w0: torch.Tensor,  # [b, dh, d], fp32 or b16
    w1: torch.Tensor,  # [b, d, dh], fp32 or b16
    w2: torch.Tensor,  # [b, dh, d], fp32 or b16
    q: torch.Tensor,  # [b, l, d], bf16
    k: torch.Tensor,  # [b, l, d], bf16
    v: torch.Tensor,  # [b, l, d], bf16
    lr0: torch.Tensor,  # [b, l, 1], fp32
    lr1: torch.Tensor,  # [b, l, 1], fp32
    lr2: torch.Tensor,  # [b, l, 1], fp32
    chunk_size: int = 2048,  # test-time training chunk size
    use_muon: bool = False,
    momentum: torch.Tensor = None,  # [b, s, 1], fp32 or bf16
):
    """
    Block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))

    About precision:
        w0, w1, w2 are recommended to be fp32.
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.

    FLOPS:
        (assume dk=dv denoted as D, hidden dimension of swiglu-mlp is H, ignore muon, ignore last chunk)
        Forward pass with key: 4 * D * H * L * B
        Backward pass: 8 * D * H * L * B
        Forward with Query: 6 * D * H * L * B
        Total: 18 * D * H * L * B
    Outputs:
        o: [b, l, d]
    """

    w0_w2 = torch.cat([w0, w2], dim=1).contiguous()
    w0_w2_norm = w0_w2.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)

    w0_w2_main = w0_w2
    w1_main = w1
    w0_w2 = w0_w2.to(torch.bfloat16)
    w1 = w1.to(torch.bfloat16)

    if momentum is not None:
        # same dtype as w0_w2_main and w1_main, recommended to be fp32
        dw0_dw2_momentum = torch.zeros_like(w0_w2_main)
        dw1_momentum = torch.zeros_like(w1_main)

    q_original_length = q.shape[1]
    ### Padding the inputs to make the length a multiple of chunk_size
    k = F.pad(k, (0, 0, 0, -k.shape[1] % chunk_size))
    v = F.pad(v, (0, 0, 0, -v.shape[1] % chunk_size))
    q = F.pad(q, (0, 0, 0, -q.shape[1] % chunk_size))
    lr0 = F.pad(lr0, (0, 0, 0, -lr0.shape[1] % chunk_size))
    lr1 = F.pad(lr1, (0, 0, 0, -lr1.shape[1] % chunk_size))
    lr2 = F.pad(lr2, (0, 0, 0, -lr2.shape[1] % chunk_size))
    if momentum is not None:
        momentum = F.pad(momentum, (0, 0, 0, -momentum.shape[1] % chunk_size))
    num_chunks = (q.shape[1] + chunk_size - 1) // chunk_size

    k = rearrange(k, "b (n c) d -> n b c d", n=num_chunks)
    v = rearrange(v, "b (n c) d -> n b c d", n=num_chunks)
    q = rearrange(q, "b (n c) d -> n b c d", n=num_chunks)
    lr0 = rearrange(lr0, "b (n c) d -> n b (c d)", n=num_chunks, d=1)
    lr1 = rearrange(lr1, "b (n c) d -> n b (c d)", n=num_chunks, d=1)
    lr2 = rearrange(lr2, "b (n c) d -> n b (c d)", n=num_chunks, d=1)
    if momentum is not None:
        momentum = rearrange(momentum, "b (n c) 1 -> n b c 1", n=num_chunks)
    output = torch.zeros_like(q)

    e_index = 0
    seq_len = k.shape[1]
    for chunk_idx in range(num_chunks - 1):

        # [b, l, dk]
        ki = k[chunk_idx].contiguous()  # bf16
        # [b, l, dv]
        vi = v[chunk_idx].contiguous()  # bf16
        # [b, dh, l]
        qi = q[chunk_idx].contiguous()
        # [b, l, d/1] fp32
        lr1i = lr1[chunk_idx].contiguous()  # [b, l, d/1] fp32
        lr2i = lr2[chunk_idx].contiguous()  # [b, l, d/1] fp32
        lr0i = lr0[chunk_idx].contiguous()  # [b, l, d/1] fp32

        # apply first, perform swiglu ffn forward pass with the qi.
        output[chunk_idx] = fused_swiglu_ffn_fwd(w0_w2, w1, qi)

        # then, compute test-time training gradients for w0, w1, w2. under negative dot product loss.
        dw0_w2, dw1 = fused_lact_swiglu_ffn_fast_weight_grads(
            w0_w2.to(torch.bfloat16), w1.to(torch.bfloat16), ki, vi, lr0i, lr1i, lr2i
        )

        if momentum is not None:
            m_i = momentum[chunk_idx].contiguous()
            m_i = m_i.mean(dim=1, keepdim=True)  # [b, 1, 1]

            dw0_w2 = dw0_w2 + dw0_dw2_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw0_dw2_momentum = dw0_w2
            dw1_momentum = dw1

        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0_w2 = zeropower_via_newtonschulz5(dw0_w2)

        w0_w2_main = w0_w2_main + dw0_w2
        w1_main = w1_main + dw1

        # cast to bf16
        w0_w2 = (
            w0_w2_main / (w0_w2_main.norm(dim=2, keepdim=True) + 1e-5) * w0_w2_norm
        ).to(torch.bfloat16)
        w1 = (w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm).to(
            torch.bfloat16
        )

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    qi = q[-1].contiguous()

    output[-1] = fused_swiglu_ffn_fwd(w0_w2, w1, qi)

    output = rearrange(output, "n b c d -> b (n c) d")

    return output[:, :q_original_length]
