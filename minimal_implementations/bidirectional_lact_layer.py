import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from einops import rearrange


@torch.compile()
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


@torch.compile()
def l2_norm(x: torch.Tensor):
    """
    x: [b, l, d]
    """
    x_type = x.dtype
    ret = x / (x.norm(dim=-1, keepdim=True) + 1e-5)  # norm will upcast to float32
    return ret.type(x_type)


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


@torch.compile
def bidirectional_lact_swiglu(
    w0: torch.Tensor,  # [b, dh, dk]
    w1: torch.Tensor,  # [b, dv, dh]
    w2: torch.Tensor,  # [b, dh, dk]
    q: torch.Tensor,  # [b, l, dk]
    k: torch.Tensor,  # [b, l, dk]
    v: torch.Tensor,  # [b, l, dv]
    lr0: torch.Tensor,  # [b, l, 1]
    lr1: torch.Tensor,  # [b, l, 1]
    lr2: torch.Tensor,  # [b, l, 1]
    use_muon: bool = True,
) -> torch.Tensor:
    """
    Bidirectional LaCT with SwiGLU fast weight function.
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))

    About precision:
        w0, w1, w2 are mostly likely fp32. 
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.


    FLOPS:
        (assume dk=dv denoted as D, hidden dimension of swiglu-mlp is H, ignore muon)
        Forward pass with key: 4 * D * H * L * B
        Backward pass: 8 * D * H * L * B
        Forward with Query: 6 * D * H * L * B
        Total: 18 * D * H * L * B
    Outputs:
        o: [b, l, dv]
    """

    # adding detach here sometimes improves stability.
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    ######### update the fast weight w0, w1, w2 with test-time training #########

    #### Forward pass with key
    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    gate_before_act = torch.bmm(w0, k.transpose(1, 2))
    hidden_before_mul = torch.bmm(w2, k.transpose(1, 2))
    hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

    #### Backward pass to compute fast weight gradients
    # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
    dhidden = torch.bmm(w1.transpose(1, 2), v)

    dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
    dgate = dhidden * hidden_before_mul
    dgate_before_act = silu_backprop(dgate, gate_before_act)

    # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
    dw1 = torch.bmm(v, (hidden.transpose(1, 2) * lr1).type_as(v))  # [b, d, d]
    # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
    dw0 = torch.bmm(dgate_before_act, (k * lr0).type_as(dgate_before_act))
    dw2 = torch.bmm(dhidden_before_mul, (k * lr2).type_as(dhidden_before_mul))
    

    if use_muon:
        w0 = zeropower_via_newtonschulz5(dw0)
        w1 = zeropower_via_newtonschulz5(dw1)
        w2 = zeropower_via_newtonschulz5(dw2)

    w1 = w1 + dw1
    w0 = w0 + dw0
    w2 = w2 + dw2

    w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
    w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
    w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    ######### apply the updated fast weights to the query #########

    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    h = torch.bmm(w2, q)
    gate = F.silu(torch.bmm(w0, q), inplace=True)
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
    o = torch.bmm(w1, gate * h).transpose(1, 2)

    return o


def inv_softplus(x):
    if isinstance(x, torch.Tensor):
        y = x + torch.log(-torch.expm1(-x))
    else:
        y = x + math.log(-math.expm1(-x))
    return y


class BidirectionalLaCTSwiGLU(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        head_dim: int,
        inter_multi: float = 1,
        use_o_norm: bool = True,  # recommended to be True
        qk_l2_norm: bool = True,  # recommended to be True
        use_muon: bool = True,  # if your seq len > head_dim * 2, recommended to be True
        base_lr: float = 1e-2,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.inter_multi = inter_multi
        self.use_o_norm = use_o_norm
        self.qk_l2_norm = qk_l2_norm

        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.lr_dim = 1   # single scalar learning rate for each head
        self.lr_proj = nn.Linear(dim, self.lr_dim * 3 * self.num_heads, bias=False)
        self.base_lr = base_lr
        self.base_lr_inv = inv_softplus(base_lr)

        # create initial fast weights
        d_in, d_out = self.head_dim, self.head_dim
        d_h = int(self.head_dim * self.inter_multi)

        self.w0 = nn.Parameter(torch.randn(self.num_heads, d_h, d_in) / math.sqrt(d_in))
        self.w1 = nn.Parameter(torch.randn(self.num_heads, d_out, d_h) / math.sqrt(d_h))
        self.w2 = nn.Parameter(
            torch.randn(self.num_heads, d_h, d_in) / math.sqrt(d_in)
        )

        self.qk_l2_norm = qk_l2_norm
        self.use_muon = use_muon

        self.use_o_norm = use_o_norm
        if self.use_o_norm:
            self.o_norm = nn.RMSNorm(head_dim, eps=1e-5, elementwise_affine=True)
        else:
            self.o_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [b, l, d]
        """

        qkv = F.silu(self.to_qkv(x), inplace=True)  # SiLU - Linear

        # [b * num_heads, l, head_dim]
        q, k, v = rearrange(
            qkv,
            "b l (qkv h d) -> qkv (b h) l d",
            qkv=3,
            h=self.num_heads,
            d=self.head_dim,
        )

        if self.qk_l2_norm:
            q = l2_norm(q)
            k = l2_norm(k)

        # better to have float32 for lr.
        # For muon, I found that float16 is still very good.
        with torch.autocast(device_type="cuda", enabled=False):
            lr = self.lr_proj(x)  # [b, l, lr_dim]

        lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)

        # [b * num_heads, l, 1] for each lr
        lr0, lr1, lr2 = rearrange(
            lr, "b l (h lrs d) -> lrs (b h) l d", lrs=3, h=self.num_heads, d=self.lr_dim
        )

        # [nh, d, d] -> [b * nh, d, d]
        w0 = self.w0.repeat(x.shape[0], 1, 1)
        w1 = self.w1.repeat(x.shape[0], 1, 1)
        w2 = self.w2.repeat(x.shape[0], 1, 1)

        # [b * num_heads, l, head_dim]
        output = bidirectional_lact_swiglu(
            w0, w1, w2, q, k, v, lr0, lr1, lr2, self.use_muon
        )

        output = self.o_norm(output)
        output = rearrange(
            output, "(b h) l d -> b l (h d)", h=self.num_heads, b=x.shape[0]
        )
        output = self.o_proj(output)

        # [b, l, d]
        return output



def _test_layer():
    B, L, D, HeadDim = 4, 32768, 2048, 512
    
    layer = BidirectionalLaCTSwiGLU(D, HeadDim, use_muon=True)

    layer = layer.to("cuda")

    x = torch.randn(B, L, D).to("cuda")

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        output = layer(x)
    print(output.shape, output.dtype)
    print("Input norm", x.norm(), "Output norm", output.norm())


if __name__ == "__main__":
    _test_layer()
