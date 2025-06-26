# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.utils import logging

from fla.models.utils import Cache
from fla.modules import RMSNorm, RotaryEmbedding
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat

from .ttt_operation import block_causal_lact_swiglu, prenorm_block_causal_lact_swiglu, l2_norm

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)


def inv_softplus(x):
    if isinstance(x, torch.Tensor):
        y = x + torch.log(-torch.expm1(-x))
    else:
        y = x + math.log(-math.expm1(-x))
    return y


class LowRankFastWeight(nn.Module):
    """
    Low rank fast weight. This is a compromise to keep the number of parameters low when comparing against baselines. 
    Idealy, low-rank parameterization always hurts the performance. 
    Args:
        num_heads: number of heads
        out_features: output features
        in_features: input features
        rank: rank of the low rank fast weight
        init_gain: initialization gain
        add_identity: whether to add identity matrix to the fast weight
    Returns:
        W: [num_heads, out_features, in_features]
    W = W_left @ W_right + I * 0.5
        where I is the identity matrix if add_identity is True.
    """
    def __init__(self, num_heads, out_features, in_features, rank=32, 
                 init_gain=0.5, add_identity=False):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank
        self.add_identity = add_identity
        
        self.w_left = nn.Parameter(torch.randn(num_heads, out_features, rank))
        self.w_right = nn.Parameter(torch.randn(num_heads, rank, in_features))
        self.init_gain = init_gain

        print("init low rank fast weight", num_heads, out_features, in_features, rank)

    def _init_weights(self):
        
        nn.init.normal_(self.w_left, std=1.0 / math.sqrt(self.rank) * self.init_gain)
        nn.init.normal_(self.w_right, std=1.0 / math.sqrt(self.in_features) * self.init_gain)

    def forward(self,):
        """
        Returns:
            W: [num_heads, out_features, in_features]
            W = W_left @ W_right + I * 0.5
            where I is the identity matrix if add_identity is True.
        """

        W = self.w_left @ self.w_right

        if self.add_identity:
            W += torch.eye(self.out_features, self.in_features, device=W.device, dtype=W.dtype).unsqueeze(0) * 0.5

        return W


class LaCTSWIGLULayer(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_attn_heads: int,
        num_lact_heads: int,
        inter_multi: float,
        window_size: int,
        lact_chunk_size: int,
        qkv_bias: bool = False,
        attn_qk_norm: bool = True,
        qkv_silu: bool = True,
        lr_dim: int = 1,
        use_muon: bool = False,
        lr_parameterization: str = "mamba",
        learnable_ttt_scale: bool = False,
        ttt_prenorm: bool = False,
        ttt_nope: bool = False,
        rope_theta: float = 500000.0,
        layer_idx: int = None,
        max_position_embeddings: int = 2048,
        w0_w2_low_rank: int = -1,
        use_momentum: bool = False,
        ttt_loss_type: str = "dot_product",
        fw_init_gain: float = 0.5, # init the fast weights
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_attn_heads # num of heads for attention
        self.inter_multi = inter_multi
        self.window_size = window_size
        # head dim for attention
        self.head_dim = hidden_size // num_attn_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        
        self.attn_qk_norm = attn_qk_norm
        if self.attn_qk_norm:
            self.q_norm = RMSNorm(self.hidden_size)
            self.k_norm = RMSNorm(self.hidden_size)

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rope_theta = rope_theta
        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)
        self.layer_idx = layer_idx
        self.max_position_embeddings = max_position_embeddings

        ### Fast Weight init
        self.use_muon = use_muon
        self.lact_chunk_size = lact_chunk_size
        self.num_fw_heads = num_lact_heads
        self.fw_head_dim = self.hidden_size // self.num_fw_heads
        self.qkv_silu = qkv_silu
        self.ttt_prenorm = ttt_prenorm
        self.ttt_nope = ttt_nope
        
        d_in, d_out = self.fw_head_dim, self.fw_head_dim
        d_h = int(d_in * inter_multi)

        self.d_h = d_h
        self.d_in = d_in
        self.d_out = d_out
        self.w0_w2_low_rank = w0_w2_low_rank
        self.fw_init_gain = fw_init_gain

        # Low Rank parameterization of the fast weights.  
        # This is a compromise to keep the number of parameters low when comparing against baselines. 
        # Idealy, low-rank parameterization always hurts the performance. 
        if self.w0_w2_low_rank > 0:
            self.w0 = LowRankFastWeight(self.num_fw_heads, d_h, d_in, self.w0_w2_low_rank, init_gain=self.fw_init_gain, add_identity=True)
            self.w2 = LowRankFastWeight(self.num_fw_heads, d_h, d_in, self.w0_w2_low_rank, init_gain=self.fw_init_gain, add_identity=True)
        else:
            self.w0 = nn.Parameter(
                torch.randn(self.num_fw_heads, int(d_h), d_in)
                / math.sqrt(d_in)
            )  # [num_fw_heads, d_h, d_in]
            self.w2 = nn.Parameter(
                torch.randn(self.num_fw_heads, int(d_h), d_in)
                / math.sqrt(d_in)
            )  # [num_fw_heads, d_h, d_in]
        self.w1 = nn.Parameter(
            torch.randn(self.num_fw_heads, int(d_out), d_h)
            / math.sqrt(d_h)
        )  # [num_fw_heads, d_out, d_h]
        
        #### Per-Token LR parameterization. 
        self.lr_dim = int(lr_dim * 3 * self.num_fw_heads)
        self.lr_proj = nn.Linear(self.hidden_size, self.lr_dim)
        base_lr = 0.001
        # Lr parameterization and initialization
        if lr_parameterization.lower() == "mamba":
            self.base_lr_inv = inv_softplus(base_lr)
        self.lr_parameterization = lr_parameterization
        
        #### per-channel scaling and offset for Q, and K. 
        self.qk_scale = nn.Parameter(torch.ones(hidden_size, 2))
        self.qk_offset = nn.Parameter(torch.zeros(hidden_size, 2))
        self.learnable_ttt_scale = learnable_ttt_scale
        if self.learnable_ttt_scale:
            # per-head scaling. 
            self.ttt_scale_proj = nn.Linear(hidden_size, self.num_fw_heads)
        
        # ttt output norm per head. 
        self.ttt_norm = RMSNorm(self.fw_head_dim, elementwise_affine=True)
        
        self.use_momentum = use_momentum
        if self.use_momentum:
            self.momentum_proj = nn.Sequential(
                nn.Linear(hidden_size, self.num_fw_heads),
                nn.Sigmoid(),
            )

        self.ttt_loss_type = ttt_loss_type
        
        assert self.ttt_loss_type in ["dot_product"], f"Loss type {self.ttt_loss_type} not supported"

        
    def _rescale_qk(self, q, k):
        """
        Args:
            q: [b, s, d]
            k: [b, s, d]
        Returns:
            q: [b, s, d]
            k: [b, s, d]
        """
        qk_scale = self.qk_scale.view(1, 1, -1, 2)
        qk_offset = self.qk_offset.view(1, 1, -1, 2)
        q = q * qk_scale[:, :, :, 0] + qk_offset[:, :, :, 0]
        k = k * qk_scale[:, :, :, 1] + qk_offset[:, :, :, 1]
        return q, k
    
    def forward(
        self,
        hidden_states: torch.Tensor, # [b, s, d]
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
        #### compute window attention first, then do ttt. ####

        if self.attn_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # rescale and reshift the q, k for test-time training layer.
        fast_q, fast_k = self._rescale_qk(q, k)
        fast_v = v
        
        q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        # WARNING: current implementation ignores cu_seqlens for ttt-layer. 
        cu_seqlens = kwargs.get('cu_seqlens', None)

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        # [b, s, n_h, d]
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size)
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            q, k, v, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(q, k, v, attention_mask, q_len)
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            )
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            o = flash_attn_varlen_func(
                q.squeeze(0), k.squeeze(0), v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            ).unsqueeze(0)
        else:
            o = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            )
        o = o.reshape(batch_size, q_len, -1)

        ##### TTT starts here. 
        # Split heads then merge it to batch dimension
        fast_q = rearrange(fast_q, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_fw_heads)
        fast_k = rearrange(fast_k, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_fw_heads)
        fast_v = rearrange(fast_v, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_fw_heads)
        

        if self.qkv_silu:
            fast_q = F.silu(fast_q)
            fast_k = F.silu(fast_k)
            fast_v = F.silu(fast_v)

        # per head l2 norm for fast_q, fast_k. 
        fast_q = l2_norm(fast_q)
        fast_k = l2_norm(fast_k)
        
        if not self.ttt_nope:
            #### Apply rotary embedding.  Here we use the same rope as the attention layer. 
            # I observed that using NoPE for ttt (No positional encoding) here also works. 
            fast_q = rearrange(fast_q, '(b n_h) s d -> b s (n_h d)', n_h=self.num_fw_heads)
            fast_k = rearrange(fast_k, '(b n_h) s d -> b s (n_h d)', n_h=self.num_fw_heads)

            fast_q = rearrange(fast_q, 'b s (n_h d) -> b s n_h d', n_h=self.num_heads)
            fast_k = rearrange(fast_k, 'b s (n_h d) -> b s n_h d', n_h=self.num_heads)

            fast_q, fast_k = self.rotary(fast_q, fast_k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

            fast_q = rearrange(fast_q, 'b s n_h d -> b s (n_h d)', n_h=self.num_heads)
            fast_k = rearrange(fast_k, 'b s n_h d -> b s (n_h d)', n_h=self.num_heads)

            fast_q = rearrange(fast_q, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_fw_heads)
            fast_k = rearrange(fast_k, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_fw_heads)
            #### RoPE done. ####

        if self.w0_w2_low_rank > 0:
            fw_w0 = self.w0().repeat(batch_size, 1, 1)
            fw_w2 = self.w2().repeat(batch_size, 1, 1)
        else:
            fw_w0 = self.w0.repeat(batch_size, 1, 1) # [nh, d_h, d_in] -> [b*nh, d_h, d_in]
            fw_w2 = self.w2.repeat(batch_size, 1, 1) # [nh, d_h, d_in] -> [b*nh, d_h, d_in]
        
        fw_w1 = self.w1.repeat(batch_size, 1, 1) # [nh, d_out, d_h] -> [b*nh, d_out, d_h]

        lr = self.lr_proj(hidden_states) # [b, s, num_heads * lr_dim_per_head]
        if self.lr_parameterization == "mamba":
            lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)
        else:
            raise NotImplementedError(f"LR parameterization {self.lr_parameterization} not implemented")
        fw_lr = rearrange(lr, 'b s (n_h lr_dim) -> (b n_h) s lr_dim', n_h=self.num_fw_heads)
        fw_lr1, fw_lr2, fw_lr3 = fw_lr.chunk(3, dim=-1)

        if self.use_momentum:
            momentum = self.momentum_proj(hidden_states) # [b, s, nh]
            momentum = rearrange(momentum, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_fw_heads)
        else:
            momentum = None
        
        # [b * nh, s, d_ttt_head]
        if self.ttt_prenorm:
            # pre-norm version of ttt.   state = state + f(norm(state))
            fw_x = prenorm_block_causal_lact_swiglu(
                fw_w0, fw_w1, fw_w2, fast_q, fast_k, fast_v,
                fw_lr1, fw_lr2, fw_lr3,
                chunk_size=self.lact_chunk_size,
                use_muon=self.use_muon,
                momentum=momentum)
        else:
            # post-norm version of ttt.   state = norm(state + f(state))
            fw_x = block_causal_lact_swiglu(
                fw_w0, fw_w1, fw_w2, fast_q, fast_k, fast_v,
                fw_lr1, fw_lr2, fw_lr3,
                chunk_size=self.lact_chunk_size,
                use_muon=self.use_muon,
                momentum=momentum)
        
        # per-head output norm for ttt layer.
        ttt_x_normed = self.ttt_norm(fw_x)
        if self.learnable_ttt_scale: 
            ttt_scale = F.silu(self.ttt_scale_proj(hidden_states), inplace=False)
            ttt_scale = rearrange(ttt_scale, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_fw_heads)
            ttt_x_normed = ttt_x_normed * ttt_scale

        ttt_x_normed = rearrange(ttt_x_normed, '(b n_h) s d -> b s (n_h d)', n_h=self.num_fw_heads)

        o = o + ttt_x_normed
        o = self.o_proj(o)
        
        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values

    def _upad_input(self, q, k, v, attention_mask, q_len):
        batch_size, seq_len, num_key_value_heads, head_dim = k.shape
        cache_mask = attention_mask[:, -seq_len:]
        seqlens = cache_mask.sum(-1, dtype=torch.int32)
        indices_k = torch.nonzero(cache_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_k = seqlens.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

        k = index_first_axis(k.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k)
        v = index_first_axis(v.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k)
        if q_len == seq_len:
            q = index_first_axis(q.reshape(batch_size * seq_len, self.num_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_q = max_seqlen_k
            indices_q = indices_k
        elif q_len == 1:
            max_seqlen_q = 1
            # There is a memcpy here, that is very bad.
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
            indices_q = cu_seqlens_q[:-1]
            q = q.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -q_len:]
            q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, attention_mask)

        return q, k, v, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k)

        
        
        