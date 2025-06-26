# -*- coding: utf-8 -*-

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class LaCTSWIGLUConfig(PretrainedConfig):
    """
    Configuration for LaCT-SWIGLU model.
    It implements the LaCT-SWIGLU layer mixed with in-layer sliding window attention

    Args:
        hidden_size (int, optional): The hidden size of the model. Defaults to 2048.
        num_hidden_layers (int, optional): The number of hidden layers in the model. Defaults to 24.
        num_attn_heads (int, optional): The number of attention heads in the model. Defaults to 32.
        num_lact_heads (int, optional): The number of feed-forward heads in the model. Defaults to 4.
    """
    model_type = 'lact_swiglu'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_attn_heads: int = 32,
        num_lact_heads: int = 4,
        inter_multi: int = 1,
        qkv_bias: bool = False,
        attn_qk_norm: bool = False,
        lact_chunk_size: int = 2048,
        use_muon: bool = False,
        lr_dim: int = 1,
        qkv_silu: bool = True,
        lr_parameterization: str = 'mamba',
        learnable_ttt_scale: bool = True,
        use_momentum: bool = True,
        ttt_loss_type: str = "dot_product", # "l2"
        ttt_prenorm: bool = False,    # pre-norm or post-norm for ttt.   
        # prenorm ttt:  state = state + f(norm(state))
        # postnorm ttt:  state = norm(state + f(state)
        ttt_nope: bool = False, # if True, no positional encoding for query and key used in ttt.  
        w0_w2_low_rank: int = -1, # -1 means fully learnable.  > 1 means low rank parameterization of the initial learnable weights. 
        window_size: int = 2048,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: int = 2048,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.006,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        last_layer_fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        vocab_size: int = 32000,
        fw_init_gain: float = 0.5,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attn_heads = num_attn_heads
        self.num_lact_heads = num_lact_heads
        self.inter_multi = inter_multi
        self.qkv_bias = qkv_bias
        self.attn_qk_norm = attn_qk_norm
        self.lact_chunk_size = lact_chunk_size
        self.use_muon = use_muon
        self.lr_dim = lr_dim
        self.qkv_silu = qkv_silu
        self.window_size = window_size
        self.lr_parameterization = lr_parameterization
        self.learnable_ttt_scale = learnable_ttt_scale
        self.ttt_prenorm = ttt_prenorm
        self.ttt_nope = ttt_nope
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        self.fuse_norm = fuse_norm
        self.last_layer_fuse_norm = last_layer_fuse_norm # seems that you need to set this to False to use activation checkpointing for every layer. 
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.vocab_size = vocab_size

        self.use_momentum = use_momentum
        self.ttt_loss_type = ttt_loss_type
        self.w0_w2_low_rank = w0_w2_low_rank
        self.fw_init_gain = fw_init_gain
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
