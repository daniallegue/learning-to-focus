"""
Config Dataclass for the Transformer backbone.

Part of conference submission: "Learning to Focus: Prioritizing Informative Histories with Structured Attention
 Mechanisms in Partially Observable Reinforcement Learning"
"""
from typing import Optional
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    tokens_per_block: int
    max_blocks: int
    attention: str  # 'causal', 'local', 'local+routing', 'routing', 'adaptive'

    num_layers: int
    num_heads: int
    embed_dim: int

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    # for RoPE
    rope_theta: float
    max_seq_len: int
    rotary_emb: bool = False

    local_window_size: Optional[int] = None

    # Adaptive Hybrid Params
    init_adaptive_span: Optional[float] = 6.0
    max_adaptive_span: Optional[int] = 20.0
    adapt_span_ramp : Optional[int] = 3
    adapt_span_loss : Optional[float] = 0.025

    # GAAM Params
    init_adaptive_mu: Optional[float] = 4.0  # where to initialize each head’s mean offset
    init_adaptive_sigma: Optional[float] = 1.0  # where to initialize each head’s variance (before softplus)

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks