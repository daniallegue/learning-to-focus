"""
Adaptive Attention Mechanism for Transformer World Model
"""
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .kv_caching import KeysValues
from .transformer import apply_rotary_emb
from .attention import Attention
from .transformer_config import TransformerConfig

class AdaptiveSpanAttention(Attention):
    """
    Implements adaptive span self-attention mechanism for transformers.

    Arguments:
        config (:obj:`TransformerConfig`): Configuration object containing hyperparameters.

    Attributes:
        - config (:obj:`TransformerConfig`): Stores the configuration for the self-attention module.
        - num_heads (:obj:`int`): Number of attention heads.
        - key (:obj:`nn.Linear`): Linear layer to project input to key vectors.
        - query (:obj:`nn.Linear`): Linear layer to project input to query vectors.
        - value (:obj:`nn.Linear`): Linear layer to project input to value vectors.
        - attn_drop (:obj:`nn.Dropout`): Dropout layer for attention weights.
        - resid_drop (:obj:`nn.Dropout`): Dropout layer for residual connection.
        - proj (:obj:`nn.Linear`): Final linear layer for projection.
        - mask (:obj:`torch.Tensor`): Mask tensor for causal or block-causal attention.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        assert config.embed_dim % config.num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.config = config
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim
        self.head_dim = config.embed_dim // config.num_heads
        self.max_len = config.max_tokens

        # Number of bottom layers use as strict local attention
        self.hybrid_local_layers: int = 4
        self.aha : bool = True

        # projectors
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)

        # dropout & output projections
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        # adaptive span parameters
        self.init_span = config.init_adaptive_span or config.local_window_size
        inverse_softplus = lambda x : math.log(math.expm1(x))
        self.init_p = inverse_softplus(self.init_span)

        self.span_p = nn.Parameter(torch.full((self.num_heads, ), self.init_p))

    def forward(self, x: torch.Tensor, kv_cache: Optional[KeysValues] = None,
                valid_context_lengths: Optional[torch.Tensor] = None, freqs_cis: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the adaptive span self-attention mechanism.

        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor of shape (B, T, C) where B is batch size,
                                        T is sequence length, and C is embedding dimension.
            - kv_cache (:obj:`Optional[KeysValues]`): Optional key-value cache for faster inference.
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Optional tensor containing valid context lengths.
            - freqs_cis (:obj:`torch.Tensor`): Frequency components for rotary position embeddings, used to modulate the attention mechanism (default: None).

        Returns:
            - torch.Tensor: Output tensor of shape (B, T, C).
        """

        B, T, C = x.size()
        device = x.device()

        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, d)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if getattr(self.config, 'rotary_emb', False):
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()  # (B, nh, L+T, d)
            L = k.size(-2) - T
        else:
            L = 0  # no past context

        total_len = L + T  # number of keys available

        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, nh, T, total_len)

        # adaptive attention masks
        spans = F.softplus(self.span_p) # (nh, )
        spans_int = spans.floor().clamp(max=self.max_len).long() # (nh, )

        # computes pairwise distances |i-j| to check if attend
        query_pos = torch.arange(L, L + T, device=device).unsqueeze(1)
        key_pos = torch.arange(0, total_len, device=device).unsqueeze(0)
        distances = (query_pos - key_pos).abs()  # (T, total_len)

        # masks
        d_expanded = distances.unsqueeze(0).expand(self.num_heads, -1, -1)
        spans_exp = spans_int.unsqueeze(1).unsqueeze(2)  # (nh, 1, 1)
        head_mask = d_expanded <= spans_exp

        # expand (B, nh, T, total_len)
        mask = head_mask.unsqueeze(0).expand(B, -1, -1, -1)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        y = attn @ v  # (B, nh, T, d)
        y = rearrange(y, 'b h t d -> b t (h d)')

        y = self.resid_drop(self.proj(y))
        return y

    @torch.no_grad()
    def get_attention_map(self, x: torch.Tensor, kv_cache: Optional[KeysValues] = None,
                          valid_context_lengths: Optional[torch.Tensor] = None):
        """
        Returns the attention weight maps for visualization.
        """
        B, T, C = x.shape
        device = x.device

        # same projection steps as forward, but only compute scores & mask
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if getattr(self.config, 'rotary_emb', False):
            q, k = apply_rotary_emb(q, k, freqs_cis=None)

        if kv_cache is not None:
            kv_cache.update(k, self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2))
            k, _ = kv_cache.get()
            L = k.size(-2) - T
        else:
            L = 0

        total_len = L + T
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # build same mask as in forward
        spans = F.softplus(self.span_p)
        spans_int = spans.floor().clamp(max=self.max_len).long()
        query_pos = torch.arange(L, L + T, device=device).unsqueeze(1)
        key_pos = torch.arange(0, total_len, device=device).unsqueeze(0)
        distances = (query_pos - key_pos).abs()
        d_exp = distances.unsqueeze(0).expand(self.num_heads, -1, -1)
        spans_e = spans_int.unsqueeze(1).unsqueeze(2)
        head_mask = d_exp <= spans_e
        mask = head_mask.unsqueeze(0).expand(B, -1, -1, -1)

        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return attn
