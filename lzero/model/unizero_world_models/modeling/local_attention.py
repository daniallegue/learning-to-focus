"""
Local Self-Attention Module for Transformers (Global Attention)
"""
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from .kv_caching import KeysValues
from .transformer import apply_rotary_emb
from .attention import Attention
from .transformer_config import TransformerConfig

class LocalAttention(Attention):
    """
    Implements local self-attention mechanism for transformers.

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
        assert config.embed_dim % config.num_heads == 0, \
            "Embedding dimension must be divisible by number of heads."

        self.config = config
        self.num_heads = config.num_heads
        self.window = config.local_window_size

        # projectors
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)

        # dropout & output proj
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        # build a (max_tokens × max_tokens) local band mask once:
        indices = torch.arange(config.max_tokens)
        # mask[i,j] = True iff |i - j| <= window
        local_mask = (indices.unsqueeze(1) - indices.unsqueeze(0)).abs() <= self.window
        self.register_buffer('mask', local_mask)

    def forward(
            self,
            x: torch.Tensor,
            kv_cache: Optional[KeysValues] = None,
            valid_ctx_len: Optional[torch.Tensor] = None,
            freqs_cis: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for LocalAttention with cache-aware masking.

        Args:
            x (torch.Tensor): Input of shape (B, T, C).
            kv_cache (KeysValues, optional): Cache for fast autoregressive inference.
            valid_context_lengths (torch.Tensor, optional): For each batch element, how many of
                the last L cached positions are actually filled.
            freqs_cis (torch.Tensor, optional): Rotary embeddings slice.

        Returns:
            torch.Tensor: Output of shape (B, T, C).
        """
        B, T, C = x.shape

        # 1) Figure out how much past we have in cache
        if kv_cache is not None:
            b, nh, L, c = kv_cache.shape
            assert nh == self.num_heads and b == B and c * nh == C, "Cache dims mismatch."
        else:
            L = 0

        # 2) Project to heads
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        # 3) Apply rotary if needed
        if getattr(self.config, 'rotary_emb', False) and freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # 4) Update / retrieve cache
        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()  # shapes: (B, nh, L+T, head_dim)

        # 5) Raw dot-product scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, L+T)

        # 6) Base sliding-window band mask
        #    mask[i,j] = True iff |(L + i) - j| <= window
        base_mask = self.mask[L: L + T, : L + T]  # (T, L+T)
        mask = base_mask.unsqueeze(0).unsqueeze(1)  # (1,1,T,L+T)
        mask = mask.expand(B, self.num_heads, T, L + T).to(att.device)

        # 7) Carve out any *stale* cache slots for each sample
        if valid_ctx_len is not None:
            # valid_context_lengths[i] tells us how many of the
            # LAST L positions are actually real
            for i in range(B):
                valid = int(valid_ctx_len[i].item())
                stale = L - valid
                if stale > 0:
                    # zero out mask for any of those stale positions
                    mask[i, :, :, :stale] = False

        # 8) Apply mask, softmax, dropout
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # 9) Attend to values and re-project
        y = att @ v  # (B, nh, T, head_dim)
        y = rearrange(y, 'b h t d -> b t (h d)')
        return self.resid_drop(self.proj(y))

    @torch.no_grad()
    def get_attention_map(self,
                          x: torch.Tensor,
                          kv_cache: Optional[KeysValues] = None,
                          valid_context_lengths: Optional[torch.Tensor] = None
                          ) -> torch.Tensor:
        B, T, C = x.shape
        if kv_cache is not None:
            b, nh, L, c = kv_cache.shape
            kv_cache.update(*[t.view(B, nh, T, C // nh).transpose(1, 2)
                              for t in (self.query(x), self.key(x), self.value(x))][:2])
            k, v = kv_cache.get()
        else:
            L = 0

        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        # raw scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # local mask
        mask = self.mask[L:L + T, :L + T]
        mask = mask.unsqueeze(0).unsqueeze(1).expand(B, self.num_heads, T, L + T).to(att.device)
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)

        return att