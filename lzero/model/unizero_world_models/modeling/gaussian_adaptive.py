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


class GaussianAdaptiveSpanAttention(Attention):
    """
    Combines Adaptive Span (triangular soft mask) with per-head
    Gaussian relative-position bias. Outside the learned span,
    attention weights are zero.

    Part of conference submission: "Learning to Focus: Prioritizing Informative Histories with Structured Attention
 Mechanisms in Partially Observable Reinforcement Learning"
    """
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        assert config.embed_dim % config.num_heads == 0, \
            "Embedding dimension must be divisible by number of heads."

        self.config    = config
        self.num_heads = config.num_heads
        self.head_dim  = config.embed_dim // config.num_heads
        self.max_len   = config.max_tokens

        # projections
        self.key   = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)

        # dropout + output projection
        self.attn_drop  = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj       = nn.Linear(config.embed_dim, config.embed_dim)

        # Adaptive-span params (softplus domain)
        init_span    = config.init_adaptive_span or config.max_tokens
        inv_softplus = lambda x: math.log(math.expm1(x))
        self.span_p  = nn.Parameter(torch.full((self.num_heads,), inv_softplus(init_span)))

        # Gaussian params (softplus domain)
        init_sigma   = getattr(config, 'init_adaptive_sigma', 1.0)
        init_mu      = getattr(config, 'init_adaptive_mu', 0.0)
        self.sigma_p  = nn.Parameter(torch.full((self.num_heads,), inv_softplus(init_sigma)))
        self.mu_p_raw = nn.Parameter(torch.full((self.num_heads,), inv_softplus(init_mu)))

        # precompute causal mask
        causal = torch.tril(torch.ones(self.max_len, self.max_len, dtype=torch.bool))
        self.register_buffer('causal_mask', causal)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KeysValues] = None,
        valid_ctx_len: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.size()
        device  = x.device

        # project Q, K, V
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = self.key(x)  .view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        # rotary embeddings
        if getattr(self.config, 'rotary_emb', False) and freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # update cache
        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()
            L = k.shape[2] - T
        else:
            L = 0
        total_len = L + T

        # raw scores
        scores = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(self.head_dim))

        # build causal + stale mask
        base = self.causal_mask[L:L+T, :L+T]
        if valid_ctx_len is not None:
            mask = torch.zeros(B, T, total_len, dtype=torch.bool, device=device)
            for i in range(B):
                valid = int(valid_ctx_len[i])
                stale = L - valid
                sub   = base.clone()
                if stale > 0:
                    sub[:, :stale] = False
                mask[i] = sub
        else:
            mask = base.unsqueeze(0).expand(B, T, total_len)
        mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # positional distances
        qpos = torch.arange(L, L+T, device=device).unsqueeze(1)  # (T,1)
        kpos = torch.arange(0, total_len, device=device).unsqueeze(0)  # (1,total_len)
        dist = (qpos - kpos).abs()  # (T, total_len)

        # triangular span gating
        span   = F.softplus(self.span_p).view(1, self.num_heads, 1, 1)
        tri_w  = (1.0 - dist.unsqueeze(0).unsqueeze(1) / span).clamp(min=0.0)
        tri_w  = tri_w.masked_fill(~mask, 0.0)

        # Gaussian relative-position bias
        sigma  = F.softplus(self.sigma_p).view(1, self.num_heads, 1, 1)
        mu     = F.softplus(self.mu_p_raw).clamp(max=self.max_len).view(1, self.num_heads, 1, 1)
        dist_h = dist.unsqueeze(0).unsqueeze(2).expand(B, self.num_heads, T, total_len)
        gauss  = torch.exp(-((dist_h - mu)**2) / (2 * sigma**2))
        gauss  = gauss.masked_fill(~mask, 0.0)

        # combine masks
        mask_weights = tri_w * gauss

        # apply gating & bias, then softmax
        scores = scores * mask_weights
        attn   = F.softmax(scores, dim=-1)
        attn   = self.attn_drop(attn)

        # compute output
        y = attn @ v
        y = rearrange(y, 'b h t d -> b t (h d)')
        return self.resid_drop(self.proj(y))

    @torch.no_grad()
    def get_attention_map(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KeysValues] = None,
        valid_ctx_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns the (B, nh, T, total_len) attention weights
        after combining triangular span and Gaussian masks.
        """
        B, T, C = x.shape
        device  = x.device

        # project Q, K
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        if kv_cache is not None:
            k, _ = kv_cache.get()
            L    = k.shape[2] - T
        else:
            k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
            L = 0
        total_len = L + T

        # raw scores
        scores = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(self.head_dim))

        # causal + stale mask (same as forward)
        base = self.causal_mask[L:L+T, :L+T]
        if valid_ctx_len is not None:
            mask = torch.zeros(B, T, total_len, dtype=torch.bool, device=device)
            for i in range(B):
                valid = int(valid_ctx_len[i])
                stale = L - valid
                sub   = base.clone()
                if stale > 0:
                    sub[:, :stale] = False
                mask[i] = sub
        else:
            mask = base.unsqueeze(0).expand(B, T, total_len)
        mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # distances
        qpos = torch.arange(L, L+T, device=device).unsqueeze(1)
        kpos = torch.arange(0, total_len, device=device).unsqueeze(0)
        dist = (qpos - kpos).abs()

        # triangular span gating
        span  = F.softplus(self.span_p).view(1, self.num_heads, 1, 1)
        tri_w = (1.0 - dist.unsqueeze(0).unsqueeze(2) / span).clamp(min=0.0)
        tri_w = tri_w.masked_fill(~mask, 0.0)

        # Gaussian bias
        sigma  = F.softplus(self.sigma_p).view(1, self.num_heads, 1, 1)
        mu     = F.softplus(self.mu_p_raw).clamp(max=self.max_len).view(1, self.num_heads, 1, 1)
        dist_h = dist.unsqueeze(0).unsqueeze(2).expand(B, self.num_heads, T, total_len)
        gauss  = torch.exp(-((dist_h - mu)**2) / (2 * sigma**2))
        gauss  = gauss.masked_fill(~mask, 0.0)

        # combine, softmax
        mask_weights = tri_w * gauss
        scores = scores * mask_weights
        attn   = F.softmax(scores, dim=-1)
        return attn
