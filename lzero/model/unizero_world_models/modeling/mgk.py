"""
Mixture of Gaussian Keys (MGK) Attention for Transformers.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .kv_caching import KeysValues
from .attention import Attention
from .transformer_config import TransformerConfig

class MGK(Attention):
    """
    Mixture-of-Kernels Attention for transformers.

    Each head models keys as an M-component Gaussian Mixture Model (GMM),
    with learned per-component means, variances, and mixture weights.
    """
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        assert config.embed_dim % config.num_heads == 0, \
            "Embedding dimension must be divisible by number of heads."

        # core dimensions
        self.config      = config
        self.num_heads   = config.num_heads
        self.head_dim    = config.embed_dim // config.num_heads
        self.num_mix     = getattr(config, 'num_mixtures', 4)
        self.max_len     = config.max_tokens

        # projections
        self.query       = nn.Linear(config.embed_dim, config.embed_dim)
        self.key_means   = nn.Linear(config.embed_dim, config.embed_dim * self.num_mix)
        self.value       = nn.Linear(config.embed_dim, config.embed_dim)
        self.proj        = nn.Linear(config.embed_dim, config.embed_dim)

        # dropout
        self.attn_drop   = nn.Dropout(config.attn_pdrop)
        self.resid_drop  = nn.Dropout(config.resid_pdrop)

        # Shared learnable bias
        self.key_bias = nn.Parameter(
            torch.zeros(self.num_heads, self.num_mix, self.head_dim)
        )

        # softplus
        inv_sp = lambda x: math.log(math.expm1(x))
        init_sigma = getattr(config, 'init_mgk_sigma', 1.0)

        # per-head, per-mixture raw variance parameters
        self.sigma_p = nn.Parameter(
            torch.full((self.num_heads, self.num_mix), inv_sp(init_sigma))
        )
        # per-head, raw mixture logits
        self.pi_p    = nn.Parameter(torch.zeros(self.num_heads, self.num_mix))

        # causal mask
        mask = torch.tril(torch.ones(self.max_len, self.max_len, dtype=torch.bool))
        self.register_buffer('causal_mask', mask)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KeysValues] = None,
        valid_ctx_len: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.size()
        device  = x.device

        # project Q, key base, V
        q      = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k_base = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        v      = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        # build mixture means: broadcast base + biases
        # k_base: (B, nh, T, hd); key_bias: (nh, M, hd)
        k_means = k_base.unsqueeze(3) + self.key_bias.unsqueeze(0).unsqueeze(2)
        # shape -> (B, nh, T, M, hd)

        # mixture parameters
        sigma = F.softplus(self.sigma_p)                           # (nh, M)
        pi    = F.softmax(self.pi_p, dim=-1)                       # (nh, M)

        # compute raw scores per mixture
        # expand sigma, pi for broadcasting
        sigma_e = sigma.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        pi_e    = pi.unsqueeze(0).unsqueeze(2).unsqueeze(2)

        # dot product q and k_means
        scores = (q.unsqueeze(3) * k_means).sum(-1)  # (B, nh, T, M)

        # normalize
        scores = scores / (sigma_e.squeeze(-2)**2)
        exp_s  = torch.exp(scores)   # (B, nh, T, M)
        weighted = exp_s * pi_e.squeeze(-2)   # weight by mixture
        num   = weighted.sum(-1)   # sum mixtures -> (B, nh, T)

        # initial attn over T keys
        attn = num / num.sum(-1, keepdim=True)
        # reshape to (B, nh, T, T)
        attn = attn.unsqueeze(-1).expand(-1, -1, -1, T)

        # handle cache length L
        if kv_cache is not None:
            k_old, _ = kv_cache.get()
            L = k_old.shape[2] - T
            kv_cache.update(k_base, v)
            _, v = kv_cache.get()
        else:
            L = 0
        total_len = L + T

        # pad for past positions
        if L > 0:
            pad = torch.zeros(B, self.num_heads, T, L, device=device, dtype=attn.dtype)
            attn = torch.cat([pad, attn], dim=-1)

        # causal & stale mask
        base = self.causal_mask[L:L+T, :L+T]
        if valid_ctx_len is not None:
            m = torch.zeros(B, T, total_len, dtype=torch.bool, device=device)
            for i in range(B):
                valid = int(valid_ctx_len[i].item())
                stale = L - valid
                sub   = base.clone()
                if stale > 0:
                    sub[:, :stale] = False
                m[i] = sub
            mask = m.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        else:
            mask = base.unsqueeze(0).unsqueeze(1).expand(B, self.num_heads, T, total_len)

        attn = attn.masked_fill(~mask, 0.0)
        attn = attn / (attn.sum(-1, keepdim=True) + 1e-6)

        # apply dropout and project
        y = self.attn_drop(attn) @ v                              # (B, nh, T, hd)
        y = rearrange(y, 'b h t d -> b t (h d)')
        return self.resid_drop(self.proj(y))

    @torch.no_grad()
    def get_attention_map(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KeysValues] = None,
        valid_context_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Raw MGK attention weights (B, nh, T, total_len) without updating cache.
        """
        B, T, C = x.size()
        device  = x.device

        # project Q, key base
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k_base = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        # mixture means
        k_means = k_base.unsqueeze(3) + self.key_bias.unsqueeze(0).unsqueeze(2)

        # mixture params
        sigma = F.softplus(self.sigma_p)
        pi = F.softmax(self.pi_p, dim=-1)
        sigma_e = sigma.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        pi_e = pi.unsqueeze(0).unsqueeze(2).unsqueeze(2)

        # raw mixture scores
        scores = (q.unsqueeze(3) * k_means).sum(-1)
        scores = scores / (sigma_e.squeeze(-2)**2)
        exp_s = torch.exp(scores)
        weighted = exp_s * pi_e.squeeze(-2)
        num = weighted.sum(-1)
        attn = num / num.sum(-1, keepdim=True)

        # handle cache
        if kv_cache is not None:
            k_old, _ = kv_cache.get()
            L = k_old.shape[2] - T
        else:
            L = 0
        total_len = L + T
        if L > 0:
            pad = torch.zeros(B, self.num_heads, T, L, device=device, dtype=attn.dtype)
            attn = torch.cat([pad, attn], dim=-1)

        # mask
        base = self.causal_mask[L:L+T, :L+T]
        if valid_context_lengths is not None:
            m = torch.zeros(B, T, total_len, dtype=torch.bool, device=device)
            for i in range(B):
                valid = int(valid_context_lengths[i].item())
                stale = L - valid
                sub   = base.clone()
                if stale > 0:
                    sub[:, :stale] = False
                m[i] = sub
            mask = m.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        else:
            mask = base.unsqueeze(0).unsqueeze(1).expand(B, self.num_heads, T, total_len)
        attn = attn.masked_fill(~mask, 0.0)
        attn = attn / (attn.sum(-1, keepdim=True) + 1e-6)

        return attn
