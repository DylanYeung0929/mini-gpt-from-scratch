"""A small GPT language model implemented from scratch.

Components: token embedding, RoPE, causally-masked multi-head attention
(with optional grouped-query attention), RMSNorm, SwiGLU MLP, weight tying.
"""
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        n = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(n + self.eps) * self.weight


def precompute_rope(head_dim: int, max_seq: int, theta: float = 10000.0):
    """Return cos/sin tables of shape (max_seq, head_dim // 2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)          # (seq, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, D). cos/sin: (T, D/2)."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout
        cos, sin = precompute_rope(self.head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        offset = past_kv[0].shape[2] if past_kv is not None else 0
        cos = self.rope_cos[offset: offset + T]
        sin = self.rope_sin[offset: offset + T]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v) if use_cache else None

        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(past_kv is None),
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y), new_kv


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(cfg.ffn_mult * cfg.n_embd)
        hidden = 64 * ((hidden + 63) // 64)
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, past_kv=None, use_cache=False):
        a, new_kv = self.attn(self.norm1(x), past_kv, use_cache)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.grad_checkpointing = False
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 style scaled init for residual output projections
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_weights:
                n -= self.lm_head.weight.numel()
        return n

    def forward(self, idx, targets=None, past_kvs=None, use_cache=False):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        new_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            pkv = past_kvs[i] if past_kvs is not None else None
            if self.grad_checkpointing and self.training and pkv is None:
                x, kv = checkpoint(block, x, None, False, use_reentrant=False)
            else:
                x, kv = block(x, pkv, use_cache)
            if use_cache:
                new_kvs.append(kv)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss, new_kvs

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None, top_p=None,
                 repetition_penalty=1.0, eos_id=None):
        """Autoregressive sampling with KV cache."""
        from .generate import sample_next_token

        max_ctx = self.cfg.block_size
        if idx.shape[1] >= max_ctx:
            warnings.warn(f"prompt has {idx.shape[1]} tokens but block_size is {max_ctx}; "
                          f"keeping the last {max_ctx - 1}")
            idx = idx[:, -(max_ctx - 1):]
        if idx.shape[1] + max_new_tokens > max_ctx:
            max_new_tokens = max_ctx - idx.shape[1]
            warnings.warn(f"capping generation to {max_new_tokens} tokens to fit block_size")
        if max_new_tokens <= 0:
            return idx

        past_kvs = None
        for _ in range(max_new_tokens):
            if past_kvs is None:
                cond = idx
            else:
                cond = idx[:, -1:]
            logits, _, past_kvs = self(cond, past_kvs=past_kvs, use_cache=True)
            logits = logits[:, -1, :]
            if repetition_penalty != 1.0:
                for b in range(idx.shape[0]):
                    seen = idx[b].tolist()
                    for tok in set(seen):
                        if logits[b, tok] > 0:
                            logits[b, tok] /= repetition_penalty
                        else:
                            logits[b, tok] *= repetition_penalty
            next_tok = sample_next_token(logits, temperature, top_k, top_p)
            idx = torch.cat([idx, next_tok[:, None]], dim=1)
            if eos_id is not None and (next_tok == eos_id).all():
                break
        return idx
