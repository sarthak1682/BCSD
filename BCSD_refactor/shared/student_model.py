"""Student distillation model architecture modules."""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


class StudentDistillationModule(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers=2, pad_id=0, enable_nested_tensor=False):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=1024)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers, 
            enable_nested_tensor=enable_nested_tensor
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids, key_padding_mask=None):
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)

        if key_padding_mask is None:
            key_padding_mask = (input_ids == self.pad_id)

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return self.out_proj(x)


# Adapted from the official NV-Embed LatentAttentionModel implementation.
# Paper: "NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models"
#        Lee et al., 2024 — https://arxiv.org/abs/2405.17428
# Source: https://huggingface.co/nvidia/NV-Embed-v2/blob/main/modeling_nvembed.py
#         (class LatentAttentionModel)
# Changes: uses nn.MultiheadAttention instead of a custom Attention module;
#          GELU + 4x expansion instead of GEGLU + 8x; adapted for our student pipeline.
class LatentAttentionLayer(nn.Module):
    """Distills sequence into a single embedding via NV-Embed-style cross-attention.
    Q=sequence, K/V=latents; Pre-LN on both Q and context, residuals after
    attention and feedforward, then masked mean-pool over the sequence.

    Args:
        hidden_states: [B, S, D] sequence output from the encoder.
        key_padding_mask: [B, S] bool — True for padding positions. Used only
            for mean pooling if pool_mask is not provided.
        pool_mask: [B, S] bool — True for tokens to EXCLUDE from mean pooling
            (padding + instruction prefix). When provided, takes precedence over
            key_padding_mask for the pooling step. Mirrors NV-Embed's pool_mask
            that zeroes out instruction token positions before mean pooling.
    """
    def __init__(self, hidden_dim, num_latents=512, num_heads=8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim))

        # Pre-LN: separate norms for query (sequence) and context (latents)
        self.norm_seq = nn.LayerNorm(hidden_dim)
        self.norm_latents = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)

        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, hidden_states, key_padding_mask=None, pool_mask=None):
        batch_size = hidden_states.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        # Pre-LN on both Q (sequence) and K/V (latents)
        normed_seq = self.norm_seq(hidden_states)
        normed_latents = self.norm_latents(latents)

        # Cross-attention: Q=sequence, K/V=latents — no mask needed on K/V (latents have no padding)
        attn_output, _ = self.cross_attn(query=normed_seq, key=normed_latents, value=normed_latents)
        hidden_states = hidden_states + attn_output  # Residual 1

        # Pre-LN + FeedForward + Residual
        hidden_states = hidden_states + self.mlp(self.norm_ff(hidden_states))  # Residual 2

        # Masked mean pool — pool_mask excludes padding AND instruction prefix tokens;
        # falls back to key_padding_mask (padding only) if pool_mask is not supplied.
        exclude_mask = pool_mask if pool_mask is not None else key_padding_mask
        if exclude_mask is not None:
            mask = (~exclude_mask).unsqueeze(-1).to(hidden_states.dtype)  # [B, S, 1]
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return hidden_states.mean(dim=1)
