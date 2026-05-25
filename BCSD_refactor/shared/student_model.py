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
        return x + self.pe[:, :x.size(1), pe_slice_or_pad_is_not_needed_for_x_device].to(x.device) if hasattr(self, 'pe_slice_or_pad_is_not_needed_for_x_device') else x + self.pe[:, :x.size(1), :].to(x.device)


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


class LatentAttentionLayer(nn.Module):
    """Distills sequence into a single embedding via Perceiver-style cross-attention."""
    def __init__(self, hidden_dim, num_latents=512, num_heads=8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim))

        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, hidden_states, key_padding_mask=None):
        batch_size = hidden_states.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        normed_latents = self.attn_norm(latents)
        attn_output, _ = self.cross_attn(
            query=normed_latents,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask
        )
        latents = latents + attn_output

        mlp_out = self.mlp(self.mlp_norm(latents))
        latents = latents + mlp_out

        return latents.mean(dim=1)
