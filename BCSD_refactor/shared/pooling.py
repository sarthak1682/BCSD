"""Shared pooling modules for embedding extraction."""

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, hidden_states, label_positions):
        """Pools sequence hidden states into a single embedding based on label positions.

        Args:
            hidden_states: [B, L, D] tensor.
            label_positions: List of lists containing valid indices for pooling.

        Returns:
            [B, D] pooled embeddings.
        """
        B, L, D = hidden_states.shape
        device = hidden_states.device
        
        attn_scores = self.attention(hidden_states).squeeze(-1)
        
        mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        for i, pos_list in enumerate(label_positions):
            valid_pos = [p for p in pos_list if p < L]
            if valid_pos:
                mask[i, valid_pos] = True
            else:
                mask[i, :] = True
                
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        
        pooled_outputs = torch.sum(hidden_states * attn_weights, dim=1)
        return pooled_outputs
