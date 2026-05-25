"""Deduplicated loss functions for contrastive and distillation training tasks."""

import torch
import torch.nn.functional as F


def contrastive_loss(embeddings, temperature=0.05):
    """Basic InfoNCE loss for contrastive learning."""
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    batch_size = embeddings.shape[0]

    labels = torch.arange(batch_size, device=embeddings.device)
    labels[::2] += 1
    labels[1::2] -= 1

    mask_self = torch.eye(batch_size, device=embeddings.device).bool()
    sim_matrix.masked_fill_(mask_self, -1e9)

    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def contrastive_loss_positive_aware(embeddings, func_ids, temperature=0.05):
    """Contrastive loss that masks out same-function false negatives.

    Compatible with both python lists/sequences of IDs and PyTorch tensors.
    """
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    batch_size = embeddings.shape[0]

    labels = torch.arange(batch_size, device=embeddings.device)
    labels[::2] += 1
    labels[1::2] -= 1

    if not isinstance(func_ids, torch.Tensor):
        unique_ids = list(set(func_ids))
        id_map = {uid: i for i, uid in enumerate(unique_ids)}
        batch_numeric_ids = [id_map[fid] for fid in func_ids]
        func_ids_tensor = torch.tensor(batch_numeric_ids, device=embeddings.device)
    else:
        func_ids_tensor = func_ids.to(embeddings.device)

    id_match_matrix = func_ids_tensor.unsqueeze(1) == func_ids_tensor.unsqueeze(0)
    mask_ignore = torch.eye(batch_size, device=embeddings.device).bool() | id_match_matrix
    sim_matrix.masked_fill_(mask_ignore, -1e9)

    mask_pos = torch.zeros_like(sim_matrix, dtype=torch.bool)
    mask_pos.scatter_(1, labels.unsqueeze(1), True)
    pos_scores = (embeddings * embeddings[labels]).sum(dim=1) / temperature
    sim_matrix[mask_pos] = pos_scores

    return F.cross_entropy(sim_matrix, labels)


def cgte_loss(x_emb, y_emb, temperature=0.05):
    """Exact implementation of Equation 5 from L1NNA/binary_sim."""
    scale = 1 / temperature
    batch_size = x_emb.size(0)
    labels = torch.arange(batch_size, device=x_emb.device)
    
    xiy = F.cosine_similarity(x_emb.unsqueeze(1), y_emb.unsqueeze(0), dim=2) * scale
    yix = F.cosine_similarity(y_emb.unsqueeze(1), x_emb.unsqueeze(0), dim=2) * scale
    yix[labels, labels] = -torch.inf
    
    xix = F.cosine_similarity(x_emb.unsqueeze(1), x_emb.unsqueeze(0), dim=2) * scale
    xix[labels, labels] = -torch.inf
    
    yiy = F.cosine_similarity(y_emb.unsqueeze(1), y_emb.unsqueeze(0), dim=2) * scale
    yiy[labels, labels] = -torch.inf
    
    similarities = torch.cat([xiy, yix, xix, yiy], dim=1)
    return F.cross_entropy(similarities, labels)


def masked_mse_loss(student_hidden, teacher_hidden, key_padding_mask):
    """Masked Mean Squared Error loss for distilling sequence outputs."""
    valid = (~key_padding_mask).unsqueeze(-1)
    diff = (student_hidden - teacher_hidden) ** 2
    diff = diff * valid
    denom = valid.sum().clamp_min(1.0)
    return diff.sum() / denom
