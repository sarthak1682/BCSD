from __future__ import annotations

import torch
import torch.nn.functional as F


def shifted_causal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must have matching [B,L] dimensions")
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels.ne(-100)
    if not torch.any(valid):
        return shifted_logits.sum() * 0.0
    return F.cross_entropy(shifted_logits[valid].float(), shifted_labels[valid])


def shifted_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    if not torch.any(valid):
        return 0, 0
    predictions = shifted_logits.argmax(dim=-1)
    correct = predictions[valid].eq(shifted_labels[valid]).sum().item()
    return int(correct), int(valid.sum().item())


@torch.no_grad()
def retrieval_recall_at_1(
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    query_ids: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    chunk_size: int = 512,
    compute_device: torch.device | str | None = None,
) -> float:
    """Compute identity-aware global R@1 without materializing the full matrix."""

    if query_embeddings.ndim != 2 or target_embeddings.ndim != 2:
        raise ValueError("embeddings must be matrices")
    if query_embeddings.shape[1] != target_embeddings.shape[1]:
        raise ValueError("query and target embedding dimensions differ")
    if query_embeddings.shape[0] != query_ids.numel():
        raise ValueError("query IDs do not align with embeddings")
    if target_embeddings.shape[0] != target_ids.numel():
        raise ValueError("target IDs do not align with embeddings")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if query_embeddings.shape[0] == 0 or target_embeddings.shape[0] == 0:
        return 0.0

    if compute_device is not None:
        query_embeddings = query_embeddings.to(compute_device)
        target_embeddings = target_embeddings.to(compute_device)
        query_ids = query_ids.to(compute_device)
        target_ids = target_ids.to(compute_device)

    queries = F.normalize(query_embeddings.float(), p=2, dim=1)
    targets = F.normalize(target_embeddings.float(), p=2, dim=1)
    correct = 0
    for start in range(0, queries.shape[0], chunk_size):
        end = min(start + chunk_size, queries.shape[0])
        nearest = (queries[start:end] @ targets.T).argmax(dim=1)
        correct += query_ids[start:end].eq(target_ids[nearest]).sum().item()
    return float(correct / query_embeddings.shape[0])
