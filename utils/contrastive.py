"""Batch construction and symmetric multi-positive InfoNCE."""
import torch
from torch.nn import functional as F


def alignment_batch(rows, meta):
    indices, domains, valid_text, text = zip(*rows)
    ids = torch.full((len(rows), max(1, max(map(len, domains)))), -1, dtype=torch.long)
    inputs = torch.zeros(len(rows), len(meta['domains']))
    for i, values in enumerate(domains):
        ids[i, :len(values)] = torch.tensor(values, dtype=torch.long)
        inputs[i, values] = 1
    return dict(index=torch.tensor(indices), domain=ids, domain_input=inputs,
                text_valid=torch.stack(valid_text), text_input=torch.stack(text))


def multi_positive(a, b, groups, temperature):
    """Identical domain sets or text embeddings are multiple positives."""
    if len(groups) < 2 or len(set(groups)) < 2:
        return (a.sum() + b.sum()) * 0
    positive = torch.tensor([[x == y for y in groups] for x in groups], device=a.device)
    logits = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T / temperature
    def loss(z, mask):
        targets = mask.float() / mask.sum(1, keepdim=True)
        return -(targets * z.log_softmax(1)).sum(1).mean()
    return (loss(logits, positive) + loss(logits.T, positive.T)) / 2

