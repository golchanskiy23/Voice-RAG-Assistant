import torch


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def cls_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return hidden[:, 0]


def normalize_rows(vectors: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(vectors.float(), p=2, dim=1)


def pad_columns(vectors: torch.Tensor, dim: int) -> torch.Tensor:
    if vectors.shape[1] > dim:
        raise ValueError(f"размерность {vectors.shape[1]} больше целевой {dim}")
    return torch.nn.functional.pad(vectors, (0, dim - vectors.shape[1]))
