"""Batch helpers shared by calibration and training."""

from collections.abc import Mapping
from typing import Any, Callable, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def move_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(b, device) for b in batch)
    if isinstance(batch, Mapping):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    return batch


def batch_size(batch: Any) -> int:
    if isinstance(batch, torch.Tensor):
        return batch.shape[0]
    if isinstance(batch, (list, tuple)):
        return batch_size(batch[0])
    if isinstance(batch, Mapping):
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v.shape[0]
    raise ValueError("Cannot infer batch size")


def slice_batch(batch: Any, start: int, stop: int) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch[start:stop]
    if isinstance(batch, (list, tuple)):
        return type(batch)(slice_batch(b, start, stop) for b in batch)
    if isinstance(batch, Mapping):
        return {k: slice_batch(v, start, stop) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return batch


def iter_samples(batch: Any) -> Iterator[Any]:
    for i in range(batch_size(batch)):
        yield slice_batch(batch, i, i + 1)


def _logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (list, tuple)):
        return output[0]
    if isinstance(output, dict):
        return output["logits"]
    raise ValueError("Cannot extract logits from model output")


def default_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for classification / token-level targets, MSE for float targets."""
    if targets.is_floating_point():
        return F.mse_loss(logits.squeeze(-1), targets)
    if logits.dim() == 3 and targets.dim() == 2:
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
    return F.cross_entropy(logits, targets)


def forward_loss(
    model: nn.Module,
    batch: Any,
    loss_fn: Optional[LossFn] = None,
) -> Tuple[torch.Tensor, Any]:
    """
    Run the model on a batch and return ``(loss, output)``.

    Supported batches: ``dict`` (Hugging Face style, uses ``output.loss`` when
    present), ``(inputs, targets)`` tuples and plain input tensors.  The loss
    is averaged over the batch.
    """
    loss_fn = loss_fn or default_loss
    if isinstance(batch, Mapping):
        output = model(**batch)
        loss = getattr(output, "loss", None)
        if loss is None and isinstance(output, dict):
            loss = output.get("loss")
        if loss is None:
            labels = batch.get("labels")
            if labels is None:
                raise ValueError("Batch dict has no labels and the model returned no loss")
            loss = loss_fn(_logits(output), labels)
        return loss, output
    if isinstance(batch, (list, tuple)):
        inputs, targets = batch[0], batch[1] if len(batch) > 1 else None
        output = model(inputs)
        if targets is None:
            raise ValueError("Unsupervised batches require a custom loss function")
        return loss_fn(_logits(output), targets), output
    raise ValueError(f"Unsupported batch type {type(batch)}")


def output_logits(output: Any) -> torch.Tensor:
    return _logits(output)
