import math
import time

import torch
from torch.utils.data import DataLoader

from src.models.bundle import TaskType
from src.types import EvalMetrics
from src.utils.utils import Byte, get_model_sparsity


def _sample_batch(dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch = next(iter(dataloader))
    if isinstance(batch, dict):
        inputs = batch["input_ids"]
        targets = batch.get("labels", batch["input_ids"])
        return inputs, targets
    inputs, targets = batch
    return inputs, targets


def evaluate_accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Classification accuracy (%) over the full dataloader."""
    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            if outputs.ndim > 1:
                outputs = outputs.argmax(dim=1)
            num_correct += (outputs == targets).sum().item()
            num_samples += targets.size(0)

    if num_samples == 0:
        return 0.0
    return 100.0 * num_correct / num_samples


def evaluate_perplexity(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Mean token perplexity over the dataloader (lower is better)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if not isinstance(batch, dict):
                raise ValueError("Text-generation evaluation expects dict batches with input_ids.")

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            labels = input_ids.clone()
            if attention_mask is not None:
                labels = labels.masked_fill(attention_mask == 0, -100)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            num_batches += 1

    if num_batches == 0:
        return float("inf")
    return math.exp(total_loss / num_batches)


def evaluate_quality(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    task: TaskType = "classification",
) -> float:
    """Return task-appropriate quality metric (accuracy % or perplexity)."""
    if task == "classification":
        return evaluate_accuracy(model, dataloader, device)
    return evaluate_perplexity(model, dataloader, device)


def measure_latency_ms(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    task: TaskType = "classification",
    warmup: int = 10,
    runs: int = 50,
) -> float:
    """Average forward-pass latency in milliseconds for one batch."""
    model.eval()
    sample_inputs, _ = _sample_batch(dataloader)
    sample_inputs = sample_inputs.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            if task == "text_generation":
                model(input_ids=sample_inputs)
            else:
                model(sample_inputs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            if task == "text_generation":
                model(input_ids=sample_inputs)
            else:
                model(sample_inputs)

        if device.type == "cuda":
            torch.cuda.synchronize()

    return (time.perf_counter() - start) / runs * 1000.0


def estimate_memory_mib(
    model: torch.nn.Module,
    bits_per_weight: int = 32,
    use_effective_sparsity: bool = False,
) -> float:
    """
    Estimate model weight memory in MiB.

    When ``use_effective_sparsity`` is True, pruned zeros are not counted
    toward storage (theoretical compressed size).
    """
    num_bytes = 0
    for param in model.parameters():
        if use_effective_sparsity:
            num_values = param.count_nonzero().item()
        else:
            num_values = param.numel()
        num_bytes += num_values * bits_per_weight / Byte

    return num_bytes / (1024 * 1024)


def precision_to_bits(precision: str) -> int:
    """Map a precision label to its storage bit width."""
    mapping = {
        "fp32": 32,
        "fp16": 16,
        "bf16": 16,
        "fp8": 8,
        "fp4": 4,
        "int8": 8,
        "int4": 4,
    }
    if precision not in mapping:
        raise ValueError(f"Unknown precision '{precision}'. Expected one of {list(mapping)}.")
    return mapping[precision]


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label: str,
    *,
    task: TaskType = "classification",
    bits_per_weight: int = 32,
    use_effective_sparsity: bool = False,
    precision: str = "fp32",
    warmup: int = 10,
    latency_runs: int = 50,
) -> EvalMetrics:
    """Run quality, memory, and latency evaluation for one model variant."""
    metric_label = "accuracy" if task == "classification" else "perplexity"

    return EvalMetrics(
        label=label,
        accuracy=evaluate_quality(model, dataloader, device, task=task),
        memory_mib=estimate_memory_mib(
            model,
            bits_per_weight=bits_per_weight,
            use_effective_sparsity=use_effective_sparsity,
        ),
        latency_ms=measure_latency_ms(
            model,
            dataloader,
            device,
            task=task,
            warmup=warmup,
            runs=latency_runs,
        ),
        sparsity=get_model_sparsity(model) if use_effective_sparsity else 0.0,
        precision=precision,
        metric_label=metric_label,
    )
