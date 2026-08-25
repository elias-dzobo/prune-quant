from collections.abc import Callable

import torch
from torch.utils.data import DataLoader
from torchprofile import profile_macs

from src.models.bundle import TaskType
from src.types import SensitivityScanResult

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


def get_model_macs(model: torch.nn.Module, inputs: torch.Tensor) -> int:
    return profile_macs(model, inputs)


def get_tensor_sparsity(tensor: torch.Tensor) -> float:
    return 1.0 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: torch.nn.Module) -> float:
    num_nonzeros, num_elements = 0, 0

    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()

    return 1.0 - float(num_nonzeros) / num_elements


def _iter_sparsity_levels(
    sparsity_levels: list[float] | None,
    scan_start: float,
    scan_end: float,
    scan_step: float,
) -> list[float]:
    """Build the list of sparsity values to evaluate during the sensitivity scan."""
    if sparsity_levels is not None:
        return sorted({min(max(s, 0.0), 1.0) for s in sparsity_levels})

    levels: list[float] = []
    sparsity = scan_start
    while sparsity <= scan_end + 1e-9:
        levels.append(min(max(sparsity, 0.0), 1.0))
        sparsity += scan_step
    return levels


def _fine_grained_prune_mask(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    Build a binary mask for magnitude-based fine-grained pruning.

    Weights with the smallest absolute values are zeroed out until the target
    sparsity is reached. Returns 1 for weights to keep and 0 for weights to prune.
    """
    sparsity = min(max(sparsity, 0.0), 1.0)
    if sparsity == 0.0:
        return torch.ones_like(tensor)
    if sparsity == 1.0:
        return torch.zeros_like(tensor)

    num_elements = tensor.numel()
    num_zeros = int(num_elements * sparsity)
    if num_zeros == 0:
        return torch.ones_like(tensor)
    if num_zeros >= num_elements:
        return torch.zeros_like(tensor)

    importance = torch.abs(tensor)
    threshold = importance.reshape(-1).kthvalue(num_zeros).values
    return (importance > threshold).to(dtype=tensor.dtype)


def _default_evaluate_accuracy(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute classification accuracy (%) on a single batch.

    Assumes logits of shape (batch, num_classes) and integer class targets.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        if outputs.ndim > 1:
            outputs = outputs.argmax(dim=1)
        num_correct = (outputs == targets).sum().item()
    return 100.0 * num_correct / targets.size(0)


def _select_best_sparsity(
    sparsity_levels: list[float],
    scores: list[float],
    *,
    higher_is_better: bool = True,
) -> float:
    """
    Pick the best sparsity from a sensitivity scan.

    For classification, higher scores are better (accuracy %).
    For perplexity, lower scores are better.
    Ties prefer the highest sparsity when compression is equal.
    """
    if higher_is_better:
        target_score = max(scores)
        is_better = lambda score, best: score == best
    else:
        target_score = min(scores)
        is_better = lambda score, best: score == best

    best_sparsity = sparsity_levels[0]
    for sparsity, score in zip(sparsity_levels, scores):
        if is_better(score, target_score) and sparsity >= best_sparsity:
            best_sparsity = sparsity
    return best_sparsity


def generate_sparsity_dict(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    sparsity_levels: list[float] | None = None,
    scan_start: float = 0.0,
    scan_end: float = 1.0,
    scan_step: float = 0.1,
    evaluate_fn: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], float]
    | None = None,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Build a per-parameter sparsity config via layer-wise sensitivity scanning.

    See ``sensitivity_scan`` for the full scan details. This helper returns only
    the selected sparsity for each layer.
    """
    _, sparsity_dict = sensitivity_scan(
        model=model,
        inputs=inputs,
        targets=targets,
        sparsity_levels=sparsity_levels,
        scan_start=scan_start,
        scan_end=scan_end,
        scan_step=scan_step,
        evaluate_fn=evaluate_fn,
        verbose=verbose,
    )
    return sparsity_dict


def sensitivity_scan(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    sparsity_levels: list[float] | None = None,
    scan_start: float = 0.0,
    scan_end: float = 1.0,
    scan_step: float = 0.1,
    evaluate_fn: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], float]
    | None = None,
    verbose: bool = False,
    higher_is_better: bool = True,
    metric_label: str = "accuracy",
) -> tuple[SensitivityScanResult, dict[str, float]]:
    """
    Run a layer-wise pruning sensitivity scan and build a sparsity dict.

    For each prunable weight tensor (parameters with ``dim() > 1``), only that
    layer is pruned at each candidate sparsity while accuracy is measured on the
    provided batch. The original weights are restored after every trial.

    Returns:
        A tuple of (scan_result, sparsity_dict). ``scan_result`` contains the
        full accuracy curves used for sensitivity plots. ``sparsity_dict`` maps
        each parameter name to the sparsity with the highest recorded accuracy.
    """
    if evaluate_fn is None:
        evaluate_fn = _default_evaluate_accuracy

    device = next(model.parameters()).device
    inputs = inputs.to(device)
    targets = targets.to(device)

    levels = _iter_sparsity_levels(sparsity_levels, scan_start, scan_end, scan_step)
    sparsity_dict: dict[str, float] = {}
    layer_names: list[str] = []
    all_accuracies: list[list[float]] = []

    dense_accuracy = evaluate_fn(model, inputs, targets)

    prunable_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.dim() > 1
    ]

    for layer_index, (name, param) in enumerate(prunable_params):
        param_backup = param.detach().clone()
        layer_accuracies: list[float] = []

        for sparsity in levels:
            mask = _fine_grained_prune_mask(param_backup, sparsity)
            param.data.copy_(param_backup * mask)
            accuracy = evaluate_fn(model, inputs, targets)
            layer_accuracies.append(accuracy)

            if verbose:
                print(
                    f"[{layer_index + 1}/{len(prunable_params)}] {name} "
                    f"sparsity={sparsity:.2f} {metric_label}={accuracy:.2f}"
                )

        param.data.copy_(param_backup)

        layer_names.append(name)
        all_accuracies.append(layer_accuracies)
        sparsity_dict[name] = _select_best_sparsity(
            levels,
            layer_accuracies,
            higher_is_better=higher_is_better,
        )

        if verbose:
            best_score = max(layer_accuracies) if higher_is_better else min(layer_accuracies)
            print(
                f"Selected {name}: sparsity={sparsity_dict[name]:.2f} "
                f"(best {metric_label}={best_score:.2f})"
            )

    scan_result = SensitivityScanResult(
        layer_names=layer_names,
        sparsity_levels=levels,
        accuracies=all_accuracies,
        dense_accuracy=dense_accuracy,
        metric_label=metric_label,
        higher_is_better=higher_is_better,
    )
    return scan_result, sparsity_dict


def sensitivity_scan_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    sparsity_levels: list[float] | None = None,
    scan_start: float = 0.4,
    scan_end: float = 1.0,
    scan_step: float = 0.1,
    task: TaskType = "classification",
    verbose: bool = False,
) -> tuple[SensitivityScanResult, dict[str, float]]:
    """Sensitivity scan using full-dataloader quality metric for each trial."""
    from src.eval import evaluate_quality

    higher_is_better = task == "classification"
    metric_label = "accuracy" if higher_is_better else "perplexity"

    def evaluate_fn(
        eval_model: torch.nn.Module,
        _: torch.Tensor,
        __: torch.Tensor,
    ) -> float:
        return evaluate_quality(eval_model, dataloader, device, task=task)

    sample_batch = next(iter(dataloader))
    if isinstance(sample_batch, dict):
        sample_inputs = sample_batch["input_ids"]
        sample_targets = sample_batch.get("labels", sample_batch["input_ids"])
    else:
        sample_inputs, sample_targets = sample_batch

    return sensitivity_scan(
        model=model,
        inputs=sample_inputs,
        targets=sample_targets,
        sparsity_levels=sparsity_levels,
        scan_start=scan_start,
        scan_end=scan_end,
        scan_step=scan_step,
        evaluate_fn=evaluate_fn,
        verbose=verbose,
        higher_is_better=higher_is_better,
        metric_label=metric_label,
    )
