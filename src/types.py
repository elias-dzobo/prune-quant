from dataclasses import dataclass, field

import torch


@dataclass
class EvalMetrics:
    """Quality, memory, and latency metrics for a single model variant."""

    label: str
    accuracy: float
    memory_mib: float
    latency_ms: float
    sparsity: float = 0.0
    precision: str = "fp32"
    metric_label: str = "accuracy"


@dataclass
class SensitivityScanResult:
    """Layer-wise quality recorded at each sparsity level."""

    layer_names: list[str] = field(default_factory=list)
    sparsity_levels: list[float] = field(default_factory=list)
    accuracies: list[list[float]] = field(default_factory=list)
    dense_accuracy: float = 0.0
    metric_label: str = "accuracy"
    higher_is_better: bool = True


@dataclass
class PipelineResult:
    """Full output from a prune-quant compression run."""

    base: EvalMetrics
    pruned: EvalMetrics
    quantized: EvalMetrics
    pruned_quantized: EvalMetrics
    sensitivity: SensitivityScanResult
    sparsity_dict: dict[str, float] = field(default_factory=dict)
    models: dict[str, torch.nn.Module] = field(default_factory=dict)
    benchmark: dict | None = None
