import copy
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from src.eval import evaluate_model, precision_to_bits
from src.models.bundle import ModelBundle, TaskType
from src.pruner.main import Pruner
from src.profiler import PipelineProfiler
from src.quantize.main import LinearQuantizer
from src.types import PipelineResult
from src.utils.utils import sensitivity_scan_dataloader


@dataclass
class PipelineConfig:
    """Configuration for a full prune-quant compression run."""

    precision: str = "int8"
    scan_start: float = 0.4
    scan_end: float = 1.0
    scan_step: float = 0.1
    sparsity_levels: list[float] | None = None
    latency_warmup: int = 10
    latency_runs: int = 50
    verbose: bool = True
    profile: bool = True
    task: TaskType = "classification"


class PruneQuantPipeline:
    """
    End-to-end compression pipeline.

    Workflow:
      1. Evaluate the base model (quality, memory, latency)
      2. Run layer-wise sensitivity scan and prune
      3. Evaluate the pruned model
      4. Quantize the base model and evaluate
      5. Quantize the pruned model and evaluate
      6. Return all metrics (and optional stage benchmarks) for reporting
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def run(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device | None = None,
        *,
        task: TaskType | None = None,
        model_name: str = "model",
    ) -> PipelineResult:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        task = task or self.config.task
        model = copy.deepcopy(model).to(device)
        bits = precision_to_bits(self.config.precision)
        profiler = PipelineProfiler(model_name, device) if self.config.profile else None

        def stage(name: str):
            if profiler is None:
                from contextlib import nullcontext

                return nullcontext()
            return profiler.stage(name)

        if self.config.verbose:
            print("Evaluating base model...")
        with stage("evaluate_base"):
            base_metrics = evaluate_model(
                model,
                dataloader,
                device,
                label="Base Model",
                task=task,
                bits_per_weight=32,
                warmup=self.config.latency_warmup,
                latency_runs=self.config.latency_runs,
            )

        if self.config.verbose:
            print("Running sensitivity scan...")
        with stage("sensitivity_scan"):
            sensitivity, sparsity_dict = sensitivity_scan_dataloader(
                model=model,
                dataloader=dataloader,
                device=device,
                sparsity_levels=self.config.sparsity_levels,
                scan_start=self.config.scan_start,
                scan_end=self.config.scan_end,
                scan_step=self.config.scan_step,
                task=task,
                verbose=self.config.verbose,
            )

        if self.config.verbose:
            print("Pruning model...")
        with stage("prune"):
            base_for_prune = copy.deepcopy(model)
            pruner = Pruner(base_for_prune, sparsity_dict)
            pruned_model = pruner.prune_copy().to(device)

        if self.config.verbose:
            print("Evaluating pruned model...")
        with stage("evaluate_pruned"):
            pruned_metrics = evaluate_model(
                pruned_model,
                dataloader,
                device,
                label="Pruned Model",
                task=task,
                bits_per_weight=32,
                use_effective_sparsity=True,
                warmup=self.config.latency_warmup,
                latency_runs=self.config.latency_runs,
            )

        if self.config.verbose:
            print(f"Quantizing base model to {self.config.precision}...")
        with stage("quantize_base"):
            quantizer = LinearQuantizer(precision=self.config.precision)
            quantized_model = quantizer.quantize_copy(model).to(device)

        if self.config.verbose:
            print("Evaluating quantized model...")
        with stage("evaluate_quantized"):
            quantized_metrics = evaluate_model(
                quantized_model,
                dataloader,
                device,
                label="Quantized Model",
                task=task,
                bits_per_weight=bits,
                precision=self.config.precision,
                warmup=self.config.latency_warmup,
                latency_runs=self.config.latency_runs,
            )

        if self.config.verbose:
            print(f"Quantizing pruned model to {self.config.precision}...")
        with stage("quantize_pruned"):
            pruned_quantized_model = quantizer.quantize_copy(pruned_model).to(device)

        if self.config.verbose:
            print("Evaluating pruned + quantized model...")
        with stage("evaluate_pruned_quantized"):
            pruned_quantized_metrics = evaluate_model(
                pruned_quantized_model,
                dataloader,
                device,
                label="Pruned + Quantized Model",
                task=task,
                bits_per_weight=bits,
                use_effective_sparsity=True,
                precision=self.config.precision,
                warmup=self.config.latency_warmup,
                latency_runs=self.config.latency_runs,
            )

        benchmark = profiler.report.to_dict() if profiler is not None else None
        if profiler is not None and self.config.verbose:
            print(f"\n{profiler.summary()}")

        return PipelineResult(
            base=base_metrics,
            pruned=pruned_metrics,
            quantized=quantized_metrics,
            pruned_quantized=pruned_quantized_metrics,
            sensitivity=sensitivity,
            sparsity_dict=sparsity_dict,
            models={
                "base": model,
                "pruned": pruned_model,
                "quantized": quantized_model,
                "pruned_quantized": pruned_quantized_model,
            },
            benchmark=benchmark,
        )

    def run_bundle(self, bundle: ModelBundle, device: torch.device | None = None) -> PipelineResult:
        """Run the pipeline on a ``ModelBundle`` (model + dataloader + task)."""
        config = copy.copy(self.config)
        config.task = bundle.task
        runner = PruneQuantPipeline(config)
        return runner.run(
            bundle.model,
            bundle.dataloader,
            device=device,
            task=bundle.task,
            model_name=bundle.name,
        )
