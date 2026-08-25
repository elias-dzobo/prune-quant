"""
Run prune-quant benchmarks on bundled test models.

Models:
  - resnet18  : ResNet-18 + CIFAR-10 (image classification)
  - distilgpt2: DistilGPT-2 + WikiText-2 (text generation / perplexity)
  - all       : run both
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.models.loaders import load_distilgpt2_wikitext, load_resnet18_cifar10
from src.pipeline import PipelineConfig, PruneQuantPipeline
from src.utils.visualisation import generate_visual_report


def _print_quality(result, label: str, metric) -> None:
    unit = "%" if metric.metric_label == "accuracy" else ""
    print(f"{label:<28} {metric.accuracy:.2f}{unit}")


def run_model_benchmark(
    bundle_loader,
    *,
    output_dir: Path,
    precision: str,
    scan_start: float,
    scan_end: float,
    scan_step: float,
    max_samples: int | None,
    device: torch.device,
):
    bundle = bundle_loader(max_samples=max_samples)
    print(f"\n{'=' * 60}")
    print(f"Model: {bundle.name}")
    print(f"Task:  {bundle.task}")
    print(f"Source: {bundle.source}")
    print(f"{'=' * 60}")

    config = PipelineConfig(
        precision=precision,
        scan_start=scan_start,
        scan_end=scan_end,
        scan_step=scan_step,
        task=bundle.task,
        profile=True,
        verbose=True,
    )

    result = PruneQuantPipeline(config).run_bundle(bundle, device=device)
    report_path = generate_visual_report(
        result,
        models=result.models,
        output_dir=output_dir,
        report_name=bundle.name,
    )

    print(f"\nReport: {report_path}")
    print(f"{'Metric':<28} {result.base.metric_label}")
    _print_quality(result, "Base", result.base)
    _print_quality(result, "Pruned", result.pruned)
    _print_quality(result, "Quantized", result.quantized)
    _print_quality(result, "Pruned + Quantized", result.pruned_quantized)

    benchmark_path = output_dir / f"{bundle.name}_benchmark.json"
    if result.benchmark:
        import json

        benchmark_path.write_text(json.dumps(result.benchmark, indent=2))
        print(f"Benchmark JSON: {benchmark_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune-quant model benchmarks")
    parser.add_argument(
        "--model",
        choices=["all", "resnet18", "distilgpt2"],
        default="all",
        help="Which bundled model(s) to benchmark",
    )
    parser.add_argument("--precision", default="int8", help="Quantization precision")
    parser.add_argument("--scan-start", type=float, default=0.0)
    parser.add_argument("--scan-end", type=float, default=0.8)
    parser.add_argument("--scan-step", type=float, default=0.2)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=256,
        help="Cap dataset size for faster benchmark runs",
    )
    parser.add_argument("--output-dir", default="reports", help="Report output directory")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, or cuda",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    loaders = {
        "resnet18": load_resnet18_cifar10,
        "distilgpt2": load_distilgpt2_wikitext,
    }

    selected = list(loaders.keys()) if args.model == "all" else [args.model]
    for model_key in selected:
        run_model_benchmark(
            loaders[model_key],
            output_dir=output_dir,
            precision=args.precision,
            scan_start=args.scan_start,
            scan_end=args.scan_end,
            scan_step=args.scan_step,
            max_samples=args.max_samples,
            device=device,
        )


if __name__ == "__main__":
    main()
