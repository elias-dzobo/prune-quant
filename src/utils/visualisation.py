from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages

from src.types import EvalMetrics, PipelineResult, SensitivityScanResult


def _metrics_for_plot(result: PipelineResult) -> list[EvalMetrics]:
    return [
        result.base,
        result.pruned,
        result.quantized,
        result.pruned_quantized,
    ]


def plot_metrics_comparison(
    result: PipelineResult,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Bar chart comparing accuracy, memory, and latency across model variants."""
    metrics = _metrics_for_plot(result)
    labels = [metric.label for metric in metrics]
    x = range(len(labels))
    width = 0.25

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    quality_label = metrics[0].metric_label.capitalize()
    if quality_label == "Accuracy":
        quality_values = [metric.accuracy for metric in metrics]
        quality_axis_label = "Accuracy (%)"
    else:
        quality_values = [metric.accuracy for metric in metrics]
        quality_axis_label = "Perplexity"

    ax.bar([index - width for index in x], quality_values, width, label=quality_axis_label)
    memories = [metric.memory_mib for metric in metrics]
    latencies = [metric.latency_ms for metric in metrics]
    ax.bar(x, memories, width, label="Memory (MiB)")
    ax.bar([index + width for index in x], latencies, width, label="Latency (ms)")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Model Comparison: Accuracy, Memory, Latency")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sensitivity_curves(
    sensitivity: SensitivityScanResult,
    ax: plt.Axes | None = None,
    max_layers: int = 8,
) -> plt.Figure:
    """Plot accuracy vs sparsity for each layer (sensitivity scan)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    layers_to_plot = sensitivity.layer_names[:max_layers]
    for layer_name, layer_accuracies in zip(
        layers_to_plot,
        sensitivity.accuracies[:max_layers],
    ):
        short_name = layer_name.split(".")[-2] if "." in layer_name else layer_name
        ax.plot(
            sensitivity.sparsity_levels,
            layer_accuracies,
            marker="o",
            linewidth=1.5,
            label=short_name,
        )

    if sensitivity.dense_accuracy:
        ax.axhline(
            sensitivity.dense_accuracy,
            linestyle="--",
            color="black",
            linewidth=1.0,
            label="Dense baseline",
        )

    ax.set_xlabel("Sparsity")
    ylabel = "Accuracy (%)" if sensitivity.metric_label == "accuracy" else "Perplexity"
    ax.set_ylabel(ylabel)
    ax.set_title("Layer Sensitivity Curves")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_weight_distribution(
    model: torch.nn.Module,
    title: str,
    bins: int = 64,
    count_non_zero: bool = False,
    max_layers: int = 6,
) -> plt.Figure:
    """Histogram of weight distributions for prunable layers."""
    layer_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.dim() > 1
    ][:max_layers]

    if not layer_params:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No prunable layers found", ha="center", va="center")
        ax.set_title(title)
        return fig

    rows = (len(layer_params) + 1) // 2
    cols = min(2, len(layer_params))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (name, param) in zip(axes_list, layer_params):
        weights = param.detach().reshape(-1).cpu()
        if count_non_zero:
            weights = weights[weights != 0]

        ax.hist(weights, bins=bins, density=True, color="steelblue", alpha=0.75)
        ax.set_title(name.split(".")[-1])
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Density")

    for ax in axes_list[len(layer_params) :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    return fig


def _metrics_summary_table(result: PipelineResult) -> str:
    metric_name = result.base.metric_label.capitalize()
    unit = "%" if result.base.metric_label == "accuracy" else ""
    lines = [
        f"Variant                  {metric_name:<10} Memory(MiB)  Latency(ms)  Sparsity  Precision",
        "-" * 78,
    ]
    for metric in _metrics_for_plot(result):
        lines.append(
            f"{metric.label:<24} {metric.accuracy:>8.2f}{unit:<3} "
            f"{metric.memory_mib:>11.2f} {metric.latency_ms:>11.2f} "
            f"{metric.sparsity:>9.2%} {metric.precision:>9}"
        )
    if result.benchmark:
        lines.extend(["", "Stage benchmarks (wall time):", "-" * 40])
        for stage in result.benchmark.get("stages", []):
            lines.append(f"  {stage['name']:<26} {stage['wall_time_s']:>6.2f}s")
        lines.append(f"  {'total':<26} {result.benchmark.get('total_time_s', 0):>6.2f}s")
    return "\n".join(lines)


def plot_benchmark_stages(
    benchmark: dict,
    title: str = "Pipeline Stage Benchmarks",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Horizontal bar chart of wall-clock time per pipeline stage."""
    stages = benchmark.get("stages", [])
    names = [stage["name"] for stage in stages]
    times = [stage["wall_time_s"] for stage in stages]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)))
    else:
        fig = ax.figure

    ax.barh(names, times, color="steelblue", alpha=0.85)
    ax.set_xlabel("Wall time (s)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def generate_visual_report(
    result: PipelineResult,
    models: dict[str, torch.nn.Module],
    output_dir: str | Path = "reports",
    report_name: str = "prune_quant_report",
) -> Path:
    """
    Generate a multi-page visual report for a prune-quant run.

    Sections:
      - Summary metrics table
      - Metrics comparison chart
      - Sensitivity curves
      - Base / Pruned / Quantized / Pruned+Quantized weight distributions
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path / f"{report_name}.pdf"

    with PdfPages(pdf_path) as pdf:
        # Summary page
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            "Prune-Quant Compression Report\n\n" + _metrics_summary_table(result),
            va="top",
            family="monospace",
            fontsize=10,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Metrics comparison
        fig = plot_metrics_comparison(result)
        pdf.savefig(fig)
        plt.close(fig)

        # Sensitivity graph
        fig = plot_sensitivity_curves(result.sensitivity)
        pdf.savefig(fig)
        plt.close(fig)

        if result.benchmark:
            fig = plot_benchmark_stages(
                result.benchmark,
                title=f"Benchmark — {result.benchmark.get('model_name', 'model')}",
            )
            pdf.savefig(fig)
            plt.close(fig)

        # Weight distribution pages for each model variant
        section_titles = {
            "base": "Base Model — Weight Distribution",
            "pruned": "Pruned Model — Weight Distribution",
            "quantized": "Quantized Model — Weight Distribution",
            "pruned_quantized": "Pruned + Quantized Model — Weight Distribution",
        }
        for key, title in section_titles.items():
            if key not in models:
                continue
            count_non_zero = key in {"pruned", "pruned_quantized"}
            fig = plot_weight_distribution(
                models[key],
                title=title,
                count_non_zero=count_non_zero,
            )
            pdf.savefig(fig)
            plt.close(fig)

    # Also save the comparison chart as a standalone PNG for quick viewing.
    fig = plot_metrics_comparison(result)
    png_path = output_path / f"{report_name}_metrics.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    return pdf_path
