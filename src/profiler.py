"""Timing and memory profiling for prune-quant pipeline stages."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import time
from pathlib import Path

import torch


@dataclass
class StageBenchmark:
    """Wall-clock and peak GPU memory for one pipeline stage."""

    name: str
    wall_time_s: float
    peak_memory_mib: float | None = None


@dataclass
class BenchmarkReport:
    """Collected timings for a full prune-quant run."""

    model_name: str
    stages: list[StageBenchmark] = field(default_factory=list)

    @property
    def total_time_s(self) -> float:
        return sum(stage.wall_time_s for stage in self.stages)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "total_time_s": self.total_time_s,
            "stages": [
                {
                    "name": stage.name,
                    "wall_time_s": stage.wall_time_s,
                    "peak_memory_mib": stage.peak_memory_mib,
                }
                for stage in self.stages
            ],
        }

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2))
        return output_path


class PipelineProfiler:
    """Accumulates per-stage benchmark results during a pipeline run."""

    def __init__(self, model_name: str, device: torch.device):
        self.report = BenchmarkReport(model_name=model_name)
        self.device = device

    @contextmanager
    def stage(self, name: str):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        start = time.perf_counter()
        try:
            yield
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            elapsed = time.perf_counter() - start
            peak_memory_mib = None
            if self.device.type == "cuda":
                peak_memory_mib = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

            self.report.stages.append(
                StageBenchmark(
                    name=name,
                    wall_time_s=elapsed,
                    peak_memory_mib=peak_memory_mib,
                )
            )

    def summary(self) -> str:
        lines = [
            f"Benchmark: {self.report.model_name}",
            f"Total time: {self.report.total_time_s:.2f}s",
            "",
            f"{'Stage':<28} {'Time (s)':>10} {'Peak GPU (MiB)':>16}",
            "-" * 58,
        ]
        for stage in self.report.stages:
            peak = f"{stage.peak_memory_mib:.1f}" if stage.peak_memory_mib is not None else "n/a"
            lines.append(f"{stage.name:<28} {stage.wall_time_s:>10.2f} {peak:>16}")
        return "\n".join(lines)
