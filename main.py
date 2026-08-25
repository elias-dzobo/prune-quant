"""Quick smoke test with synthetic data (no model downloads)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.pipeline import PipelineConfig, PruneQuantPipeline
from src.utils.visualisation import generate_visual_report


def build_demo_model(num_classes: int = 10) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 32 * 32, num_classes),
    )


def build_demo_dataloader(batch_size: int = 32, num_samples: int = 128) -> DataLoader:
    inputs = torch.randn(num_samples, 3, 32, 32)
    targets = torch.randint(0, 10, (num_samples,))
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PipelineConfig(
        precision="int8",
        scan_start=0.0,
        scan_end=0.8,
        scan_step=0.2,
        profile=True,
        verbose=True,
    )
    result = PruneQuantPipeline(config).run(
        build_demo_model(),
        build_demo_dataloader(),
        device=device,
        model_name="demo-cnn",
    )
    report_path = generate_visual_report(result, models=result.models, report_name="demo")
    print(f"\nSmoke-test report: {report_path}")


if __name__ == "__main__":
    main()
