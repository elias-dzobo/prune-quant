from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import DataLoader


TaskType = Literal["classification", "text_generation"]


@dataclass
class ModelBundle:
    """A model paired with evaluation data and task metadata."""

    name: str
    model: torch.nn.Module
    dataloader: DataLoader
    task: TaskType
    description: str = ""
    source: str = ""

    @property
    def higher_is_better(self) -> bool:
        return self.task == "classification"

    @property
    def metric_label(self) -> str:
        return "accuracy" if self.task == "classification" else "perplexity"
