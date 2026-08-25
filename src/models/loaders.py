"""
Download and prepare small benchmark models for prune-quant testing.

Models
------
- ResNet-18 on CIFAR-10 (image classification)
- DistilGPT-2 on WikiText-2 (causal language modelling / text generation)
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

from src.models.bundle import ModelBundle

_DATA_ROOT = Path("data")


def _ensure_data_root() -> Path:
    _DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return _DATA_ROOT


def load_resnet18_cifar10(
    batch_size: int = 64,
    num_workers: int = 0,
    max_samples: int | None = 512,
    download: bool = True,
) -> ModelBundle:
    """
    Fetch ResNet-18 (10 classes) and a CIFAR-10 test subset.

    Downloads CIFAR-10 via torchvision on first run (~170 MB).
    """
    data_root = _ensure_data_root()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    test_dataset = datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=download,
        transform=transform,
    )

    if max_samples is not None and max_samples < len(test_dataset):
        indices = list(range(max_samples))
        test_dataset = Subset(test_dataset, indices)

    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = resnet18(num_classes=10)

    return ModelBundle(
        name="resnet18-cifar10",
        model=model,
        dataloader=dataloader,
        task="classification",
        description="ResNet-18 image classifier evaluated on CIFAR-10",
        source="torchvision.models.resnet18 + torchvision.datasets.CIFAR10",
    )


def load_distilgpt2_wikitext(
    batch_size: int = 4,
    max_samples: int | None = 128,
    max_length: int = 128,
    download: bool = True,
) -> ModelBundle:
    """
    Fetch DistilGPT-2 and a WikiText-2 test subset for perplexity evaluation.

    Downloads model weights from Hugging Face and WikiText-2 via ``datasets``.
    """
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def tokenize_batch(examples: dict) -> dict:
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    # Filter empty lines produced by WikiText formatting.
    dataset = dataset.filter(lambda row: len(row["text"].strip()) > 0)

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    dataloader = DataLoader(tokenized, batch_size=batch_size, shuffle=False)

    return ModelBundle(
        name="distilgpt2-wikitext",
        model=model,
        dataloader=dataloader,
        task="text_generation",
        description="DistilGPT-2 causal LM evaluated on WikiText-2 (perplexity)",
        source="huggingface/distilgpt2 + wikitext-2-raw-v1",
    )
