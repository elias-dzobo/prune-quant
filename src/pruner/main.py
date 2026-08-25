import copy

import torch

from src.utils.utils import _fine_grained_prune_mask


def apply_fine_grained_masks(
    model: torch.nn.Module,
    masks: dict[str, torch.Tensor],
) -> torch.nn.Module:
    """Apply precomputed pruning masks to a model's weight tensors."""
    for name, param in model.named_parameters():
        if name in masks:
            param.data.mul_(masks[name])
    return model


class Pruner:
    """Fine-grained magnitude pruner driven by a per-layer sparsity dictionary."""

    def __init__(self, model: torch.nn.Module, sparsity_dict: dict[str, float]):
        self.model = model
        self.sparsity_dict = sparsity_dict
        self.masks = self.generate_prune_masks(sparsity_dict)

    def generate_prune_masks(
        self,
        sparsity_dict: dict[str, float],
    ) -> dict[str, torch.Tensor]:
        """Build binary masks without mutating the live model weights."""
        masks: dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.dim() > 1 and name in sparsity_dict:
                masks[name] = _fine_grained_prune_mask(param.detach(), sparsity_dict[name])
        return masks

    def fine_grained_prune(self) -> torch.nn.Module:
        """Apply generated masks to the model."""
        return apply_fine_grained_masks(self.model, self.masks)

    def prune_copy(self) -> torch.nn.Module:
        """Return a deep-copied model with pruning applied."""
        pruned_model = copy.deepcopy(self.model)
        apply_fine_grained_masks(pruned_model, self.masks)
        return pruned_model

    def channel_prune(self):
        raise NotImplementedError("Channel pruning is not implemented yet.")
