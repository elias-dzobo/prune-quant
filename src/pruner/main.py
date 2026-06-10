import torch 

"""
A Class that takes a model and a pruning config and prunes the model.
"""

def fine_grain_prune(tensor: torch.Tensor, sparsity: float) -. torch.Tensor:
    sparsity = min(max(sparsity, 0.0), 1.0)
    if sparsity == 1.0:
        tensor = tensor.zero_()
        return tensor.zeros_like(tensor)
    elif sparsity == 0.0:
        return tensor.ones_like(tensor)

    num_elements = tensor.numel() 

    num_zeros = int(num_elements * sparsity)
    importance = torch.abs(tensor)
    threshold = importance.view(-1).kthvalue(num_zeros).values 
    mask = importance > threshold 
    tensor.mul_(mask)

    return mask 


class Pruner:
    def __init__(self, model: torch.nn.Module, sparsity_dict: dict[str, float]):
        self.model = model 
        self.masks = self.generate_prune_masks(sparsity_dict)

    def _validate_model(self):
        """
        Check if the model is valid. ie. we can scan the model architecture
        """
        pass

    def generate_prune_masks(self, sparsity_dict: dict[str, float]) -> dict[str, torch.Tensor]:
        masks = dict()
        for name, param in self.model.named_parameters():
            if param.dim() > 1:
                masks[name] = fine_grain_prune(param, sparsity_dict[name])

        return masks  

    def fine_grained_prune(self):
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.mul_(self.masks[name])
 
    def channel_prune(self):
        pass 