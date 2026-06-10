import torch 
from torchprofile import profile_macs

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

def get_model_macs(model: torch.nn.Module, inputs: torch.Tensor) -> int:
    return profile_macs(model, inputs)

def get_tensor_sparsity(tensor: torch.Tensor) -> float:
    return 1.0 - float(tensor.count_non_zero()) / tensor.numel()

def get_model_sparsity(model: torch.nn.Module) ->  float:
    num_nonzeros, num_elements = 0, 0 

    for param in model.parameters():
        num_nonzeros += param.count_non_zero()
        num_elements += param.numel()

    return 1.0 - float(num_nonzeros) / num_elements