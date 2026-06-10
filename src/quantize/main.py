"""
Linear Quantization
===================

Maps floating-point tensor values to low-bit integers and back, using a
linear scale (and optionally a zero-point). Two variants are supported.

Shared concepts
---------------
- n_bits: target bit width (e.g. 8 for int8, 4 for int4).
- Integer representable range [q_min, q_max] depends on signedness:
    signed n-bit:   q_min = -2^(n-1),       q_max = 2^(n-1) - 1
    unsigned n-bit: q_min = 0,                q_max = 2^n - 1
  Example: signed int8 → q_min = -128, q_max = 127.

- scale: float step size; one integer step corresponds to `scale` in float space.
- zero_point (z): integer offset aligning float 0 with a quant level (asymmetric only).
- clamp: after rounding, clip q to [q_min, q_max] to avoid overflow.

General quantize / dequantize (affine form)
-------------------------------------------
  q = clamp(round(x / scale + z), q_min, q_max)
  x_hat = (q - z) * scale

`x_hat` is the reconstructed float (dequantized) value.


Approach 1: Asymmetric (min–max + zero-point)
---------------------------------------------
Use when the tensor is NOT zero-centered (e.g. ReLU activations, skewed
ranges). Uses the full float range [x_min, x_max] and the full integer
range [q_min, q_max].

When to use
  - Values are mostly one-sided (all positive or strongly skewed).
  - |x_min| and x_max are very different.
  - You want to avoid wasting half the integer range.

Algorithm
  1. Compute tensor bounds:
       x_min = min(x)
       x_max = max(x)

  2. Look up integer bounds for the target precision:
       q_min, q_max  (from config / dtype)

  3. Compute scale:
       scale = (x_max - x_min) / (q_max - q_min)

  4. Compute zero-point (maps x_min → q_min):
       z = round(q_min - x_min / scale)
       z = clamp(z, q_min, q_max)

  5. Quantize each value x:
       q = clamp(round(x / scale + z), q_min, q_max)

  6. Dequantize each integer q:
       x_hat = (q - z) * scale

Example (uint8, activations in [0, 10])
  q_min=0, q_max=255, x_min=0, x_max=10
  scale = 10 / 255 ≈ 0.0392
  z = 0
  x=10 → q=255 → x_hat≈10


Approach 2: Symmetric (abs-max, zero-point = 0)
------------------------------------------------
Use when the tensor is roughly zero-centered (typical for weights).
Only magnitude matters; negative and positive ranges are treated equally.

When to use
  - Values straddle zero with similar extent on both sides.
  - x_min ≈ -x_max (symmetric around zero).
  - Common default for weight quantization.

Algorithm
  1. Compute maximum absolute value:
       max_abs = max(|x|)

  2. Look up positive integer bound:
       q_max  (for signed: q_max = 2^(n-1) - 1; q_min = -q_max - 1 or -2^(n-1))

  3. Compute scale (maps max_abs → q_max):
       scale = max_abs / q_max

  4. Zero-point is fixed:
       z = 0

  5. Quantize each value x:
       q = clamp(round(x / scale), q_min, q_max)

  6. Dequantize each integer q:
       x_hat = q * scale

Example (signed int8, weights in [-5, 5])
  q_max=127, max_abs=5
  scale = 5 / 127 ≈ 0.0394
  x=5 → q=127 → x_hat≈5
  x=-5 → q=-127 → x_hat≈-5


Choosing between approaches
---------------------------
  Symmetric:     zero-centered, |min| ≈ max  →  abs-max, z = 0
  Asymmetric:    skewed or all-positive      →  min–max, z ≠ 0

Do NOT use mean alone to decide; use x_min, x_max, and whether 0 lies
in the useful range.


Implementation notes
--------------------
- Per-tensor: one (scale, z) for the whole tensor.
- Per-channel: compute (scale, z) per output channel (better accuracy).
- Rounding: standard round-to-nearest; some runtimes use floor or
  stochastic rounding.
- Quantize divides by scale; dequantize multiplies by scale (not the reverse).
"""
import yaml 
import torch 


precision_config = yaml.load(open('config/qbits.yml'), Loader=yaml.FullLoader)


class LinearQuantizer:
    def __init__(self, precision: str):
        self.scale = None 

    def quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = torch.abs(tensor)
        tensor_max = torch.max(tensor)
        scale = tensor_max / self.precision_config[precision]['max']
        self.scale = scale
        quantized_tensor = torch.round(tensor / scale)
        return quantized_tensor

    def dequantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.scale

    def quantize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        for name, param in model.named_parameters():
            if param.dim() > 1:
                param.data = self.quantize_tensor(param.data)
        return model

    def dequantize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        for name, param in model.named_parameters():
            if param.dim() > 1:
                param.data = self.dequantize_tensor(param.data)
        return model