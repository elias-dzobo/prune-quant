# prune-quant

A model compression engine written from first principles. It prunes and quantizes PyTorch models,
then measures what that actually cost you in accuracy — pruning alone, quantization alone, and both
together, evaluated on the same footing.

I built this to understand compression by implementing it rather than calling
`torch.quantization`. The quantization math, the mask generation, and the sensitivity analysis are
all hand-rolled and documented inline.

## What it does

- **Fine-grained magnitude pruning** driven by a per-layer sparsity dictionary. Masks are computed
  without mutating live weights, so you can scan sparsity levels non-destructively.
- **Per-layer sensitivity scanning** — sweeps sparsity across a range and records where each layer
  starts to degrade, so the sparsity budget goes where the model can absorb it.
- **Linear quantization from scratch** in both asymmetric (min–max with zero-point) and symmetric
  forms, with the affine mapping `q = clamp(round(x / scale + z), q_min, q_max)` implemented and
  explained in `src/quantize/main.py`.
- **Multiple precisions** — int8, int4, fp8 (e4m3fn), fp4 (nvfp4 e2m1), fp16, bf16 — with
  representable ranges declared in `src/config/qbits.yml`.
- **Stage profiling** — wall time and peak memory per stage, written out as JSON.
- **Visual reports** — per-run PDF and metric plots in `reports/`.

## The pipeline

`PruneQuantPipeline` runs the full comparison matrix so the four conditions are measured
identically:

```
evaluate_base
  → sensitivity_scan
  → prune            → evaluate_pruned
  → quantize_base    → evaluate_quantized
  → quantize_pruned  → evaluate_pruned_quantized
```

That last branch is the interesting one: it answers whether stacking compression is worth the extra
accuracy hit, or whether one technique alone gets you most of the memory back.

## Quickstart

Smoke test on a synthetic CNN, no downloads required:

```bash
uv sync
uv run python main.py
```

Run the real benchmark (distilgpt2 on wikitext):

```bash
uv run python benchmark.py
```

Results land in `reports/` as a PDF, a metrics plot, and a stage-timing JSON. A sample run is
committed there.

## Layout

```
src/
  pipeline.py    — orchestrates the prune → quantize → evaluate matrix
  pruner/        — mask generation and fine-grained magnitude pruning
  quantize/      — linear quantization/dequantization, asymmetric and symmetric
  profiler.py    — per-stage wall time and peak memory
  eval.py        — perplexity / accuracy evaluation
  models/        — model + dataloader bundles (loaders for demo and HF models)
  config/qbits.yml — representable ranges per precision
  utils/         — mask helpers and plotting
notebooks/       — exploratory pruning and LLM-compression experiments
reports/         — generated benchmark artifacts
```

Datasets and checkpoints are gitignored; `benchmark.py` downloads what it needs.
