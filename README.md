# EffectCMA-Flow

EffectCMA-Flow is a PyTorch research codebase for text-controlled time series generation. It learns a residual flow from a base trajectory `B` to a text-induced counterfactual target `Y` through a time-channel-operator effect field.

The first release focuses on a runnable core system:

- Weather `.npy` loading with train-only normalization.
- Semi-synthetic counterfactual pair construction.
- Frozen text encoders with `hash`, `precomputed`, and HuggingFace `hf` modes.
- Cross-modal effect mapping: `A_t`, `A_c`, `A_o` -> `G`.
- Residual operator bank trained with a single conditional flow matching loss.
- Euler sampling, metrics, visualization hooks, tests, and smoke scripts.

## Model Notes

The current core model avoids using a plain MLP as the main method:

- `TimePatchEncoder` uses patch tokens followed by a Transformer encoder.
- `ChannelEncoder` uses shared temporal patch encoding per channel, then a channel Transformer mixer. This keeps the channel-independent inductive bias used by PatchTST-style models while still allowing multivariate channel interaction.
- `EffectMapper` builds a multi-rank time-channel-operator field. Each text slot can activate several interpretable sub-effects instead of one shallow outer-product gate.
- `ResidualOperatorBank` contains independent dilated temporal experts. A low-cost channel mixer shares cross-channel context, and each expert keeps its own TCN-style residual stack and small-initialized head.

Design inspirations include PatchTST-style patch/channel representation, TCN-style dilated residual temporal blocks, and mixture-of-experts specialization. The implementation here is purpose-built for EffectCMA-Flow rather than copied from those repositories.

## Data

The expected Weather directory is supplied by `--data-root` or the `WEATHER_ROOT` environment variable:

```text
Weather
  meta.json
  train_ts.npy
  valid_ts.npy
  test_ts.npy
  train_text_caps.npy
  valid_text_caps.npy
  test_text_caps.npy
  train_attrs_idx.npy
  valid_attrs_idx.npy
  test_attrs_idx.npy
  text_embeddings_128_all_caps.npy
  text_embedding_caption_counts.npy
```

For the current Weather data, `*_ts.npy` has shape `[N, 36, 21]`. The code validates shapes at runtime and does not hard-code `L=36` or `C=21` inside model layers.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Run a tiny CPU smoke train:

```bash
python scripts/train_weather.py --config configs/weather_core.yaml --data-root /path/to/Weather --max-steps 2 --batch-size 4 --text-encoder-mode hash --checkpoint-dir checkpoints/weather_core
```

Evaluate:

```bash
python scripts/eval_weather.py --config configs/weather_core.yaml --data-root /path/to/Weather --checkpoint checkpoints/weather_core/latest.pt --max-batches 2 --text-encoder-mode hash
```

Sample:

```bash
python scripts/sample_weather.py --config configs/weather_core.yaml --data-root /path/to/Weather --checkpoint checkpoints/weather_core/latest.pt --index 0 --text-encoder-mode hash
```

## Text Encoder Modes

- `hash`: deterministic offline hashing encoder for smoke tests and CI.
- `precomputed`: uses cached embeddings for the synthetic effect slots. It must not use original Weather caption embeddings for semi-synthetic effect control.
- `hf`: frozen HuggingFace encoder, for server training. Example model: `BAAI/bge-small-en-v1.5` or `sentence-transformers/all-MiniLM-L6-v2`.

Unit tests do not require network access.

## Boundary

Synthetic effect operators are used only to build training pairs `(B, text, Y)` and evaluation masks. They are not model inputs and are not shared with the learnable residual operator bank.

`train.cfm_loss_mode=balanced` is a reweighted single CFM objective for sparse local effects. It is not an auxiliary mask loss; inside/outside losses are logged as diagnostics.
