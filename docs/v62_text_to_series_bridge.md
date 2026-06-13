# V6.2 Text-to-Series Cross-Modal Bridge

## 1. Diagnosis

The V6.1 text-to-series path encoded captions into slot tokens, then reduced them to a single `text_context` before routing and velocity prediction. The global operator gate and operator bank did receive this vector, but the model had no explicit token-level interaction between caption semantics and the current flow state. In practice, this made it easy for the generator to learn mostly unconditional statistics from `x_t` and weakly use text.

The latent regime adapter was useful as a diagnostic ablation, but it is not a strong text-to-series mechanism by itself. It predicts a soft posterior from state, text, and flow time, then injects a residual context. If the posterior is nearly uniform, the learned regime path cannot prove meaningful dynamic control. The old legacy mapper also produced text/time/channel affinity tensors, but previous entropy diagnostics showed those affinities were not reliable enough to be the main innovation.

## 2. Reference Lessons

VerbalTS shows that conditions should enter the generator at multiple internal points, not only as input concatenation. TimeCMA shows that explicit cross-modal attention is a stronger alignment primitive than add/concat fusion. Rectified-flow T2S implementations support keeping the main objective simple. DiT/AdaLN-style designs motivate conditioning internal blocks, but a full block rewrite would be too invasive for this repair.

V6.2 therefore adds one bridge module instead of stacking new routers or many losses.

## 3. New Data Flow

`CrossModalConditionBridge` runs inside `TextToTSFlow.forward` after text encoding and before operator routing:

```text
caption -> text_encoder -> slot_tokens [B,J,D]
x_t,t -> bridge state tokens:
  time patch tokens [B,P,D]
  channel summary tokens [B,C,D]
  spectral band tokens [B,S,D]

slot_tokens <-> state_tokens via bidirectional cross-attention
  -> bridge_context [B,D]
  -> expert_context [B,3,D]

bridge_context -> GlobalOperatorGate
bridge_context + expert_context -> ResidualOperatorBank
```

The three expert contexts correspond to temporal, channel, and frequency velocity experts. This makes the text-state interaction affect both operator selection and expert-specific velocity prediction.

## 4. Loss Design

The main objective remains conditional flow matching:

```text
L = L_CFM + lambda_regime * L_regime_ortho + lambda_bridge * L_bridge_align
```

For V6.2 bridge experiments, `lambda_regime=0` because latent regime is off by default.

`L_bridge_align` is optional and uses symmetric batch InfoNCE between pooled text-side bridge features and pooled state-side bridge features. It does not use handcrafted trend, frequency, volatility, or patch pseudo labels. The intended ablation is:

- `weather_v62_bridge.yaml`: `bridge_alignment_weight=0.0`
- `weather_v62_bridge_align.yaml`: `bridge_alignment_weight=0.001`

If the alignment loss does not improve caption sensitivity, keep the bridge and disable the loss.

## 5. Modified Files

- `effectcma_flow/models/cross_modal_bridge.py`: new bidirectional text-state bridge.
- `effectcma_flow/models/text_to_ts_flow.py`: bridge insertion and context routing.
- `effectcma_flow/models/operator_bank.py`: optional expert-specific context injection.
- `effectcma_flow/models/build.py`: V6.2 config passthrough.
- `effectcma_flow/training/train_step.py`: optional bridge alignment loss and diagnostics.
- `scripts/train_weather.py`: config passthrough and bridge logging.
- `configs/weather_v62_bridge*.yaml`: V6.2 experiment configs.
- `tests/test_model_training.py`: bridge, train-step, and compatibility tests.

## 6. Train And Validate

Pure bridge:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/train_weather.py \
  --config configs/weather_v62_bridge.yaml \
  --data-root /home/newuser001/huangyu/Research/Effect123/datasets/synth-m \
  --text-encoder-mode longclip \
  --text-encoder-model /home/newuser001/huangyu/Research/Effect123/save/Longclip \
  --batch-size 1024 \
  --eval-batch-size 512 \
  --max-steps 10000 \
  --checkpoint-dir checkpoints/v62_bridge_gpu0 \
  > logs/v62_bridge_gpu0.log 2>&1 &
```

Bridge + alignment:

```bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/train_weather.py \
  --config configs/weather_v62_bridge_align.yaml \
  --data-root /home/newuser001/huangyu/Research/Effect123/datasets/synth-m \
  --text-encoder-mode longclip \
  --text-encoder-model /home/newuser001/huangyu/Research/Effect123/save/Longclip \
  --batch-size 1024 \
  --eval-batch-size 512 \
  --max-steps 10000 \
  --checkpoint-dir checkpoints/v62_bridge_align_gpu1 \
  > logs/v62_bridge_align_gpu1.log 2>&1 &
```

Evaluate each checkpoint with the same VerbalTS normal, shuffled, and blank caption protocol used for V6.1.

## 7. Expected Signals

The primary success signal is not only lower FID/JFTSD. Normal captions should outperform shuffled and blank captions, especially on CTTP and JFTSD sensitivity. Bridge attention diagnostics should be finite, and operator usage should remain non-degenerate. If `bridge_align` improves caption sensitivity without hurting FID/JFTSD, keep it; otherwise use the bridge-only variant as the main V6.2 mechanism.
