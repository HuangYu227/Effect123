"""CFM training step for V6.1/V6.2.

Training objective:
    L = L_CFM
        + lambda_ortho * L_regime_ortho
        + lambda_bridge * L_bridge_align
        + lambda_rank * L_caption_rank
        + lambda_spectral * L_spectral

No legacy routing entropy, channel entropy, field mass, trend, frequency,
volatility, or artificial semantic-slot losses are used. The optional caption
ranking loss uses real captions only: positive caption versus batch-shuffled
caption. TSP scale statistics are diagnostics only, not training losses.

The optional spectral loss supervises the rFFT magnitude of the predicted clean
target (recovered from the flow state as ``x_t + (1 - t) * pred_v``) against the
ground-truth target. Pure time-domain velocity MSE under-weights high-frequency
structure, which is a known cause of poor spectral / distributional fidelity
(FID, J-FTSD); this term adds direct frequency-domain pressure.
"""
from __future__ import annotations

from typing import Any
import warnings

import torch

from effectcma_flow.training.normalized_spectral_loss import normalized_multi_resolution_fft_loss
from effectcma_flow.training.utils import batch_to_device, text_condition_from_batch


def cfm_train_step(
    model: torch.nn.Module,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    *,
    device: torch.device | None = None,
    grad_clip: float | None = 1.0,
    text_encoder_mode: str = "hash",
    cfm_loss_mode: str = "global",
    task_mode: str = "edit",
    noise_scale: float = 1.0,
    caption_slot_strategy: str = "single",
    max_caption_slots: int = 1,
    include_all_caption_candidates: bool = False,
    condition_dropout_prob: float = 0.0,
    regime_ortho_weight: float = 0.0,
    bridge_alignment_weight: float = 0.0,
    caption_ranking_weight: float = 0.0,
    caption_ranking_margin: float = 0.05,
    spectral_loss_weight: float = 0.0,
    spectral_loss_type: str = "magnitude",
    spectral_fft_sizes: Any = (16, 32, 64, 128),
    spectral_distance: str = "l1",
    spectral_log_magnitude: bool = True,
    spectral_time_weight_power: float = 0.0,
    operator_balance_weight: float = 0.0,
    # Backward-compatibility guard: old routing losses must stay disabled.
    routing_loss_weight: float = 0.0,
    **unused: Any,
) -> dict[str, Any]:
    if float(routing_loss_weight) != 0.0:
        raise ValueError("V6.1 forbids old routing/meta losses. Use regime_ortho_weight only.")
    for removed_key in ("tsp_scale_entropy_weight", "tsp_scale_balance_weight"):
        if removed_key in unused and float(unused[removed_key]) != 0.0:
            raise ValueError(
                f"{removed_key} has been removed: TSP scale statistics are diagnostics only, "
                "not auxiliary losses."
            )
    if unused:
        warnings.warn(f"Unused train_step kwargs ignored: {sorted(unused.keys())}", RuntimeWarning, stacklevel=2)
    if device is not None:
        batch = batch_to_device(batch, device)
    task_mode = str(task_mode).lower()

    negative_pred_v: torch.Tensor | None = None
    if task_mode == "edit":
        _require_keys(batch, ("B", "Y"), mode="edit")
        base = _expect_series(batch["B"], name="batch['B']")
        target = _expect_series(batch["Y"], name="batch['Y']")
        _require_same_shape(base, target, "B", "Y")
        source = base
        t = torch.rand(base.shape[0], device=base.device, dtype=base.dtype)
        x_t = (1.0 - t[:, None, None]) * base + t[:, None, None] * target
        target_v = target - base
        text_condition = text_condition_from_batch(batch, text_encoder_mode, condition_key="slots")
        pred_v, aux = model(base, x_t, t, text_condition)
    elif task_mode == "text2ts":
        _require_keys(batch, ("Y",), mode="text2ts")
        target = _expect_series(batch["Y"], name="batch['Y']")
        source = torch.randn_like(target) * float(noise_scale)
        t = _sample_flow_time(target.shape[0], device=target.device, dtype=target.dtype)
        x_t = (1.0 - t[:, None, None]) * source + t[:, None, None] * target
        target_v = target - source
        text_condition = text_condition_from_batch(
            _maybe_blank_caption_batch(batch, p=float(condition_dropout_prob)),
            text_encoder_mode,
            condition_key="caption",
            caption_slot_strategy=caption_slot_strategy,
            max_caption_slots=max_caption_slots,
            include_all_caption_candidates=include_all_caption_candidates,
        )
        # Note: prepare_condition is called inside model.forward(), not here.
        # This avoids redundant text-encoding and keeps dtype handling consistent.
        # Pass the clean target so a dense/clean-target alignment can encode it;
        # fall back gracefully for models whose forward does not accept it.
        try:
            pred_v, aux = model(x_t, t, text_condition, target=target)
        except TypeError as exc:
            if "target" not in str(exc):
                raise
            pred_v, aux = model(x_t, t, text_condition)
        if float(caption_ranking_weight) > 0.0 and target.shape[0] > 1:
            negative_batch = _caption_negative_batch(batch, device=target.device)
            negative_condition = text_condition_from_batch(
                _maybe_blank_caption_batch(negative_batch, p=float(condition_dropout_prob)),
                text_encoder_mode,
                condition_key="caption",
                caption_slot_strategy=caption_slot_strategy,
                max_caption_slots=max_caption_slots,
                include_all_caption_candidates=include_all_caption_candidates,
            )
            try:
                negative_pred_v, _ = model(x_t, t, negative_condition, compute_bridge_alignment=False)
            except TypeError as exc:
                if "compute_bridge_alignment" not in str(exc):
                    raise
                negative_pred_v, _ = model(x_t, t, negative_condition)
    else:
        raise ValueError(f"Unknown task_mode {task_mode!r}; expected 'text2ts' or 'edit'")

    if pred_v.shape != target_v.shape:
        raise ValueError(f"model pred_v shape {tuple(pred_v.shape)} must match target_v {tuple(target_v.shape)}")
    sq_error = (pred_v - target_v) ** 2
    cfm_loss, loss_parts = _cfm_loss(sq_error, batch.get("mask"), mode=cfm_loss_mode)
    regime_ortho = pred_v.new_zeros(())
    if float(regime_ortho_weight) > 0.0 and isinstance(aux, dict) and torch.is_tensor(aux.get("regime_ortho_loss")):
        regime_ortho = aux["regime_ortho_loss"].to(device=pred_v.device, dtype=pred_v.dtype)
    bridge_alignment = pred_v.new_zeros(())
    if float(bridge_alignment_weight) > 0.0 and isinstance(aux, dict) and torch.is_tensor(aux.get("bridge_alignment_loss")):
        bridge_alignment = aux["bridge_alignment_loss"].to(device=pred_v.device, dtype=pred_v.dtype)
    operator_balance = pred_v.new_zeros(())
    if float(operator_balance_weight) > 0.0 and isinstance(aux, dict) and torch.is_tensor(aux.get("operator_balance_loss")):
        operator_balance = aux["operator_balance_loss"].to(device=pred_v.device, dtype=pred_v.dtype)
    spectral_loss = pred_v.new_zeros(())
    if float(spectral_loss_weight) > 0.0:
        pred_x0 = x_t + (1.0 - t[:, None, None]) * pred_v
        if float(spectral_time_weight_power) > 0.0:
            # Opt-in (1 - t)^power time weighting: emphasize samples nearer the
            # source (small t) where the recovered x0 is least constrained by x_t.
            # Note: power > 0 shrinks the effective spectral magnitude (the mean
            # of a sub-unit weight is < 1); to compensate the user may raise
            # spectral_loss_weight in the config. We deliberately do NOT rescale
            # the weight here so the knob stays a pure, isolated time reweighting.
            per_sample = _clean_spectral_loss(
                pred_x0,
                target,
                mask=batch.get("mask"),
                loss_type=spectral_loss_type,
                fft_sizes=spectral_fft_sizes,
                distance=spectral_distance,
                log_magnitude=bool(spectral_log_magnitude),
                reduction="none",
            )
            time_weight = (1.0 - t).clamp_min(0.0) ** float(spectral_time_weight_power)
            spectral_loss = (per_sample * time_weight.to(per_sample.dtype)).mean()
        else:
            spectral_loss = _clean_spectral_loss(
                pred_x0,
                target,
                mask=batch.get("mask"),
                loss_type=spectral_loss_type,
                fft_sizes=spectral_fft_sizes,
                distance=spectral_distance,
                log_magnitude=bool(spectral_log_magnitude),
            )
    caption_ranking = pred_v.new_zeros(())
    caption_pos_mse = _per_sample_mse(pred_v, target_v, batch.get("mask")).mean()
    caption_neg_mse = pred_v.new_zeros(())
    caption_ranking_acc = pred_v.new_zeros(())
    if negative_pred_v is not None:
        if negative_pred_v.shape != target_v.shape:
            raise ValueError(f"negative pred_v shape {tuple(negative_pred_v.shape)} must match target_v {tuple(target_v.shape)}")
        pos_sample = _per_sample_mse(pred_v, target_v, batch.get("mask"))
        neg_sample = _per_sample_mse(negative_pred_v, target_v, batch.get("mask"))
        caption_ranking = torch.relu(float(caption_ranking_margin) + pos_sample - neg_sample).mean()
        caption_pos_mse = pos_sample.mean()
        caption_neg_mse = neg_sample.mean()
        caption_ranking_acc = (neg_sample > pos_sample).to(dtype=pred_v.dtype).mean()
    loss = (
        cfm_loss
        + float(regime_ortho_weight) * regime_ortho
        + float(bridge_alignment_weight) * bridge_alignment
        + float(caption_ranking_weight) * caption_ranking
        + float(spectral_loss_weight) * spectral_loss
        + float(operator_balance_weight) * operator_balance
    )

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    with torch.no_grad():
        stats = _flow_diagnostics(target=target, source=source, pred_v=pred_v, target_v=target_v)
        diag = _aux_diagnostics(aux)
    return {
        "loss": loss.detach(),
        "loss_cfm": cfm_loss.detach(),
        "loss_regime_ortho": regime_ortho.detach(),
        "loss_bridge_alignment": bridge_alignment.detach(),
        "loss_caption_ranking": caption_ranking.detach(),
        "loss_spectral": spectral_loss.detach(),
        "loss_operator_balance": operator_balance.detach(),
        "caption_pos_mse": caption_pos_mse.detach(),
        "caption_neg_mse": caption_neg_mse.detach(),
        "caption_ranking_acc": caption_ranking_acc.detach(),
        **loss_parts,
        "pred_v": pred_v.detach(),
        "target_v": target_v.detach(),
        "aux": detach_aux(aux),
        **diag,
        **stats,
    }


def detach_aux(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {k: detach_aux(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detach_aux(v) for v in value)
    return value


def _maybe_blank_caption_batch(batch: dict[str, Any], *, p: float) -> dict[str, Any]:
    if p <= 0.0 or "caption" not in batch:
        return batch
    captions = list(batch["caption"])
    if not captions:
        return batch
    # Sample-level dropout: each caption is independently blanked.
    mask = torch.rand(len(captions)) < p
    if not mask.any():
        return batch
    new_captions = ["" if m else c for m, c in zip(mask, captions)]
    new_batch = dict(batch)
    new_batch["caption"] = new_captions
    # Also blank caption_candidates if present, to prevent text leakage.
    if "caption_candidates" in batch and batch["caption_candidates"] is not None:
        new_batch["caption_candidates"] = [[] if m else c for m, c in zip(mask, batch["caption_candidates"])]
    return new_batch


def _caption_negative_batch(batch: dict[str, Any], *, device: torch.device) -> dict[str, Any]:
    if "caption" not in batch:
        raise ValueError("caption_ranking_weight requires batch['caption']")
    batch_size = len(batch["caption"])
    if batch_size <= 1:
        return batch
    perm = torch.randperm(batch_size, device=device)
    if torch.equal(perm.cpu(), torch.arange(batch_size)):
        perm = torch.roll(perm, shifts=1)
    indices = [int(i) for i in perm.cpu().tolist()]
    out = dict(batch)
    for key in ("caption", "caption_candidates", "captions"):
        value = batch.get(key)
        if value is not None:
            out[key] = [value[i] for i in indices]
    for key in ("caption_embeddings",):
        value = batch.get(key)
        if torch.is_tensor(value) and value.shape[0] == batch_size:
            out[key] = value.index_select(0, perm.to(value.device))
    return out


def _sample_flow_time(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    mode: str = "logit_normal",
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Sample flow-matching time steps t ∈ (0, 1).

    ``logit_normal`` (default) draws from logit-N(mean, std²) following SD3/Flux,
    concentrating mass on the mid-range where the velocity field is hardest to
    learn. ``uniform`` falls back to the original torch.rand behaviour.
    """
    if mode == "uniform":
        return torch.rand(batch_size, device=device, dtype=dtype)
    z = torch.randn(batch_size, device=device, dtype=dtype) * std + mean
    return torch.sigmoid(z).clamp(1e-5, 1.0 - 1e-5)


def _expect_series(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.ndim != 3:
        raise ValueError(f"{name} must be [B,L,C], got {tuple(x.shape)}")
    return x.float() if not torch.is_floating_point(x) else x


def _require_keys(batch: dict[str, Any], keys: tuple[str, ...], *, mode: str) -> None:
    missing = [k for k in keys if k not in batch]
    if missing:
        raise ValueError(f"{mode} batch missing keys: {missing}")


def _require_same_shape(a: torch.Tensor, b: torch.Tensor, a_name: str, b_name: str) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{a_name} and {b_name} must have same shape, got {tuple(a.shape)} vs {tuple(b.shape)}")


def _cfm_loss(sq_error: torch.Tensor, mask: torch.Tensor | None, *, mode: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    global_loss = sq_error.mean()
    parts = {"loss_global": global_loss.detach()}
    if mode == "global" or mask is None:
        return global_loss, parts
    if mode != "balanced":
        raise ValueError(f"Unknown cfm_loss_mode {mode!r}; expected 'global' or 'balanced'")
    mask = mask.to(sq_error.device, dtype=sq_error.dtype)
    if mask.shape == sq_error.shape[:2]:
        mask = mask.unsqueeze(-1).expand_as(sq_error)
    elif mask.shape == (*sq_error.shape[:2], 1):
        mask = mask.expand_as(sq_error)
    elif mask.shape != sq_error.shape:
        raise ValueError(f"balanced CFM requires mask broadcastable to {tuple(sq_error.shape)}, got {tuple(mask.shape)}")
    inv_mask = 1.0 - mask
    inside = (sq_error * mask).sum() / mask.sum().clamp_min(1.0)
    outside = (sq_error * inv_mask).sum() / inv_mask.sum().clamp_min(1.0)
    parts.update({"loss_inside": inside.detach(), "loss_outside": outside.detach()})
    return 0.5 * (inside + outside), parts


def _per_sample_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    sq_error = (pred - target) ** 2
    if mask is None:
        return sq_error.flatten(1).mean(dim=1)
    mask = mask.to(sq_error.device, dtype=sq_error.dtype)
    if mask.shape == sq_error.shape[:2]:
        mask = mask.unsqueeze(-1).expand_as(sq_error)
    elif mask.shape == (*sq_error.shape[:2], 1):
        mask = mask.expand_as(sq_error)
    elif mask.shape != sq_error.shape:
        raise ValueError(f"caption ranking mask must be broadcastable to {tuple(sq_error.shape)}, got {tuple(mask.shape)}")
    return (sq_error * mask).flatten(1).sum(dim=1) / mask.flatten(1).sum(dim=1).clamp_min(1.0)


def _spectral_magnitude_loss(
    pred_x0: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    log_magnitude: bool,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Frequency-domain MSE between predicted and target clean signals.

    Compares per-channel rFFT magnitude spectra along the time axis. This adds
    direct supervision on periodicity / high-frequency structure that pure
    time-domain velocity MSE under-weights. Magnitudes are optionally passed
    through ``log1p`` to compress the dynamic range so that high-frequency bands
    (typically orders of magnitude smaller than the DC / low-frequency
    components) still contribute a meaningful gradient.

    When a mask is supplied, masked-out steps are zeroed before the transform so
    that padded regions do not inject spurious spectral energy. The mask is only
    used to gate which samples/steps are valid; it is not a band mask.

    ``reduction="mean"`` (default) returns a scalar mean over all elements,
    preserving the original behaviour exactly. ``reduction="none"`` returns a
    per-sample ``[B]`` tensor (mean over the freq/channel axes), which lets the
    caller apply a per-sample time weight before averaging.
    """
    if reduction not in {"mean", "none"}:
        raise ValueError(f"Unknown reduction {reduction!r}; expected 'mean' or 'none'")
    if pred_x0.shape != target.shape:
        raise ValueError(f"pred_x0 shape {tuple(pred_x0.shape)} must match target {tuple(target.shape)}")
    if pred_x0.ndim != 3:
        raise ValueError(f"spectral loss expects [B, L, C] tensors, got {tuple(pred_x0.shape)}")
    if pred_x0.shape[1] < 2:
        return pred_x0.new_zeros(pred_x0.shape[0]) if reduction == "none" else pred_x0.new_zeros(())
    if mask is not None:
        mask = mask.to(pred_x0.device, dtype=pred_x0.dtype)
        if mask.shape == pred_x0.shape[:2]:
            mask = mask.unsqueeze(-1)
        if mask.shape == (*pred_x0.shape[:2], 1):
            mask = mask.expand_as(pred_x0)
        elif mask.shape != pred_x0.shape:
            raise ValueError(f"spectral mask must broadcast to {tuple(pred_x0.shape)}, got {tuple(mask.shape)}")
        pred_x0 = pred_x0 * mask
        target = target * mask
    # rFFT along the time axis (dim=1). Cast half precision up for FFT stability.
    fft_dtype = torch.float32 if pred_x0.dtype in {torch.float16, torch.bfloat16} else pred_x0.dtype
    pred_freq = torch.fft.rfft(pred_x0.to(fft_dtype), dim=1)
    target_freq = torch.fft.rfft(target.to(fft_dtype), dim=1)
    pred_mag = pred_freq.abs()
    target_mag = target_freq.abs()
    if log_magnitude:
        pred_mag = torch.log1p(pred_mag)
        target_mag = torch.log1p(target_mag)
    sq = (pred_mag - target_mag).square()
    loss = sq.mean(dim=(1, 2)) if reduction == "none" else sq.mean()
    return loss.to(dtype=pred_x0.dtype)


def _clean_spectral_loss(
    pred_x0: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    loss_type: str,
    fft_sizes: Any,
    distance: str,
    log_magnitude: bool,
    reduction: str = "mean",
) -> torch.Tensor:
    mode = str(loss_type).lower()
    if mode in {"magnitude", "rfft_magnitude", "fft_magnitude"}:
        return _spectral_magnitude_loss(
            pred_x0,
            target,
            mask=mask,
            log_magnitude=log_magnitude,
            reduction=reduction,
        )
    if mode in {"normalized_mrfft", "normalized_multi_resolution_fft", "normalised_mrfft"}:
        return normalized_multi_resolution_fft_loss(
            pred_x0,
            target,
            mask=mask,
            fft_sizes=_parse_spectral_fft_sizes(fft_sizes),
            log_magnitude=log_magnitude,
            distance=distance,
            reduction=reduction,
        )
    raise ValueError(
        "spectral_loss_type must be one of: 'magnitude', 'rfft_magnitude', "
        "'normalized_mrfft'"
    )


def _parse_spectral_fft_sizes(value: Any) -> tuple[int, ...]:
    if value is None:
        return (16, 32, 64, 128)
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",")]
        return tuple(int(part) for part in parts if part)
    try:
        return tuple(int(v) for v in value)
    except TypeError as exc:
        raise TypeError(f"spectral_fft_sizes must be a sequence of ints or comma string, got {value!r}") from exc


def _flow_diagnostics(*, target: torch.Tensor, source: torch.Tensor, pred_v: torch.Tensor, target_v: torch.Tensor) -> dict[str, torch.Tensor]:
    pred_v_rms = pred_v.detach().square().mean().sqrt()
    target_v_rms = target_v.detach().square().mean().sqrt()
    return {
        "source_mse": (source.detach() - target.detach()).square().mean(),
        "source_std": source.detach().std(unbiased=False),
        "target_std": target.detach().std(unbiased=False),
        "pred_v_rms": pred_v_rms,
        "target_v_rms": target_v_rms,
        "pred_target_rms_ratio": pred_v_rms / target_v_rms.clamp_min(1e-8),
        "velocity_cos": (pred_v.detach() * target_v.detach()).mean() / (pred_v_rms * target_v_rms).clamp_min(1e-8),
    }


def _aux_diagnostics(aux: dict[str, Any]) -> dict[str, torch.Tensor]:
    if not isinstance(aux, dict):
        return {}
    out: dict[str, torch.Tensor] = {}
    for key in [
        "regime_entropy",
        "regime_entropy_norm",
        "regime_max_prob",
        "regime_state_weight",
        "operator_gate_entropy",
        "operator_gate_entropy_norm",
        "operator_gate_max_prob",
        "operator_balance_loss",
        "operator_gate_attention_entropy",
        "operator_gate_attention_entropy_norm",
        "operator_gate_attention_text_mass",
        "operator_gate_attention_state_mass",
        "bridge_alignment_loss",
        "text_slot_count",
        "text_slot_count_min",
        "text_slot_count_max",
        "bridge_text_to_state_entropy",
        "bridge_text_to_state_entropy_norm",
        "bridge_text_to_state_max_prob",
        "bridge_state_to_text_entropy",
        "bridge_state_to_text_entropy_norm",
        "bridge_state_to_text_max_prob",
        "bridge_context_norm",
        "bridge_text_context_norm",
        "bridge_state_context_norm",
        "bridge_expert_context_norm",
        "bridge_channel_context_norm",
        "bridge_scale_context_norm",
        "bridge_stage_context_norm",
        "bridge_memory_token_count",
        "bridge_focal_channel_entropy_norm",
        "bridge_focal_expert_entropy_norm",
        "bridge_focal_scale_entropy_norm",
        "bridge_focal_stage_entropy_norm",
        "bridge_alignment_logit_pos",
        "bridge_alignment_logit_std",
        "bridge_alignment_dense",
        "bridge_patch_token_count",
        "bridge_budget_pool_active",
        "bridge_token_budget",
        "bridge_connector_patch_merger",
        "spectral_prompt_entropy",
        "spectral_prompt_entropy_norm",
        "spectral_prompt_gate_low",
        "spectral_prompt_gate_mid",
        "spectral_prompt_gate_high",
        "spectral_prompt_delta_norm",
        "spectral_prompt_raw_delta_norm",
        "spectral_prompt_attn_entropy_norm",
        "tsp_connector_active",
        "tsp_raw_token_count",
        "tsp_budget_token_count",
        "tsp_num_scales",
        "tsp_state_token_norm",
        "tsp_anchor_token_norm",
        "tsp_anchor_token_count",
        "tsp_anchor_norm",
        "tsp_inject_strength_mean",
        "tsp_scale_gate_entropy_norm",
        "tsp_scale_context_norm",
        "tsp_scale_context_norm_global",
        "tsp_scale_entropy_gap",
        "tsp_scale_usage_imbalance",
        "tsp_route_residual_strength",
        "tsp_global_gate_s0",
        "tsp_global_gate_s1",
        "tsp_global_gate_s2",
        "tsp_global_gate_s3",
        "tsp_local_gate_s0",
        "tsp_local_gate_s1",
        "tsp_local_gate_s2",
        "tsp_local_gate_s3",
        "tsp_route_weight_s0",
        "tsp_route_weight_s1",
        "tsp_route_weight_s2",
        "tsp_route_weight_s3",
        "tsp_route_factor_s0",
        "tsp_route_factor_s1",
        "tsp_route_factor_s2",
        "tsp_route_factor_s3",
        "bridge_alignment_tsp_clean_encoded",
        "bridge_alignment_tsp_clean_detached",
    ]:
        val = aux.get(key)
        if torch.is_tensor(val):
            out[key] = val.detach()
    if torch.is_tensor(aux.get("regime_usage")):
        out["regime_usage"] = aux["regime_usage"].detach()
    if torch.is_tensor(aux.get("operator_usage")):
        out["operator_usage"] = aux["operator_usage"].detach()
    return out
