from __future__ import annotations

from typing import Any

from effectcma_flow.models.effectcma_flow import EffectCMAFlow
from effectcma_flow.models.text_encoder import build_text_encoder
from effectcma_flow.models.text_to_ts_flow import TextToTSFlow


def build_model(config: dict[str, Any], *, sequence_length: int, num_channels: int) -> EffectCMAFlow | TextToTSFlow:
    model_cfg = config.get("model", config)
    text_cfg = config.get("text_encoder", {"mode": "hash"})
    task_mode = str(config.get("task", {}).get("mode", "edit")).lower()
    d_model = int(model_cfg.get("d_model", 128))
    text_encoder = build_text_encoder(text_cfg, d_model=d_model)
    kwargs = dict(
        sequence_length=sequence_length,
        num_channels=num_channels,
        patch_len=int(model_cfg.get("patch_len", 6)),
        d_model=d_model,
        num_operators=int(model_cfg.get("num_operators", 7)),
        text_encoder=text_encoder,
        transformer_layers=int(model_cfg.get("transformer_layers", 2)),
        transformer_heads=int(model_cfg.get("transformer_heads", 4)),
        operator_hidden=int(model_cfg.get("operator_hidden", 128)),
        operator_t_dim=int(model_cfg.get("operator_t_dim", 32)),
        operator_max_velocity=float(model_cfg.get("operator_max_velocity", 5.0)),
        operator_depth=int(model_cfg.get("operator_depth", 2)),
        operator_kernel_size=int(model_cfg.get("operator_kernel_size", 3)),
        operator_dropout=float(model_cfg.get("operator_dropout", 0.0)),
        channel_patch_len=_optional_int(model_cfg.get("channel_patch_len")),
        channel_stride=_optional_int(model_cfg.get("channel_stride")),
        channel_temporal_layers=int(model_cfg.get("channel_temporal_layers", 1)),
        channel_layers=int(model_cfg.get("channel_layers", 1)),
        channel_heads=_optional_int(model_cfg.get("channel_heads")),
        channel_dropout=float(model_cfg.get("channel_dropout", 0.0)),
        mapper_field_rank=int(model_cfg.get("mapper_field_rank", 4)),
        mapper_slot_layers=int(model_cfg.get("mapper_slot_layers", 1)),
        mapper_slot_heads=_optional_int(model_cfg.get("mapper_slot_heads")),
        mapper_dropout=float(model_cfg.get("mapper_dropout", 0.0)),
        mapper_normalizer=str(model_cfg.get("mapper_normalizer", "softmax")),
        mapper_bounded_field_gate=bool(model_cfg.get("mapper_bounded_field_gate", True)),
        mapper_flow_time_condition=bool(model_cfg.get("mapper_flow_time_condition", True)),
        operator_context_film=bool(model_cfg.get("operator_context_film", True)),
        operator_norm=str(model_cfg.get("operator_norm", "group")),
    )
    if task_mode == "edit":
        return EffectCMAFlow(**kwargs)
    if task_mode == "text2ts":
        return TextToTSFlow(**kwargs)
    raise ValueError(f"Unknown task.mode {task_mode!r}; expected 'text2ts' or 'edit'")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
