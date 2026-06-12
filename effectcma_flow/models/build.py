from __future__ import annotations

from typing import Any

from effectcma_flow.models.effectcma_flow import EffectCMAFlow
from effectcma_flow.models.text_encoder import build_text_encoder
from effectcma_flow.models.text_to_ts_flow import TextToTSFlow


_V61_KEYS = frozenset({
    "use_latent_regime_adapter", "num_regimes", "regime_dim", "regime_hidden_dim",
    "regime_temperature", "regime_append_token", "regime_dropout", "regime_state_weight_mode",
    "router_mode", "operator_gate_temperature", "operator_gate_dropout",
})


def build_model(config: dict[str, Any], *, sequence_length: int, num_channels: int) -> EffectCMAFlow | TextToTSFlow:
    model_cfg = config.get("model", config)
    text_cfg = config.get("text_encoder", {"mode": "hash"})
    task_cfg = config.get("task", {})
    if "mode" not in task_cfg:
        raise ValueError("config.task.mode must be explicitly set to 'text2ts' or 'edit'.")
    task_mode = str(task_cfg["mode"]).lower()
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
        mapper_gate_rescale=model_cfg.get("mapper_gate_rescale", "auto"),
        mapper_operator_router=str(model_cfg.get("mapper_operator_router", "text")),
        mapper_time_segment_scales=model_cfg.get("mapper_time_segment_scales"),
        mapper_router_epsilon=float(model_cfg.get("mapper_router_epsilon", 1e-4)),
        mapper_flow_time_condition=bool(model_cfg.get("mapper_flow_time_condition", True)),
        operator_context_film=bool(model_cfg.get("operator_context_film", True)),
        operator_context_mode=str(model_cfg.get("operator_context_mode", "global")),
        operator_norm=str(model_cfg.get("operator_norm", "group")),
        operator_architecture=str(model_cfg.get("operator_architecture", "homogeneous")),
        operator_channel_heads=int(model_cfg.get("operator_channel_heads", 4)),
        # V6.1 latent regime adapter
        use_latent_regime_adapter=bool(model_cfg.get("use_latent_regime_adapter", False)),
        num_regimes=int(model_cfg.get("num_regimes", 4)),
        regime_dim=_optional_int(model_cfg.get("regime_dim")),
        regime_hidden_dim=_optional_int(model_cfg.get("regime_hidden_dim")),
        regime_temperature=float(model_cfg.get("regime_temperature", 0.7)),
        regime_append_token=bool(model_cfg.get("regime_append_token", True)),
        regime_dropout=float(model_cfg.get("regime_dropout", 0.0)),
        regime_state_weight_mode=str(model_cfg.get("regime_state_weight_mode", "linear_t")),
        # V6.1 routing mode
        router_mode=str(model_cfg.get("router_mode", "legacy")),
        operator_gate_temperature=float(model_cfg.get("operator_gate_temperature", 1.0)),
        operator_gate_dropout=float(model_cfg.get("operator_gate_dropout", 0.0)),
    )
    if task_mode == "edit":
        # EffectCMAFlow does not accept V6.1 regime/gate params.
        edit_kwargs = {k: v for k, v in kwargs.items() if k not in _V61_KEYS}
        return EffectCMAFlow(**edit_kwargs)
    if task_mode == "text2ts":
        return TextToTSFlow(**kwargs)
    raise ValueError(f"Unknown task.mode {task_mode!r}; expected 'text2ts' or 'edit'")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
