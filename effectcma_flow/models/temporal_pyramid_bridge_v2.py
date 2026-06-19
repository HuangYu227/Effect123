from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class PyramidLevel:
    """Container for one temporal-pyramid level.

    tokens: [B, C, N, D]
    centers: [N] normalized temporal centers in [0, 1].
    patch_len: patch length used to produce this level.
    """

    tokens: torch.Tensor
    centers: torch.Tensor
    patch_len: int


def _choose_heads(d_model: int, num_heads: int) -> int:
    heads = max(1, min(int(num_heads), int(d_model)))
    while d_model % heads != 0 and heads > 1:
        heads -= 1
    return heads


def _ffn(d_model: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, 4 * d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(4 * d_model, d_model),
        nn.Dropout(dropout),
    )


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return tokens.mean(dim=1)
    mask = mask.to(device=tokens.device, dtype=tokens.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (tokens * mask[..., None]).sum(dim=1) / denom


class TemporalConvBlock(nn.Module):
    """Depthwise temporal smoothing + FFN for [B, C, N, D] tokens.

    This is the 1D analogue of the 3x3 smoothing conv in FPN. The depthwise
    convolution preserves per-feature channels and avoids the heavy cost of full
    token attention at every pyramid level.
    """

    def __init__(self, d_model: int, dropout: float = 0.0, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for length-preserving smoothing")
        self.norm = nn.LayerNorm(d_model)
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model)
        self.pwconv = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=1), nn.GELU(), nn.Dropout(dropout))
        self.ffn = _ffn(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"TemporalConvBlock expects [B,C,N,D], got {tuple(x.shape)}")
        b, c, n, d = x.shape
        y = self.norm(x).reshape(b * c, n, d).transpose(1, 2)
        y = self.pwconv(self.dwconv(y)).transpose(1, 2).reshape(b, c, n, d)
        x = x + y
        return x + self.ffn(x)


class TemporalPatchStem(nn.Module):
    """Patchifies a noisy time-series state into scale-specific temporal tokens.

    Unlike a plain unfold+linear projection, the stem augments raw patch values
    with local mean/std/slope/roughness statistics, flow-time embedding, channel
    identity, scale identity, and normalized temporal center embedding. This keeps
    the connector signal-rich without adding any auxiliary meta loss.
    """

    def __init__(
        self,
        *,
        patch_len: int,
        scale_index: int,
        d_model: int,
        num_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if patch_len <= 0:
            raise ValueError("patch_len must be positive")
        self.patch_len = int(patch_len)
        self.scale_index = int(scale_index)
        self.d_model = int(d_model)
        self.num_channels = int(num_channels)
        stat_dim = 4  # mean, std, slope, roughness
        self.value_proj = nn.Sequential(
            nn.LayerNorm(self.patch_len + stat_dim),
            nn.Linear(self.patch_len + stat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.channel_embed = nn.Embedding(num_channels, d_model)
        self.scale_embed = nn.Parameter(torch.zeros(d_model))
        self.center_proj = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.time_proj = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.local_block = TemporalConvBlock(d_model, dropout=dropout)
        nn.init.normal_(self.scale_embed, std=0.02)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> PyramidLevel:
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be [B,L,C], got {tuple(x_t.shape)}")
        b, length, channels = x_t.shape
        if channels != self.num_channels:
            raise ValueError(f"expected {self.num_channels} channels, got {channels}")
        if t.shape != (b,):
            raise ValueError(f"t must be [B], got {tuple(t.shape)}")
        dtype = x_t.dtype
        device = x_t.device
        p = min(self.patch_len, length)
        pad = (p - length % p) % p
        x_cf = x_t.transpose(1, 2)  # [B,C,L]
        if pad:
            x_cf = F.pad(x_cf, (0, pad), mode="replicate")
        patches = x_cf.unfold(dimension=-1, size=p, step=p)  # [B,C,N,P]
        if p < self.patch_len:
            patches = F.pad(patches, (0, self.patch_len - p), mode="replicate")
        # local statistics in the original dtype; std/roughness are clamped to avoid NaNs in fp16
        mean = patches.mean(dim=-1, keepdim=True)
        std = patches.float().std(dim=-1, unbiased=False, keepdim=True).to(dtype).clamp_min(1e-6)
        slope = (patches[..., -1:] - patches[..., :1]) / max(float(p - 1), 1.0)
        if self.patch_len > 1:
            diff = patches[..., 1:] - patches[..., :-1]
            rough = diff.float().square().mean(dim=-1, keepdim=True).sqrt().to(dtype)
        else:
            rough = torch.zeros_like(mean)
        features = torch.cat([patches, mean, std, slope, rough], dim=-1)
        tokens = self.value_proj(features)

        num_tokens = tokens.shape[2]
        start = torch.arange(num_tokens, device=device, dtype=dtype) * float(p)
        centers = ((start + 0.5 * float(p)) / max(float(length), 1.0)).clamp(0.0, 1.0)
        center_emb = self.center_proj(centers[:, None]).to(dtype=dtype)
        channel_ids = torch.arange(channels, device=device)
        channel_emb = self.channel_embed(channel_ids).to(dtype=dtype)
        time_emb = self.time_proj(t[:, None].to(dtype=dtype))[:, None, None, :]
        tokens = tokens + center_emb[None, None, :, :] + channel_emb[None, :, None, :] + self.scale_embed.to(dtype=dtype)[None, None, None, :] + time_emb
        tokens = self.local_block(tokens)
        return PyramidLevel(tokens=tokens, centers=centers.detach(), patch_len=self.patch_len)


class TemporalFPNPAN(nn.Module):
    """1D FPN/PAN over temporal pyramid levels.

    FPN top-down path injects coarse trend/global semantics into fine temporal
    tokens. PAN bottom-up path lets local spikes and short-period events affect
    coarse/global tokens. This module intentionally stays attention-light; the
    deeper cross-scale reasoning is delegated to CrossScaleAnchorInteraction.
    """

    def __init__(self, d_model: int, num_levels: int, dropout: float = 0.0, use_topdown: bool = True, use_bottomup: bool = True) -> None:
        super().__init__()
        self.num_levels = int(num_levels)
        self.use_topdown = bool(use_topdown)
        self.use_bottomup = bool(use_bottomup)
        self.lateral = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model)) for _ in range(num_levels)])
        self.top_smooth = nn.ModuleList([TemporalConvBlock(d_model, dropout=dropout) for _ in range(num_levels)])
        self.bottom_smooth = nn.ModuleList([TemporalConvBlock(d_model, dropout=dropout) for _ in range(num_levels)])
        self.resample_smooth = nn.ModuleList([TemporalConvBlock(d_model, dropout=dropout) for _ in range(num_levels)])

    @staticmethod
    def _resample(x: torch.Tensor, target_len: int) -> torch.Tensor:
        b, c, n, d = x.shape
        if n == target_len:
            return x
        y = x.reshape(b * c, n, d).transpose(1, 2)
        y = F.interpolate(y.float(), size=int(target_len), mode="linear", align_corners=False).to(dtype=x.dtype)
        return y.transpose(1, 2).reshape(b, c, int(target_len), d)

    def forward(self, levels: list[PyramidLevel]) -> list[PyramidLevel]:
        if len(levels) != self.num_levels:
            raise ValueError(f"expected {self.num_levels} levels, got {len(levels)}")
        feats = [self.lateral[i](level.tokens) for i, level in enumerate(levels)]
        if self.use_topdown:
            p = [None for _ in feats]
            p[-1] = self.top_smooth[-1](feats[-1])
            for i in range(self.num_levels - 2, -1, -1):
                top = self._resample(p[i + 1], feats[i].shape[2])
                p[i] = self.top_smooth[i](feats[i] + top)
        else:
            p = [self.top_smooth[i](feats[i]) for i in range(self.num_levels)]
        if self.use_bottomup:
            q = [None for _ in p]
            q[0] = self.bottom_smooth[0](p[0])
            for i in range(1, self.num_levels):
                down = self._resample(q[i - 1], p[i].shape[2])
                q[i] = self.bottom_smooth[i](p[i] + down)
        else:
            q = p
        return [PyramidLevel(tokens=q[i], centers=levels[i].centers, patch_len=levels[i].patch_len) for i in range(self.num_levels)]


class BiasedMultiheadAttention(nn.Module):
    """Multi-head attention with temporal-center and scale-pair biases.

    The temporal bias encourages cross-scale tokens to attend to aligned regions
    of the sequence while still allowing global mixing. Scale-pair bias is learned
    and can discover useful interactions such as fine<->coarse or mid<->global.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        num_scales: int | None = None,
        temporal_bias_tau: float = 0.25,
    ) -> None:
        super().__init__()
        heads = _choose_heads(d_model, num_heads)
        self.d_model = int(d_model)
        self.num_heads = heads
        self.head_dim = self.d_model // heads
        self.scale = self.head_dim ** -0.5
        self.temporal_bias_tau = float(max(temporal_bias_tau, 1e-4))
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        if num_scales is not None and num_scales > 0:
            self.scale_pair_bias = nn.Parameter(torch.zeros(heads, int(num_scales), int(num_scales)))
        else:
            self.scale_pair_bias = None

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        *,
        query_centers: torch.Tensor | None = None,
        key_centers: torch.Tensor | None = None,
        query_scale_ids: torch.Tensor | None = None,
        key_scale_ids: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query.ndim != 3 or key_value.ndim != 3:
            raise ValueError("query/key_value must be [B,N,D]")
        b, nq, d = query.shape
        nk = key_value.shape[1]
        if key_value.shape[0] != b or key_value.shape[-1] != d:
            raise ValueError("query and key_value batch/dim must match")
        q = self.q_proj(query).view(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if query_centers is not None and key_centers is not None:
            qc = query_centers.to(device=query.device, dtype=logits.dtype)
            kc = key_centers.to(device=query.device, dtype=logits.dtype)
            if qc.ndim == 1:
                qc = qc[None, :].expand(b, -1)
            if kc.ndim == 1:
                kc = kc[None, :].expand(b, -1)
            if qc.shape != (b, nq) or kc.shape != (b, nk):
                raise ValueError(f"center shapes must be {(b,nq)} and {(b,nk)}, got {tuple(qc.shape)}, {tuple(kc.shape)}")
            center_bias = -torch.abs(qc[:, :, None] - kc[:, None, :]) / self.temporal_bias_tau
            logits = logits + center_bias[:, None, :, :]
        if self.scale_pair_bias is not None and query_scale_ids is not None and key_scale_ids is not None:
            qs = query_scale_ids.to(device=query.device, dtype=torch.long)
            ks = key_scale_ids.to(device=query.device, dtype=torch.long)
            pair = self.scale_pair_bias[:, qs[:, None], ks[None, :]]  # [H,Nq,Nk]
            logits = logits + pair[None, :, :, :]
        if key_padding_mask is not None:
            mask = key_padding_mask.to(device=query.device, dtype=torch.bool)
            if mask.shape != (b, nk):
                raise ValueError(f"key_padding_mask must be {(b,nk)}, got {tuple(mask.shape)}")
            logits = logits.masked_fill(mask[:, None, None, :], -torch.finfo(logits.dtype).max)
        attn = torch.softmax(logits.float(), dim=-1).to(dtype=query.dtype)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, nq, d)
        return self.out_proj(out)


class CrossScaleAnchorBlock(nn.Module):
    """Text-aware anchor transformer block for explicit cross-scale interaction."""

    def __init__(self, d_model: int, num_heads: int, num_scales: int, dropout: float = 0.0, temporal_bias_tau: float = 0.25) -> None:
        super().__init__()
        self.anchor_norm = nn.LayerNorm(d_model)
        self.self_attn = BiasedMultiheadAttention(d_model, num_heads, dropout, num_scales=num_scales, temporal_bias_tau=temporal_bias_tau)
        self.text_norm = nn.LayerNorm(d_model)
        self.text_cross = nn.MultiheadAttention(d_model, _choose_heads(d_model, num_heads), dropout=dropout, batch_first=True)
        self.text_gate = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.ffn = _ffn(d_model, dropout)

    def forward(
        self,
        anchors: torch.Tensor,
        *,
        centers: torch.Tensor,
        scale_ids: torch.Tensor,
        slot_tokens: torch.Tensor | None,
        slot_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        a_norm = self.anchor_norm(anchors)
        delta = self.self_attn(
            a_norm,
            a_norm,
            query_centers=centers,
            key_centers=centers,
            query_scale_ids=scale_ids,
            key_scale_ids=scale_ids,
        )
        anchors = anchors + delta
        if slot_tokens is not None:
            key_padding_mask = None
            if slot_mask is not None:
                valid = slot_mask.to(dtype=torch.bool, device=slot_tokens.device)
                empty = ~valid.any(dim=1)
                if empty.any():
                    valid = valid.clone()
                    valid[empty, 0] = True
                key_padding_mask = ~valid
            text_delta, _ = self.text_cross(
                query=self.anchor_norm(anchors),
                key=self.text_norm(slot_tokens),
                value=slot_tokens,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            gate = self.text_gate(torch.cat([anchors, text_delta], dim=-1))
            anchors = anchors + gate * text_delta
        anchors = anchors + self.ffn(anchors)
        return anchors


class CrossScaleAnchorInteraction(nn.Module):
    """Anchor-based cross-scale attention with temporal alignment bias.

    This is the main hard upgrade over additive FPN/PAN. Each scale contributes a
    small set of learned anchor queries. Anchors first read their own scale with
    temporal-center bias, then interact across scales and text, and finally inject
    the cross-scale summaries back into all tokens.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_scales: int,
        anchor_tokens: int = 8,
        num_heads: int = 4,
        layers: int = 2,
        dropout: float = 0.0,
        temporal_bias_tau: float = 0.25,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_scales = int(num_scales)
        self.anchor_tokens = int(anchor_tokens)
        if self.anchor_tokens <= 0:
            raise ValueError("anchor_tokens must be positive")
        self.anchor_queries = nn.Parameter(torch.randn(self.num_scales, self.anchor_tokens, d_model) * 0.02)
        self.scale_embed = nn.Parameter(torch.randn(self.num_scales, d_model) * 0.02)
        self.anchor_pool = BiasedMultiheadAttention(d_model, num_heads, dropout, num_scales=num_scales, temporal_bias_tau=temporal_bias_tau)
        self.anchor_blocks = nn.ModuleList([
            CrossScaleAnchorBlock(d_model, num_heads, num_scales, dropout=dropout, temporal_bias_tau=temporal_bias_tau)
            for _ in range(int(layers))
        ])
        self.token_inject = BiasedMultiheadAttention(d_model, num_heads, dropout, num_scales=num_scales, temporal_bias_tau=temporal_bias_tau)
        self.token_norm = nn.LayerNorm(d_model)
        # Negative init: sigmoid(-2) ~= 0.12, so the new branch is effective but does not destroy U4 warm start.
        self.inject_logit = nn.Parameter(torch.full((self.num_scales,), -2.0))

    @staticmethod
    def _flatten_level(level: PyramidLevel) -> tuple[torch.Tensor, torch.Tensor]:
        x = level.tokens
        b, c, n, d = x.shape
        tokens = x.reshape(b, c * n, d)
        centers = level.centers.to(device=x.device, dtype=x.dtype).repeat(c)
        return tokens, centers

    def forward(
        self,
        levels: list[PyramidLevel],
        *,
        slot_tokens: torch.Tensor | None = None,
        slot_mask: torch.Tensor | None = None,
    ) -> tuple[list[PyramidLevel], torch.Tensor, dict[str, torch.Tensor]]:
        if len(levels) != self.num_scales:
            raise ValueError(f"expected {self.num_scales} levels, got {len(levels)}")
        b = levels[0].tokens.shape[0]
        dtype = levels[0].tokens.dtype
        device = levels[0].tokens.device
        anchor_list: list[torch.Tensor] = []
        anchor_centers: list[torch.Tensor] = []
        anchor_scale_ids: list[torch.Tensor] = []
        for s, level in enumerate(levels):
            tokens, centers = self._flatten_level(level)
            q_centers = torch.linspace(0.0, 1.0, self.anchor_tokens, device=device, dtype=dtype)
            queries = self.anchor_queries[s].to(device=device, dtype=dtype)[None].expand(b, -1, -1)
            queries = queries + self.scale_embed[s].to(dtype=dtype, device=device)[None, None, :]
            scale_q = torch.full((self.anchor_tokens,), s, device=device, dtype=torch.long)
            scale_k = torch.full((tokens.shape[1],), s, device=device, dtype=torch.long)
            anchors = queries + self.anchor_pool(
                queries,
                tokens,
                query_centers=q_centers,
                key_centers=centers,
                query_scale_ids=scale_q,
                key_scale_ids=scale_k,
            )
            anchor_list.append(anchors)
            anchor_centers.append(q_centers)
            anchor_scale_ids.append(scale_q)
        anchors = torch.cat(anchor_list, dim=1)
        centers_all = torch.cat(anchor_centers, dim=0)
        scale_ids_all = torch.cat(anchor_scale_ids, dim=0)
        for block in self.anchor_blocks:
            anchors = block(anchors, centers=centers_all, scale_ids=scale_ids_all, slot_tokens=slot_tokens, slot_mask=slot_mask)

        refined_levels: list[PyramidLevel] = []
        inject_strengths = torch.sigmoid(self.inject_logit).to(dtype=dtype, device=device)
        for s, level in enumerate(levels):
            b, c, n, d = level.tokens.shape
            tokens, centers = self._flatten_level(level)
            q_scale = torch.full((tokens.shape[1],), s, device=device, dtype=torch.long)
            update = self.token_inject(
                self.token_norm(tokens),
                anchors,
                query_centers=centers,
                key_centers=centers_all,
                query_scale_ids=q_scale,
                key_scale_ids=scale_ids_all,
            )
            tokens = tokens + inject_strengths[s] * update
            refined_levels.append(PyramidLevel(tokens=tokens.reshape(b, c, n, d), centers=level.centers, patch_len=level.patch_len))
        aux = {
            "tsp_anchor_token_count": torch.tensor(float(anchors.shape[1]), device=device, dtype=dtype),
            "tsp_anchor_norm": anchors.detach().norm(dim=-1).mean(),
            "tsp_inject_strength_mean": inject_strengths.detach().mean(),
        }
        for s, strength in enumerate(inject_strengths):
            aux[f"tsp_inject_strength_s{s}"] = strength.detach()
        return refined_levels, anchors, aux


class TextGuidedTokenScaleRouter(nn.Module):
    """Text-guided token-level scale routing.

    It combines a global caption-level scale preference with local token gates,
    allowing a sample to use coarse/global tokens for trend while selecting fine
    tokens around caption-indicated local events.
    """

    def __init__(self, d_model: int, num_scales: int, dropout: float = 0.0, temperature: float = 0.7) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_scales = int(num_scales)
        self.temperature = float(max(temperature, 1e-4))
        self.scale_embed = nn.Parameter(torch.randn(num_scales, d_model) * 0.02)
        self.global_gate = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.local_gate = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.scale_context_norm = nn.LayerNorm(d_model)
        # Residual route strength keeps all TSP tokens available. The 0.5 cap
        # means even a near-zero route preserves at least half of the token
        # amplitude, avoiding the non-residual gate collapse seen in ablations.
        self.route_residual_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        levels: list[PyramidLevel],
        *,
        text_context: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        if len(levels) != self.num_scales:
            raise ValueError(f"expected {self.num_scales} levels, got {len(levels)}")
        b = levels[0].tokens.shape[0]
        dtype = levels[0].tokens.dtype
        device = levels[0].tokens.device
        if text_context is None:
            text_context = torch.zeros(b, self.d_model, device=device, dtype=dtype)
        else:
            text_context = text_context.to(device=device, dtype=dtype)
        scale_ctx = []
        for level in levels:
            scale_ctx.append(level.tokens.mean(dim=(1, 2)))
        scale_ctx_t = self.scale_context_norm(torch.stack(scale_ctx, dim=1))
        text_expand = text_context[:, None, :].expand(-1, self.num_scales, -1)
        global_logits = self.global_gate(torch.cat([text_expand, scale_ctx_t, text_expand * scale_ctx_t], dim=-1)).squeeze(-1)
        global_gate = torch.softmax(global_logits / self.temperature, dim=-1)

        weighted_tokens: list[torch.Tensor] = []
        local_means = []
        route_weight_means = []
        route_factor_means = []
        route_strength = (0.5 * torch.sigmoid(self.route_residual_logit)).to(device=device, dtype=dtype)
        for s, level in enumerate(levels):
            tokens = level.tokens.reshape(b, -1, self.d_model)
            text_tok = text_context[:, None, :].expand(-1, tokens.shape[1], -1)
            scale_tok = self.scale_embed[s].to(device=device, dtype=dtype)[None, None, :].expand_as(tokens)
            local_logits = self.local_gate(torch.cat([tokens, text_tok, tokens * text_tok, scale_tok], dim=-1))
            local = torch.sigmoid(local_logits)
            local_means.append(local.detach().mean())
            # Normalize the expected cold-start route weight around 1.0:
            # sigmoid(local) ~= 0.5 and softmax(global) ~= 1/S, so
            # 2 * S * local * global ~= 1. The residual form bounds collapse:
            # even a zero route keeps factor ~= 1 - route_strength.
            route_weight = local * global_gate[:, s].view(b, 1, 1).to(dtype=dtype) * float(2 * self.num_scales)
            route_factor = 1.0 + route_strength * (route_weight - 1.0)
            route_weight_means.append(route_weight.detach().mean())
            route_factor_means.append(route_factor.detach().mean())
            weighted_tokens.append(tokens * route_factor)
        entropy = -(global_gate.float().clamp_min(1e-8) * global_gate.float().clamp_min(1e-8).log()).sum(dim=-1)
        entropy_norm = entropy / math.log(max(self.num_scales, 2))
        mean_usage = global_gate.float().mean(dim=0)
        uniform = torch.full_like(mean_usage, 1.0 / float(self.num_scales))
        aux: dict[str, torch.Tensor] = {
            "tsp_scale_gate_entropy_norm": entropy_norm.detach().mean().to(dtype=dtype),
            "tsp_scale_context_norm": scale_ctx_t.detach().norm(dim=-1).mean(),
            "tsp_scale_entropy_gap": (1.0 - entropy_norm.detach().mean()).clamp_min(0.0).to(dtype=dtype),
            "tsp_scale_usage_imbalance": (mean_usage.detach() - uniform).square().mean().to(dtype=dtype),
            "tsp_route_residual_strength": route_strength.detach(),
        }
        for s in range(self.num_scales):
            aux[f"tsp_global_gate_s{s}"] = global_gate[:, s].detach().mean().to(dtype=dtype)
            aux[f"tsp_local_gate_s{s}"] = local_means[s].to(device=device, dtype=dtype)
            aux[f"tsp_route_weight_s{s}"] = route_weight_means[s].to(device=device, dtype=dtype)
            aux[f"tsp_route_factor_s{s}"] = route_factor_means[s].to(device=device, dtype=dtype)
        return weighted_tokens, scale_ctx_t, aux


class FixedBudgetAttentionPool(nn.Module):
    """Always returns exactly token_budget tokens via learned-query attention."""

    def __init__(self, d_model: int, token_budget: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.token_budget = int(token_budget)
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        heads = _choose_heads(d_model, num_heads)
        self.queries = nn.Parameter(torch.randn(self.token_budget, d_model) * 0.02)
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model, dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,N,D], got {tuple(tokens.shape)}")
        q = self.queries[None].expand(tokens.shape[0], -1, -1).to(device=tokens.device, dtype=tokens.dtype)
        pooled, _ = self.attn(query=self.query_norm(q), key=self.memory_norm(tokens), value=tokens, need_weights=False)
        pooled = self.out_norm(q + pooled)
        return pooled + self.ffn(pooled)


class TemporalSemanticPyramidConnectorV2(nn.Module):
    """Text-Guided Cross-Scale Temporal Semantic Pyramid connector.

    This connector is designed as a drop-in replacement for the state-token
    encoder inside CrossModalConditionBridge/TSPatchMergerConnector. It keeps the
    training objective unchanged and outputs budgeted state tokens consumed by
    the existing U4 bidirectional text-state attention.

    Main components:
    1. Multi-scale temporal patch stems at the input layer.
    2. FPN/PAN propagation for coarse-to-fine trend and fine-to-coarse events.
    3. Anchor-based cross-scale attention with temporal center bias.
    4. Text-guided token-level scale routing.
    5. Fixed-budget Perceiver pooling for controlled memory cost.
    """

    def __init__(
        self,
        *,
        sequence_length: int,
        num_channels: int,
        d_model: int,
        patch_lens: Sequence[int] = (4, 8, 16, 32),
        token_budget: int = 128,
        anchor_tokens: int = 8,
        cross_scale_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.05,
        use_topdown: bool = True,
        use_bottomup: bool = True,
        use_text_routing: bool = True,
        temporal_bias_tau: float = 0.25,
        gate_temperature: float = 0.7,
    ) -> None:
        super().__init__()
        if sequence_length <= 0 or num_channels <= 0 or d_model <= 0:
            raise ValueError("sequence_length, num_channels, and d_model must be positive")
        patch_lens = tuple(int(p) for p in patch_lens)
        if not patch_lens or any(p <= 0 for p in patch_lens):
            raise ValueError("patch_lens must be a non-empty sequence of positive integers")
        self.sequence_length = int(sequence_length)
        self.num_channels = int(num_channels)
        self.d_model = int(d_model)
        self.patch_lens = patch_lens
        self.num_scales = len(patch_lens)
        self.token_budget = int(token_budget)
        self.use_text_routing = bool(use_text_routing)
        self.stems = nn.ModuleList([
            TemporalPatchStem(patch_len=p, scale_index=i, d_model=d_model, num_channels=num_channels, dropout=dropout)
            for i, p in enumerate(patch_lens)
        ])
        self.fpn_pan = TemporalFPNPAN(d_model, self.num_scales, dropout=dropout, use_topdown=use_topdown, use_bottomup=use_bottomup)
        self.cross_scale = CrossScaleAnchorInteraction(
            d_model=d_model,
            num_scales=self.num_scales,
            anchor_tokens=anchor_tokens,
            num_heads=num_heads,
            layers=cross_scale_layers,
            dropout=dropout,
            temporal_bias_tau=temporal_bias_tau,
        )
        self.router = TextGuidedTokenScaleRouter(d_model, self.num_scales, dropout=dropout, temperature=gate_temperature)
        self.budget_pool = FixedBudgetAttentionPool(d_model, token_budget, num_heads, dropout=dropout)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        slot_tokens: torch.Tensor | None = None,
        slot_mask: torch.Tensor | None = None,
        text_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be [B,L,C], got {tuple(x_t.shape)}")
        b, length, channels = x_t.shape
        if (length, channels) != (self.sequence_length, self.num_channels):
            raise ValueError(f"expected x_t [B,{self.sequence_length},{self.num_channels}], got {tuple(x_t.shape)}")
        if t.shape != (b,):
            raise ValueError(f"t must be [B], got {tuple(t.shape)}")
        if slot_tokens is not None:
            if slot_tokens.ndim != 3 or slot_tokens.shape[0] != b or slot_tokens.shape[-1] != self.d_model:
                raise ValueError(f"slot_tokens must be [B,J,{self.d_model}], got {tuple(slot_tokens.shape)}")
            if slot_mask is not None and slot_mask.shape != slot_tokens.shape[:2]:
                raise ValueError(f"slot_mask must be {tuple(slot_tokens.shape[:2])}, got {tuple(slot_mask.shape)}")
            if text_context is None:
                text_context = _masked_mean(slot_tokens, slot_mask)
        elif text_context is None:
            text_context = torch.zeros(b, self.d_model, device=x_t.device, dtype=x_t.dtype)

        levels = [stem(x_t, t) for stem in self.stems]
        raw_token_count = sum(level.tokens.shape[1] * level.tokens.shape[2] for level in levels)
        levels = self.fpn_pan(levels)
        levels, anchors, aux_anchor = self.cross_scale(levels, slot_tokens=slot_tokens, slot_mask=slot_mask)
        weighted_tokens, scale_ctx, aux_router = self.router(levels, text_context=text_context if self.use_text_routing else None)
        all_tokens = torch.cat(weighted_tokens, dim=1)
        state_tokens = self.out_norm(self.budget_pool(all_tokens))
        dtype = x_t.dtype
        device = x_t.device
        aux: dict[str, torch.Tensor] = {
            "tsp_connector_active": torch.tensor(1.0, device=device, dtype=dtype),
            "tsp_raw_token_count": torch.tensor(float(raw_token_count), device=device, dtype=dtype),
            "tsp_budget_token_count": torch.tensor(float(state_tokens.shape[1]), device=device, dtype=dtype),
            "tsp_num_scales": torch.tensor(float(self.num_scales), device=device, dtype=dtype),
            "tsp_state_token_norm": state_tokens.detach().norm(dim=-1).mean(),
            "tsp_anchor_token_norm": anchors.detach().norm(dim=-1).mean(),
            "tsp_scale_context_norm_global": scale_ctx.detach().norm(dim=-1).mean(),
        }
        for i, level in enumerate(levels):
            aux[f"tsp_level_{i}_tokens_per_channel"] = torch.tensor(float(level.tokens.shape[2]), device=device, dtype=dtype)
            aux[f"tsp_level_{i}_patch_len"] = torch.tensor(float(level.patch_len), device=device, dtype=dtype)
        aux.update(aux_anchor)
        aux.update(aux_router)
        return state_tokens, aux
