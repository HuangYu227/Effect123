from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class TimestepAwareCrossAttention(nn.Module):
    """SDPA cross-attention from learned focal queries to text/state memory.

    A small time-dependent logit scale lets the bridge change how sharply it
    reads conditions at different flow stages without introducing a new loss.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        heads = max(1, min(int(num_heads), int(d_model)))
        while d_model % heads != 0 and heads > 1:
            heads -= 1
        self.num_heads = heads
        self.head_dim = int(d_model) // heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = float(dropout)
        self.time_logit_scale = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, heads))

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 3 or memory.ndim != 3:
            raise ValueError("query and memory must be [B,N,D]")
        batch, q_len, d_model = query.shape
        if memory.shape[0] != batch or memory.shape[-1] != d_model:
            raise ValueError("query and memory batch/dim must match")
        if memory_mask.shape != memory.shape[:2]:
            raise ValueError(f"memory_mask must have shape {tuple(memory.shape[:2])}, got {tuple(memory_mask.shape)}")
        mask = memory_mask.to(device=memory.device, dtype=torch.bool)
        empty = ~mask.any(dim=1)
        if empty.any():
            mask = mask.clone()
            mask[empty, 0] = True

        q = self.q_proj(query).view(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(batch, memory.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(batch, memory.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        stage_scale = 1.0 + 0.5 * torch.tanh(self.time_logit_scale(t[:, None].to(query.dtype)))
        q = q * stage_scale[:, :, None, None]
        attn_mask = mask[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        logits = torch.matmul(q.detach(), k.detach().transpose(-2, -1)) * self.scale
        logits = logits.masked_fill(~mask[:, None, None, :], -torch.finfo(logits.dtype).max)
        attn = torch.softmax(logits, dim=-1)
        out = out.transpose(1, 2).reshape(batch, q_len, d_model)
        return self.out_proj(out), attn


class LatentGridBlock(nn.Module):
    """Lightweight latent feature block over time and variable order."""

    def __init__(self, d_model: int, num_channels: int, dropout: float) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.time_norm = nn.LayerNorm(d_model)
        self.time_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.time_mix = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_mixer = nn.Linear(self.num_channels, self.num_channels, bias=False)
        self.ffn = _ffn(d_model, dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 4:
            raise ValueError(f"latent grid must be [B,L,C,D], got {tuple(h.shape)}")
        batch, length, channels, d_model = h.shape
        if channels != self.num_channels:
            raise ValueError(f"expected {self.num_channels} channels, got {channels}")
        x = self.time_norm(h).permute(0, 2, 3, 1).reshape(batch * channels, d_model, length)
        x = self.time_mix(self.time_conv(x)).reshape(batch, channels, d_model, length).permute(0, 3, 1, 2)
        h = h + x
        y = self.channel_norm(h).permute(0, 1, 3, 2)
        y = self.channel_mixer(y).permute(0, 1, 3, 2)
        h = h + y
        return h + self.ffn(h)


class TokenBudgetPool(nn.Module):
    """Perceiver-style fixed-budget pooling used only when tokens exceed the budget."""

    def __init__(self, d_model: int, token_budget: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.token_budget = int(token_budget)
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        self.queries = nn.Parameter(torch.randn(self.token_budget, d_model) * 0.02)
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model, dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,N,D], got {tuple(tokens.shape)}")
        if tokens.shape[1] <= self.token_budget:
            return self.out_norm(tokens)
        query = self.queries[None].expand(tokens.shape[0], -1, -1).to(dtype=tokens.dtype, device=tokens.device)
        pooled, _ = self.attn(
            query=self.query_norm(query),
            key=self.memory_norm(tokens),
            value=tokens,
            need_weights=False,
        )
        pooled = self.out_norm(query + pooled)
        return pooled + self.ffn(pooled)


class AttentionPool(nn.Module):
    """Single-query attention pooling with an optional validity mask."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.query_norm = nn.LayerNorm(d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,N,D], got {tuple(tokens.shape)}")
        key_padding_mask = _safe_key_padding_mask(mask) if mask is not None else None
        query = self.query[None].expand(tokens.shape[0], -1, -1).to(dtype=tokens.dtype, device=tokens.device)
        pooled, _ = self.attn(
            query=self.query_norm(query),
            key=self.token_norm(tokens),
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.out_norm((query + pooled).squeeze(1))


class MultiQueryAttentionPool(nn.Module):
    """Pool a token set into a fixed number of region tokens via learned queries.

    Unlike :class:`AttentionPool` (a single query / global vector), this keeps
    ``num_regions`` distinct learned queries so the output is a small set of
    region embeddings suitable for token/region-level (dense) alignment.
    """

    def __init__(self, d_model: int, num_regions: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_regions = int(num_regions)
        if self.num_regions <= 0:
            raise ValueError("num_regions must be positive")
        self.queries = nn.Parameter(torch.randn(self.num_regions, d_model) * 0.02)
        self.query_norm = nn.LayerNorm(d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,N,D], got {tuple(tokens.shape)}")
        key_padding_mask = _safe_key_padding_mask(mask) if mask is not None else None
        query = self.queries[None].expand(tokens.shape[0], -1, -1).to(dtype=tokens.dtype, device=tokens.device)
        pooled, _ = self.attn(
            query=self.query_norm(query),
            key=self.token_norm(tokens),
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.out_norm(query + pooled)


class CrossAttentionPool(nn.Module):
    """Pool memory with a caller-provided set of query tokens."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(d_model)
        self.ffn = _ffn(d_model, dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if query.ndim != 3 or memory.ndim != 3:
            raise ValueError("query and memory must be [B,N,D]")
        out, _ = self.attn(
            query=self.query_norm(query),
            key=self.memory_norm(memory),
            value=memory,
            need_weights=False,
        )
        out = self.out_norm(query + out)
        return out + self.ffn(out)


class TSPatchMergerConnector(nn.Module):
    """Qwen-style patch merger connector for latent time-series state tokens."""

    def __init__(
        self,
        *,
        d_model: int,
        num_channels: int,
        sequence_length: int,
        num_experts: int,
        temporal_merge: int = 4,
        channel_merge: int = 1,
        token_budget: int = 96,
        num_heads: int = 4,
        dropout: float = 0.0,
        alignment_mode: str = "siglip",
        alignment_temperature: float = 0.07,
        alignment_dense: bool = False,
        alignment_regions: int = 8,
        alignment_target: str = "state",
        alignment_clean_prob: float = 1.0,
        text_agg_tokens: int = 0,
        diagnostics: bool = False,
        state_encoder: str = "patch_merger",
        **kwargs,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_channels <= 0 or sequence_length <= 0 or num_experts <= 0:
            raise ValueError("d_model, num_channels, sequence_length, and num_experts must be positive")
        self.d_model = int(d_model)
        self.num_channels = int(num_channels)
        self.sequence_length = int(sequence_length)
        self.num_experts = int(num_experts)
        self.temporal_merge = max(1, int(temporal_merge))
        self.channel_merge = max(1, int(channel_merge))
        self.token_budget = int(token_budget)
        self.alignment_mode = str(alignment_mode).lower()
        if self.alignment_mode not in {"siglip", "infonce"}:
            raise ValueError(f"alignment_mode must be 'siglip' or 'infonce', got {alignment_mode!r}")
        self.alignment_temperature = float(alignment_temperature)
        # Dense (token/region-level) alignment instead of pooled-vs-pooled. When
        # enabled, both text and series sides are pooled to a small set of region
        # tokens and matched with ColBERT-style MaxSim, which constrains
        # token<->timestep correspondence rather than a single global direction.
        self.alignment_dense = bool(alignment_dense)
        self.alignment_regions = max(1, int(alignment_regions))
        # Which series representation to align against: "clean" encodes the
        # ground-truth target through the same patch-merger stack (signal is not
        # diluted by flow noise); "state" reuses the fused current-state tokens.
        # "mixed" samples between them during training, matching the scheduled-
        # sampling idea of exposing the bridge loss to inference-like states.
        self.alignment_target = str(alignment_target).lower()
        if self.alignment_target not in {"state", "clean", "mixed"}:
            raise ValueError(f"alignment_target must be 'state', 'clean', or 'mixed', got {alignment_target!r}")
        self.alignment_clean_prob = float(alignment_clean_prob)
        if not 0.0 <= self.alignment_clean_prob <= 1.0:
            raise ValueError(f"alignment_clean_prob must be in [0, 1], got {alignment_clean_prob!r}")
        # Optional learnable text aggregation tokens prepended to the slot
        # sequence before the bidirectional cross-attn so the bridge can read a
        # few sequence-level summaries; they are stripped before any downstream
        # consumer. Default 0 builds no parameter and is a numerical no-op.
        self.text_agg_token_count = max(0, int(text_agg_tokens))
        self.diagnostics = bool(diagnostics)

        # State encoder type: "patch_merger" (default) or "temporal_pyramid_v2"
        self.state_encoder_type = str(state_encoder).lower()
        if self.state_encoder_type not in {"patch_merger", "temporal_pyramid_v2"}:
            raise ValueError(f"state_encoder must be 'patch_merger' or 'temporal_pyramid_v2', got {state_encoder!r}")

        heads = max(1, min(int(num_heads), self.d_model))
        while self.d_model % heads != 0 and heads > 1:
            heads -= 1
        self.num_heads = heads

        # TSP-Bridge V2 state encoder (optional)
        self.tsp_encoder = None
        if self.state_encoder_type == "temporal_pyramid_v2":
            from effectcma_flow.models.temporal_pyramid_bridge_v2 import TemporalSemanticPyramidConnectorV2

            # Extract TSP-specific parameters from kwargs
            tsp_patch_lens = kwargs.get("temporal_pyramid_patch_lens", (4, 8, 16, 32))
            tsp_token_budget = kwargs.get("temporal_pyramid_token_budget", 128)
            tsp_anchor_tokens = kwargs.get("temporal_pyramid_anchor_tokens", 8)
            tsp_cross_scale_layers = kwargs.get("temporal_pyramid_cross_scale_layers", 2)
            tsp_dropout = kwargs.get("temporal_pyramid_dropout", dropout)
            tsp_use_topdown = kwargs.get("temporal_pyramid_use_topdown", True)
            tsp_use_bottomup = kwargs.get("temporal_pyramid_use_bottomup", True)
            tsp_use_text_routing = kwargs.get("temporal_pyramid_use_text_routing", True)
            tsp_temporal_bias_tau = kwargs.get("temporal_pyramid_temporal_bias_tau", 0.25)
            tsp_gate_temperature = kwargs.get("temporal_pyramid_gate_temperature", 0.7)

            self.tsp_encoder = TemporalSemanticPyramidConnectorV2(
                sequence_length=sequence_length,
                num_channels=num_channels,
                d_model=d_model,
                patch_lens=tsp_patch_lens,
                token_budget=tsp_token_budget,
                anchor_tokens=tsp_anchor_tokens,
                cross_scale_layers=tsp_cross_scale_layers,
                num_heads=heads,
                dropout=tsp_dropout,
                use_topdown=tsp_use_topdown,
                use_bottomup=tsp_use_bottomup,
                use_text_routing=tsp_use_text_routing,
                temporal_bias_tau=tsp_temporal_bias_tau,
                gate_temperature=tsp_gate_temperature,
            )
            # TSP encoder handles its own budget pooling
            self.budget_pool = nn.Identity()
        else:
            # Standard patch merger encoder
            self.value_proj = nn.Linear(3, self.d_model)
            self.channel_embed = nn.Embedding(self.num_channels, self.d_model)
            self.latent_blocks = nn.Sequential(
                LatentGridBlock(self.d_model, self.num_channels, dropout),
                LatentGridBlock(self.d_model, self.num_channels, dropout),
            )
            merged_dim = self.d_model * self.temporal_merge * self.channel_merge
            self.patch_merger_norm = nn.LayerNorm(self.d_model)
            self.patch_merger_mlp = nn.Sequential(
                nn.Linear(merged_dim, merged_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(merged_dim, self.d_model),
            )
            self.merged_norm = nn.LayerNorm(self.d_model)
            self.budget_pool = TokenBudgetPool(self.d_model, self.token_budget, heads, dropout)

        self.text_norm = nn.LayerNorm(self.d_model)
        self.state_norm = nn.LayerNorm(self.d_model)
        self.text_to_state = nn.MultiheadAttention(self.d_model, heads, dropout=dropout, batch_first=True)
        self.state_to_text = nn.MultiheadAttention(self.d_model, heads, dropout=dropout, batch_first=True)
        self.text_gate = nn.Sequential(nn.LayerNorm(2 * self.d_model), nn.Linear(2 * self.d_model, self.d_model), nn.Sigmoid())
        self.state_gate = nn.Sequential(nn.LayerNorm(2 * self.d_model), nn.Linear(2 * self.d_model, self.d_model), nn.Sigmoid())
        self.text_ffn = _ffn(self.d_model, dropout)
        self.state_ffn = _ffn(self.d_model, dropout)

        self.bridge_pool = AttentionPool(self.d_model, heads, dropout)
        self.text_context_pool = AttentionPool(self.d_model, heads, dropout)
        self.context_gate = nn.Sequential(
            nn.LayerNorm(2 * self.d_model),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.Sigmoid(),
        )
        self.context_norm = nn.LayerNorm(self.d_model)
        self.text_align_pool = AttentionPool(self.d_model, heads, dropout)
        self.state_align_pool = AttentionPool(self.d_model, heads, dropout)
        self.expert_queries = nn.Parameter(torch.randn(self.num_experts, self.d_model) * 0.02)
        self.channel_queries = nn.Parameter(torch.randn(self.num_channels, self.d_model) * 0.02)
        self.expert_pool = CrossAttentionPool(self.d_model, heads, dropout)
        self.channel_pool = CrossAttentionPool(self.d_model, heads, dropout)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))
        self.text_align = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))
        self.state_align = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))
        if self.alignment_dense:
            # Region pools: learned multi-query attention pooling each modality to
            # `alignment_regions` tokens, keeping MaxSim a fixed [B,B,R,R] cost
            # (full-batch friendly) while staying token-level rather than pooled.
            self.text_region_pool = MultiQueryAttentionPool(self.d_model, self.alignment_regions, heads, dropout)
            self.state_region_pool = MultiQueryAttentionPool(self.d_model, self.alignment_regions, heads, dropout)

        # ZERO init so the agg tokens start as the additive identity inside the
        # cross-attn residual; randn agg tokens stall in cold-start. Only created
        # when K>0, so the default (K=0) registers no parameter and keeps the
        # state_dict and parameter count identical to before.
        if self.text_agg_token_count > 0:
            self.text_agg_tokens = nn.Parameter(torch.zeros(self.text_agg_token_count, self.d_model))

    def forward(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        slot_tokens: torch.Tensor,
        slot_mask: torch.Tensor,
        compute_alignment: bool = True,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_inputs(x_t=x_t, t=t, slot_tokens=slot_tokens, slot_mask=slot_mask)
        dtype = slot_tokens.dtype
        device = slot_tokens.device
        x_t = x_t.to(device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype).view(-1)
        slot_mask = slot_mask.to(device=device, dtype=dtype)
        if target is not None:
            target = target.to(device=device, dtype=dtype)

        # Generate state tokens using the configured encoder
        tsp_aux: dict[str, torch.Tensor] = {}
        if self.tsp_encoder is not None:
            # TSP-Bridge V2: multi-scale temporal pyramid encoder
            text_context = _masked_mean(slot_tokens, slot_mask) if slot_mask is not None else slot_tokens.mean(dim=1)
            state_tokens, tsp_aux = self.tsp_encoder(
                x_t=x_t,
                t=t,
                slot_tokens=slot_tokens,
                slot_mask=slot_mask,
                text_context=text_context,
            )
            prebudget_count = tsp_aux.get("tsp_raw_token_count", torch.tensor(0.0)).item()
        else:
            # Standard patch merger encoder
            latent_grid = self._encode_grid(x_t, t)
            merged_tokens = self._patch_merge(latent_grid)
            prebudget_count = merged_tokens.shape[1]
            state_tokens = self.budget_pool(merged_tokens)

        # Optionally prepend K learnable text aggregation tokens to the slot
        # sequence ONLY for the bidirectional cross-attn, then strip them before
        # any downstream consumer. Their mask entries are all-valid so they are
        # never padding-masked by state_to_text. K=0 reuses the original tensors,
        # so the forward is bit-for-bit identical to before.
        k_agg = self.text_agg_token_count
        if k_agg > 0:
            agg = self.text_agg_tokens[None].expand(slot_tokens.shape[0], -1, -1).to(device=device, dtype=dtype)
            bridge_slot_tokens = torch.cat([agg, slot_tokens], dim=1)
            agg_mask = slot_mask.new_ones(slot_mask.shape[0], k_agg)
            bridge_slot_mask = torch.cat([agg_mask, slot_mask], dim=1)
        else:
            bridge_slot_tokens = slot_tokens
            bridge_slot_mask = slot_mask

        text_query = self.text_norm(bridge_slot_tokens)
        state_query = self.state_norm(state_tokens)
        need_attn_weights = self.diagnostics
        text_delta, text_attn = self.text_to_state(
            query=text_query,
            key=state_query,
            value=state_tokens,
            need_weights=need_attn_weights,
            average_attn_weights=False,
        )
        text_gate = self.text_gate(torch.cat([bridge_slot_tokens, text_delta], dim=-1))
        bridged_text = bridge_slot_tokens + text_gate * text_delta
        bridged_text = bridged_text + self.text_ffn(bridged_text)
        if k_agg > 0:
            # Drop the K agg tokens to restore the original slot dim [B, J, D]
            # before masking and every downstream consumer.
            bridged_text = bridged_text[:, k_agg:]
        bridged_text = bridged_text * slot_mask[:, :, None].clamp(0.0, 1.0)

        text_key_padding_mask = _safe_key_padding_mask(bridge_slot_mask)
        state_delta, state_attn = self.state_to_text(
            query=state_query,
            key=self.text_norm(bridge_slot_tokens),
            value=bridge_slot_tokens,
            key_padding_mask=text_key_padding_mask,
            need_weights=need_attn_weights,
            average_attn_weights=False,
        )
        state_gate = self.state_gate(torch.cat([state_tokens, state_delta], dim=-1))
        fused_state = state_tokens + state_gate * state_delta
        fused_state = fused_state + self.state_ffn(fused_state)

        text_context = self.text_context_pool(bridged_text, mask=slot_mask)
        state_context = self.bridge_pool(fused_state)
        context_gate = self.context_gate(torch.cat([text_context, state_context], dim=-1))
        bridge_context = self.context_norm(text_context + context_gate * state_context)
        expert_query = self.expert_queries[None].expand(x_t.shape[0], -1, -1).to(device=device, dtype=dtype)
        channel_query = self.channel_queries[None].expand(x_t.shape[0], -1, -1).to(device=device, dtype=dtype)
        expert_context = self.expert_pool(expert_query, fused_state)
        channel_context = self.channel_pool(channel_query, fused_state)

        use_clean = False
        if compute_alignment:
            # Choose the series tokens to align against. "clean" re-encodes the
            # ground-truth target through the same patch-merger stack so the
            # alignment target is not diluted by flow noise; falls back to the
            # fused current state when no target is supplied (e.g. at sampling).
            use_clean = self._use_clean_alignment_target(target)
            if use_clean and self.tsp_encoder is not None:
                # TSP encoder does not have _encode_grid/_patch_merge; fall back
                # to fused_state for alignment target
                align_state_tokens = fused_state
            elif use_clean:
                clean_grid = self._encode_grid(target, torch.ones_like(t))
                align_state_tokens = self.budget_pool(self._patch_merge(clean_grid))
            else:
                align_state_tokens = fused_state
            if self.alignment_dense:
                alignment_loss, alignment_aux = self._dense_alignment_loss(
                    bridged_text, align_state_tokens, slot_mask
                )
            else:
                text_repr = self.text_align_pool(bridged_text, mask=slot_mask)
                state_repr = self.state_align_pool(align_state_tokens)
                alignment_loss, alignment_aux = self._alignment_loss(text_repr, state_repr)
        else:
            alignment_loss, alignment_aux = _zero_alignment_aux(fused_state)
        valid_slot_count = slot_mask.sum(dim=1)
        if need_attn_weights:
            # text_attn/state_attn span the K+J bridge sequence, so summarise
            # them with the matching extended mask (== slot_mask when K=0).
            text_entropy, text_max, text_entropy_norm = _attention_summary(text_attn, query_mask=bridge_slot_mask)
            state_entropy, state_max, state_entropy_norm = _attention_summary(state_attn, key_mask=bridge_slot_mask)
        else:
            text_entropy, text_max, text_entropy_norm = _nan_summary(fused_state)
            state_entropy, state_max, state_entropy_norm = _nan_summary(fused_state)
        aux = {
            "bridge_alignment_loss": alignment_loss,
            "text_slot_count": valid_slot_count.detach().mean(),
            "text_slot_count_min": valid_slot_count.detach().min(),
            "text_slot_count_max": valid_slot_count.detach().max(),
            "bridge_text_to_state_entropy": text_entropy.detach(),
            "bridge_text_to_state_entropy_norm": text_entropy_norm.detach(),
            "bridge_text_to_state_max_prob": text_max.detach(),
            "bridge_state_to_text_entropy": state_entropy.detach(),
            "bridge_state_to_text_entropy_norm": state_entropy_norm.detach(),
            "bridge_state_to_text_max_prob": state_max.detach(),
            "bridge_context_norm": bridge_context.detach().norm(dim=-1).mean(),
            "bridge_text_context_norm": text_context.detach().norm(dim=-1).mean(),
            "bridge_state_context_norm": state_context.detach().norm(dim=-1).mean(),
            "bridge_expert_context_norm": expert_context.detach().norm(dim=-1).mean(),
            "bridge_channel_context_norm": channel_context.detach().norm(dim=-1).mean(),
            "bridge_memory_token_count": torch.tensor(float(fused_state.shape[1]), device=device, dtype=dtype),
            "bridge_patch_token_count": torch.tensor(float(prebudget_count), device=device, dtype=dtype),
            # 1.0 iff merged tokens exceeded the budget so the Perceiver pool
            # actually ran; 0.0 means the budget pool was an identity pass-through
            # (i.e. the configured token_budget is too large to ever fire).
            "bridge_budget_pool_active": torch.tensor(
                1.0 if prebudget_count > self.token_budget else 0.0, device=device, dtype=dtype
            ),
            "bridge_token_budget": torch.tensor(float(self.token_budget), device=device, dtype=dtype),
            "bridge_connector_patch_merger": torch.tensor(1.0, device=device, dtype=dtype),
            "bridge_alignment_used_clean": torch.tensor(1.0 if compute_alignment and use_clean else 0.0, device=device, dtype=dtype),
            **alignment_aux,
        }
        # Add TSP-Bridge V2 diagnostics if available
        if tsp_aux:
            aux.update(tsp_aux)
        return bridged_text, bridge_context, expert_context, channel_context, aux

    def _use_clean_alignment_target(self, target: torch.Tensor | None) -> bool:
        if target is None:
            return False
        if self.alignment_target == "clean":
            return True
        if self.alignment_target == "state":
            return False
        if self.alignment_clean_prob <= 0.0:
            return False
        if self.alignment_clean_prob >= 1.0:
            return True
        if not self.training:
            return self.alignment_clean_prob >= 0.5
        draw = torch.rand((), device=target.device)
        return bool(draw.item() < self.alignment_clean_prob)

    def _encode_grid(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x_t.shape
        dtype = x_t.dtype
        device = x_t.device
        pos = torch.linspace(-1.0, 1.0, length, device=device, dtype=dtype)
        pos = pos[None, :, None, None].expand(batch, length, channels, 1)
        flow_time = t[:, None, None, None].expand(batch, length, channels, 1)
        features = torch.cat([x_t[..., None], pos, flow_time], dim=-1)
        h = self.value_proj(features)
        channel_ids = torch.arange(channels, device=device)
        h = h + self.channel_embed(channel_ids)[None, None, :, :].to(dtype=dtype)
        return self.latent_blocks(h)

    def _patch_merge(self, h: torch.Tensor) -> torch.Tensor:
        batch, length, channels, d_model = h.shape
        pad_l = (self.temporal_merge - length % self.temporal_merge) % self.temporal_merge
        pad_c = (self.channel_merge - channels % self.channel_merge) % self.channel_merge
        if pad_l or pad_c:
            h = F.pad(h, (0, 0, 0, pad_c, 0, pad_l))
        length_pad, channels_pad = h.shape[1], h.shape[2]
        h = h.reshape(
            batch,
            length_pad // self.temporal_merge,
            self.temporal_merge,
            channels_pad // self.channel_merge,
            self.channel_merge,
            d_model,
        )
        h = h.permute(0, 1, 3, 2, 4, 5).contiguous()
        h = h.reshape(batch, -1, self.temporal_merge * self.channel_merge, d_model)
        h = self.patch_merger_norm(h)
        h = h.reshape(batch, h.shape[1], self.temporal_merge * self.channel_merge * d_model)
        return self.merged_norm(self.patch_merger_mlp(h))

    def _alignment_loss(self, text_repr: torch.Tensor, state_repr: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        text = F.normalize(self.text_align(text_repr), dim=-1)
        state = F.normalize(self.state_align(state_repr), dim=-1)
        logits = text @ state.transpose(0, 1)
        if self.alignment_mode == "siglip":
            scale = self.logit_scale.exp().clamp(max=100.0).to(dtype=logits.dtype)
            logits = logits * scale + self.logit_bias.to(dtype=logits.dtype)
            loss = _balanced_sigmoid_contrastive_loss(logits)
        else:
            logits = logits / max(self.alignment_temperature, 1e-4)
            labels_idx = torch.arange(logits.shape[0], device=logits.device)
            loss = 0.5 * (F.cross_entropy(logits, labels_idx) + F.cross_entropy(logits.transpose(0, 1), labels_idx))
        pos = logits.diag()
        return loss, {
            "bridge_alignment_logit_pos": pos.detach().mean(),
            "bridge_alignment_logit_std": logits.detach().std(unbiased=False),
        }

    def _dense_alignment_loss(
        self,
        bridged_text: torch.Tensor,
        state_tokens: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """ColBERT-style MaxSim alignment between text and series region tokens.

        Both modalities are pooled to ``alignment_regions`` region tokens, then a
        per-pair score is the mean over text regions of the max similarity to any
        series region (MaxSim). This produces a ``[B, B]`` score matrix fed to the
        same SigLIP / InfoNCE contrastive loss, but unlike pooled-vs-pooled it
        constrains fine-grained region<->region correspondence.
        """
        text_regions = self.text_region_pool(bridged_text, mask=slot_mask)  # [B,R,D]
        state_regions = self.state_region_pool(state_tokens)  # [B,R,D]
        text = F.normalize(self.text_align(text_regions), dim=-1)
        state = F.normalize(self.state_align(state_regions), dim=-1)
        # sim[a,b,i,j] = <text_a_region_i, state_b_region_j>
        sim = torch.einsum("aid,bjd->abij", text, state)
        # MaxSim: each text region picks its best-matching series region, average
        # over text regions -> [B(text), B(series)] pairwise score.
        maxsim = sim.max(dim=-1).values.mean(dim=-1)
        if self.alignment_mode == "siglip":
            scale = self.logit_scale.exp().clamp(max=100.0).to(dtype=maxsim.dtype)
            logits = maxsim * scale + self.logit_bias.to(dtype=maxsim.dtype)
            loss = _balanced_sigmoid_contrastive_loss(logits)
        else:
            logits = maxsim / max(self.alignment_temperature, 1e-4)
            labels_idx = torch.arange(logits.shape[0], device=logits.device)
            loss = 0.5 * (F.cross_entropy(logits, labels_idx) + F.cross_entropy(logits.transpose(0, 1), labels_idx))
        pos = logits.diag()
        return loss, {
            "bridge_alignment_logit_pos": pos.detach().mean(),
            "bridge_alignment_logit_std": logits.detach().std(unbiased=False),
            "bridge_alignment_dense": logits.new_tensor(1.0),
        }

    def _validate_inputs(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        slot_tokens: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> None:
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be [B,L,C], got {tuple(x_t.shape)}")
        if slot_tokens.ndim != 3:
            raise ValueError(f"slot_tokens must be [B,J,D], got {tuple(slot_tokens.shape)}")
        batch, length, channels = x_t.shape
        if (length, channels) != (self.sequence_length, self.num_channels):
            raise ValueError(f"Expected x_t shape [B,{self.sequence_length},{self.num_channels}], got {tuple(x_t.shape)}")
        if t.shape != (batch,):
            raise ValueError(f"t must have shape [B], got {tuple(t.shape)}")
        if slot_tokens.shape[0] != batch or slot_tokens.shape[-1] != self.d_model:
            raise ValueError("slot_tokens batch/dim must match x_t and d_model")
        if slot_mask.shape != slot_tokens.shape[:2]:
            raise ValueError(f"slot_mask must have shape {tuple(slot_tokens.shape[:2])}, got {tuple(slot_mask.shape)}")


class CrossModalConditionBridge(nn.Module):
    """Bidirectional text-state bridge for text-to-series generation.

    Text tokens interact with state tokens built from the current flow state.
    The bridge keeps token-level alignment inside the velocity field instead of
    reducing text to a single vector before the generator sees the state.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_channels: int,
        sequence_length: int,
        num_experts: int,
        patch_size: int = 6,
        num_heads: int = 4,
        dropout: float = 0.0,
        num_spectral_tokens: int = 3,
        alignment_temperature: float = 0.07,
        focal_mode: str = "legacy",
        num_stage_tokens: int = 3,
        state_connector: str = "legacy",
        temporal_merge: int = 4,
        channel_merge: int = 1,
        token_budget: int = 96,
        alignment_mode: str = "auto",
        alignment_dense: bool = False,
        alignment_regions: int = 8,
        alignment_target: str = "state",
        alignment_clean_prob: float = 1.0,
        text_agg_tokens: int = 0,
        diagnostics: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_channels <= 0 or sequence_length <= 0 or num_experts <= 0:
            raise ValueError("d_model, num_channels, sequence_length, and num_experts must be positive")
        if patch_size <= 0 or num_spectral_tokens <= 0:
            raise ValueError("patch_size and num_spectral_tokens must be positive")
        heads = max(1, min(int(num_heads), int(d_model)))
        while d_model % heads != 0 and heads > 1:
            heads -= 1

        self.d_model = int(d_model)
        self.num_channels = int(num_channels)
        self.sequence_length = int(sequence_length)
        self.num_experts = int(num_experts)
        self.patch_size = int(patch_size)
        self.num_spectral_tokens = int(num_spectral_tokens)
        self.alignment_temperature = float(alignment_temperature)
        self.state_connector = str(state_connector).lower()
        if self.state_connector in {"ts_patch_merger", "patchmerger"}:
            self.state_connector = "patch_merger"
        if self.state_connector not in {"legacy", "patch_merger", "temporal_pyramid_v2"}:
            raise ValueError(f"state_connector must be 'legacy', 'patch_merger', or 'temporal_pyramid_v2', got {state_connector!r}")
        self.alignment_mode = str(alignment_mode).lower()
        if self.alignment_mode not in {"auto", "infonce", "siglip"}:
            raise ValueError(f"alignment_mode must be 'auto', 'infonce', or 'siglip', got {alignment_mode!r}")
        self.focal_mode = str(focal_mode).lower()
        if self.focal_mode not in {"legacy", "latent_query"}:
            raise ValueError(f"focal_mode must be 'legacy' or 'latent_query', got {focal_mode!r}")
        self.num_stage_tokens = int(num_stage_tokens)
        if self.num_stage_tokens <= 0:
            raise ValueError("num_stage_tokens must be positive")

        # State connector: patch_merger, temporal_pyramid_v2, or legacy
        self.patch_merger_connector = None

        if self.state_connector in {"patch_merger", "temporal_pyramid_v2"}:
            # Determine state_encoder type for TSPatchMergerConnector
            state_encoder_type = "temporal_pyramid_v2" if self.state_connector == "temporal_pyramid_v2" else "patch_merger"

            self.patch_merger_connector = TSPatchMergerConnector(
                d_model=self.d_model,
                num_channels=self.num_channels,
                sequence_length=self.sequence_length,
                num_experts=self.num_experts,
                temporal_merge=temporal_merge,
                channel_merge=channel_merge,
                token_budget=token_budget,
                num_heads=heads,
                dropout=dropout,
                alignment_mode="siglip" if self.alignment_mode == "auto" else self.alignment_mode,
                alignment_temperature=self.alignment_temperature,
                alignment_dense=alignment_dense,
                alignment_regions=alignment_regions,
                alignment_target=alignment_target,
                alignment_clean_prob=alignment_clean_prob,
                text_agg_tokens=text_agg_tokens,
                diagnostics=diagnostics,
                state_encoder=state_encoder_type,
                **kwargs,
            )

        # Legacy state-token bridge body. When the patch-merger connector owns
        # the forward pass these modules are never executed, so we skip building
        # them entirely instead of leaving inert "built but bypassed" parameters.
        self._has_legacy_body = self.state_connector not in {"patch_merger", "temporal_pyramid_v2"}
        if self._has_legacy_body:
            self.time_patch_proj = nn.Linear(self.patch_size * self.num_channels, self.d_model)
            self.channel_stat_proj = nn.Linear(3, self.d_model)
            self.spectral_proj = nn.Linear(1, self.d_model)
            self.channel_embed = nn.Embedding(self.num_channels, self.d_model)
            self.spectral_embed = nn.Embedding(self.num_spectral_tokens, self.d_model)
            # 0=time patch, 1=channel summary, 2=spectral band.
            self.type_embed = nn.Embedding(3, self.d_model)
            self.time_embed = nn.Sequential(
                nn.Linear(1, self.d_model),
                nn.SiLU(),
                nn.Linear(self.d_model, self.d_model),
            )

            self.text_norm = nn.LayerNorm(self.d_model)
            self.state_norm = nn.LayerNorm(self.d_model)
            self.text_to_state = nn.MultiheadAttention(self.d_model, heads, dropout=dropout, batch_first=True)
            self.state_to_text = nn.MultiheadAttention(self.d_model, heads, dropout=dropout, batch_first=True)
            self.text_ffn = _ffn(self.d_model, dropout)
            self.state_ffn = _ffn(self.d_model, dropout)
            self.dropout = nn.Dropout(dropout)

            self.state_to_context = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))
            self.context_gate = nn.Sequential(
                nn.LayerNorm(2 * self.d_model),
                nn.Linear(2 * self.d_model, self.d_model),
                nn.Sigmoid(),
            )
            self.context_norm = nn.LayerNorm(self.d_model)
            self.expert_context = nn.Sequential(
                nn.LayerNorm(2 * self.d_model),
                nn.Linear(2 * self.d_model, self.num_experts * self.d_model),
            )
            self.text_align = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))
            self.state_align = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))
            if self.focal_mode == "latent_query":
                init = 0.02
                self.channel_queries = nn.Parameter(torch.randn(self.num_channels, self.d_model) * init)
                self.expert_queries = nn.Parameter(torch.randn(self.num_experts, self.d_model) * init)
                self.scale_queries = nn.Parameter(torch.randn(self.num_spectral_tokens, self.d_model) * init)
                self.stage_queries = nn.Parameter(torch.randn(self.num_stage_tokens, self.d_model) * init)
                self.stage_selector = nn.Sequential(
                    nn.Linear(1, self.d_model),
                    nn.SiLU(),
                    nn.Linear(self.d_model, self.num_stage_tokens),
                )
                self.focal_memory_norm = nn.LayerNorm(self.d_model)
                self.focal_query_norm = nn.LayerNorm(self.d_model)
                self.focal_attn = TimestepAwareCrossAttention(self.d_model, heads, dropout)
                self.channel_context_norm = nn.LayerNorm(self.d_model)
                self.expert_context_norm = nn.LayerNorm(self.d_model)
                self.scale_context_norm = nn.LayerNorm(self.d_model)
                self.stage_context_norm = nn.LayerNorm(self.d_model)

    def forward(
        self,
        *,
        slot_tokens: torch.Tensor,
        slot_mask: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        compute_alignment: bool = True,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if slot_tokens.ndim != 3:
            raise ValueError(f"slot_tokens must be [B,J,D], got {tuple(slot_tokens.shape)}")
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be [B,L,C], got {tuple(x_t.shape)}")
        batch, length, channels = x_t.shape
        if slot_tokens.shape[0] != batch:
            raise ValueError("slot_tokens and x_t must have the same batch size")
        if slot_tokens.shape[-1] != self.d_model:
            raise ValueError(f"Expected slot token dim {self.d_model}, got {slot_tokens.shape[-1]}")
        if (length, channels) != (self.sequence_length, self.num_channels):
            raise ValueError(f"Expected x_t shape [B,{self.sequence_length},{self.num_channels}], got {tuple(x_t.shape)}")
        if t.shape != (batch,):
            raise ValueError(f"t must have shape [B], got {tuple(t.shape)}")
        if slot_mask.shape != slot_tokens.shape[:2]:
            raise ValueError(f"slot_mask must have shape {tuple(slot_tokens.shape[:2])}, got {tuple(slot_mask.shape)}")

        dtype = slot_tokens.dtype
        device = slot_tokens.device
        x_t = x_t.to(device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype)
        slot_mask = slot_mask.to(device=device, dtype=dtype)
        if target is not None:
            target = target.to(device=device, dtype=dtype)
        if self.patch_merger_connector is not None:
            return self.patch_merger_connector(
                x_t=x_t,
                t=t,
                slot_tokens=slot_tokens,
                slot_mask=slot_mask,
                compute_alignment=compute_alignment,
                target=target,
            )
        if not self._has_legacy_body:
            raise RuntimeError("legacy bridge body was not built; expected patch_merger_connector to handle forward")
        state_tokens = self._build_state_tokens(x_t, t)

        text_query = self.text_norm(slot_tokens)
        state_query = self.state_norm(state_tokens)
        text_delta, text_attn = self.text_to_state(
            query=text_query,
            key=state_query,
            value=state_query,
            need_weights=True,
            average_attn_weights=False,
        )
        bridged_text = slot_tokens + self.dropout(text_delta)
        bridged_text = bridged_text + self.text_ffn(bridged_text)
        bridged_text = bridged_text * slot_mask[:, :, None].clamp(0.0, 1.0)

        text_key_padding_mask = _safe_key_padding_mask(slot_mask)
        state_delta, state_attn = self.state_to_text(
            query=state_query,
            key=self.text_norm(slot_tokens),
            value=slot_tokens,
            key_padding_mask=text_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        bridged_state = state_tokens + self.dropout(state_delta)
        bridged_state = bridged_state + self.state_ffn(bridged_state)

        text_context = _masked_mean(bridged_text, slot_mask)
        time_token_count = _num_patch_tokens(length, self.patch_size)

        state_context = self.state_to_context(bridged_state.mean(dim=1))
        focal_aux: dict[str, torch.Tensor] = {}
        if self.focal_mode == "latent_query":
            bridge_context, expert_context, channel_context, focal_aux = self._focal_contexts(
                bridged_text=bridged_text,
                bridged_state=bridged_state,
                slot_mask=slot_mask,
                text_context=text_context,
                state_context=state_context,
                t=t,
            )
        else:
            channel_context = bridged_state[:, time_token_count : time_token_count + channels]
            if channel_context.shape != (batch, channels, self.d_model):
                raise RuntimeError(
                    f"internal channel_context shape {tuple(channel_context.shape)} "
                    f"must be {(batch, channels, self.d_model)}"
                )
            gate = self.context_gate(torch.cat([text_context, state_context], dim=-1))
            bridge_context = self.context_norm(text_context + gate * state_context)
            expert_context = self.expert_context(torch.cat([bridge_context, state_context], dim=-1))
            expert_context = expert_context.view(batch, self.num_experts, self.d_model)

        if compute_alignment:
            alignment_loss, alignment_aux = self._alignment_loss(text_context, state_context)
        else:
            alignment_loss, alignment_aux = _zero_alignment_aux(state_context)
        valid_slot_count = slot_mask.sum(dim=1)
        text_entropy, text_max, text_entropy_norm = _attention_summary(text_attn, query_mask=slot_mask)
        state_entropy, state_max, state_entropy_norm = _attention_summary(state_attn, key_mask=slot_mask)
        aux = {
            "bridge_alignment_loss": alignment_loss,
            "text_slot_count": valid_slot_count.detach().mean(),
            "text_slot_count_min": valid_slot_count.detach().min(),
            "text_slot_count_max": valid_slot_count.detach().max(),
            "bridge_text_to_state_entropy": text_entropy.detach(),
            "bridge_text_to_state_entropy_norm": text_entropy_norm.detach(),
            "bridge_text_to_state_max_prob": text_max.detach(),
            "bridge_state_to_text_entropy": state_entropy.detach(),
            "bridge_state_to_text_entropy_norm": state_entropy_norm.detach(),
            "bridge_state_to_text_max_prob": state_max.detach(),
            "bridge_context_norm": bridge_context.detach().norm(dim=-1).mean(),
            "bridge_text_context_norm": text_context.detach().norm(dim=-1).mean(),
            "bridge_state_context_norm": state_context.detach().norm(dim=-1).mean(),
            "bridge_expert_context_norm": expert_context.detach().norm(dim=-1).mean(),
            "bridge_channel_context_norm": channel_context.detach().norm(dim=-1).mean(),
            **focal_aux,
            **alignment_aux,
        }
        return bridged_text, bridge_context, expert_context, channel_context, aux

    def _focal_contexts(
        self,
        *,
        bridged_text: torch.Tensor,
        bridged_state: torch.Tensor,
        slot_mask: torch.Tensor,
        text_context: torch.Tensor,
        state_context: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch = bridged_text.shape[0]
        time_code = self.time_embed(t[:, None])[:, None, :]
        state_mask = torch.ones(bridged_state.shape[:2], device=slot_mask.device, dtype=slot_mask.dtype)
        memory_mask = torch.cat([slot_mask, state_mask], dim=1)
        memory = self.focal_memory_norm(torch.cat([bridged_text, bridged_state], dim=1))

        channel_query = self.channel_queries[None].expand(batch, -1, -1)
        channel_query = channel_query + self.channel_embed.weight[None].to(dtype=bridged_text.dtype) + time_code
        expert_query = self.expert_queries[None].expand(batch, -1, -1) + time_code
        scale_query = self.scale_queries[None].expand(batch, -1, -1)
        scale_query = scale_query + self.spectral_embed.weight[None].to(dtype=bridged_text.dtype) + time_code
        stage_weight = torch.softmax(self.stage_selector(t[:, None].to(bridged_text.dtype)), dim=-1)
        stage_query = torch.matmul(stage_weight[:, None, :], self.stage_queries[None].expand(batch, -1, -1)) + time_code

        channel_raw, channel_attn = self.focal_attn(self.focal_query_norm(channel_query), memory, memory_mask, t)
        expert_raw, expert_attn = self.focal_attn(self.focal_query_norm(expert_query), memory, memory_mask, t)
        scale_raw, scale_attn = self.focal_attn(self.focal_query_norm(scale_query), memory, memory_mask, t)
        stage_raw, stage_attn = self.focal_attn(self.focal_query_norm(stage_query), memory, memory_mask, t)

        channel_context = self.channel_context_norm(channel_query + channel_raw)
        scale_context = self.scale_context_norm(scale_query + scale_raw)
        stage_context = self.stage_context_norm(stage_query + stage_raw).squeeze(1)
        scale_summary = scale_context.mean(dim=1)
        expert_context = self.expert_context_norm(expert_query + expert_raw + stage_context[:, None, :] + scale_summary[:, None, :])
        gate = self.context_gate(torch.cat([text_context, state_context], dim=-1))
        bridge_context = self.context_norm(text_context + gate * state_context + 0.25 * stage_context + 0.25 * scale_summary)

        _, _, channel_h_norm = _attention_summary(channel_attn, key_mask=memory_mask)
        _, _, expert_h_norm = _attention_summary(expert_attn, key_mask=memory_mask)
        _, _, scale_h_norm = _attention_summary(scale_attn, key_mask=memory_mask)
        _, _, stage_h_norm = _attention_summary(stage_attn, key_mask=memory_mask)
        aux = {
            "bridge_focal_channel_entropy_norm": channel_h_norm.detach(),
            "bridge_focal_expert_entropy_norm": expert_h_norm.detach(),
            "bridge_focal_scale_entropy_norm": scale_h_norm.detach(),
            "bridge_focal_stage_entropy_norm": stage_h_norm.detach(),
            "bridge_scale_context_norm": scale_context.detach().norm(dim=-1).mean(),
            "bridge_stage_context_norm": stage_context.detach().norm(dim=-1).mean(),
            "bridge_memory_token_count": memory_mask.detach().sum(dim=1).mean(),
        }
        return bridge_context, expert_context, channel_context, aux

    def _build_state_tokens(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x_t.shape
        dtype = x_t.dtype
        time_code = self.time_embed(t[:, None])[:, None, :]

        patch_size = min(self.patch_size, length)
        pad = (patch_size - length % patch_size) % patch_size
        if pad:
            x_patch = F.pad(x_t, (0, 0, 0, pad))
        else:
            x_patch = x_t
        num_patches = x_patch.shape[1] // patch_size
        patches = x_patch.reshape(batch, num_patches, patch_size * channels)
        if patch_size != self.patch_size:
            patches = F.pad(patches, (0, (self.patch_size - patch_size) * channels))
        time_tokens = self.time_patch_proj(patches)
        time_tokens = time_tokens + self.type_embed.weight[0].to(dtype=dtype)[None, None, :] + time_code

        mean = x_t.mean(dim=1)
        std = x_t.std(dim=1, unbiased=False)
        if length > 1:
            roughness = (x_t[:, 1:] - x_t[:, :-1]).square().mean(dim=1).sqrt()
        else:
            roughness = torch.zeros_like(std)
        channel_stats = torch.stack([mean, std, roughness], dim=-1)
        channel_tokens = self.channel_stat_proj(channel_stats)
        channel_ids = torch.arange(channels, device=x_t.device)
        channel_tokens = channel_tokens + self.channel_embed(channel_ids)[None, :, :].to(dtype=dtype)
        channel_tokens = channel_tokens + self.type_embed.weight[1].to(dtype=dtype)[None, None, :] + time_code

        spectral_stats = self._spectral_stats(x_t)
        spectral_tokens = self.spectral_proj(spectral_stats[:, :, None])
        spectral_ids = torch.arange(self.num_spectral_tokens, device=x_t.device)
        spectral_tokens = spectral_tokens + self.spectral_embed(spectral_ids)[None, :, :].to(dtype=dtype)
        spectral_tokens = spectral_tokens + self.type_embed.weight[2].to(dtype=dtype)[None, None, :] + time_code

        return self.state_norm(torch.cat([time_tokens, channel_tokens, spectral_tokens], dim=1))

    def _spectral_stats(self, x_t: torch.Tensor) -> torch.Tensor:
        centered = x_t - x_t.mean(dim=1, keepdim=True)
        fft_source = centered.transpose(1, 2)
        fft_dtype = torch.float32 if fft_source.dtype in {torch.float16, torch.bfloat16} else fft_source.dtype
        freq = torch.fft.rfft(fft_source.to(fft_dtype), dim=-1)
        power = freq.abs().square().mean(dim=1)
        freq_len = power.shape[-1]
        bands = []
        for idx in range(self.num_spectral_tokens):
            start = int(math.floor(idx * freq_len / self.num_spectral_tokens))
            end = int(math.floor((idx + 1) * freq_len / self.num_spectral_tokens))
            end = max(start + 1, min(end, freq_len))
            bands.append(power[:, start:end].mean(dim=-1))
        stats = torch.stack(bands, dim=-1)
        stats = torch.log1p(stats)
        return stats.to(dtype=x_t.dtype)

    def _alignment_loss(self, text_context: torch.Tensor, state_context: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        text = F.normalize(self.text_align(text_context), dim=-1)
        state = F.normalize(self.state_align(state_context), dim=-1)
        logits = text @ state.transpose(0, 1)
        logits = logits / max(self.alignment_temperature, 1e-4)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))
        pos = logits.diag()
        return loss, {
            "bridge_alignment_logit_pos": pos.detach().mean(),
            "bridge_alignment_logit_std": logits.detach().std(unbiased=False),
        }


def _ffn(d_model: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, 4 * d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(4 * d_model, d_model),
        nn.Dropout(dropout),
    )


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=tokens.device, dtype=tokens.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (tokens * mask[:, :, None]).sum(dim=1) / denom


def _safe_key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0
    if valid.shape[1] == 0:
        raise ValueError("slot_mask must contain at least one slot")
    safe_valid = valid.clone()
    empty = ~safe_valid.any(dim=1)
    if empty.any():
        safe_valid[empty, 0] = True
    return ~safe_valid


def _num_patch_tokens(length: int, patch_size: int) -> int:
    effective = min(int(patch_size), int(length))
    return int(math.ceil(float(length) / float(max(effective, 1))))


def _nan_summary(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = reference.new_tensor(float("nan"))
    return value, value, value


def _zero_alignment_aux(reference: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return reference.new_zeros(()), {
        "bridge_alignment_logit_pos": reference.new_tensor(float("nan")),
        "bridge_alignment_logit_std": reference.new_tensor(float("nan")),
    }


def _balanced_sigmoid_contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError(f"contrastive logits must be square [B,B], got {tuple(logits.shape)}")
    pos_loss = -F.logsigmoid(torch.diagonal(logits)).mean()
    if logits.shape[0] <= 1:
        return pos_loss
    eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    neg_loss = -F.logsigmoid(-logits.masked_select(~eye)).mean()
    return 0.5 * (pos_loss + neg_loss)


def _attention_summary(
    attn: torch.Tensor,
    *,
    query_mask: torch.Tensor | None = None,
    key_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # attn: [B, heads, query_tokens, key_tokens]
    prob = attn.detach()
    batch, heads, queries, keys = prob.shape
    dtype = prob.dtype
    device = prob.device
    if key_mask is not None:
        key_mask = key_mask.to(device=device, dtype=dtype)
        if key_mask.shape != (batch, keys):
            raise ValueError(f"key_mask must have shape {(batch, keys)}, got {tuple(key_mask.shape)}")
        key_weight = key_mask[:, None, None, :]
        prob_for_entropy = prob * key_weight
        valid_key_count = key_mask.sum(dim=-1).clamp_min(2.0)
        norm_denom = valid_key_count.log()[:, None, None]
    else:
        key_weight = None
        prob_for_entropy = prob
        norm_denom = torch.full((batch, 1, 1), math.log(float(max(keys, 2))), device=device, dtype=dtype)

    p = prob_for_entropy.clamp_min(eps)
    entropy_per_query = -(p * p.log()).sum(dim=-1)
    if key_weight is not None:
        entropy_per_query = entropy_per_query * (key_mask.sum(dim=-1) > 0).to(dtype)[:, None, None]
        max_per_query = (prob * key_weight).max(dim=-1).values
    else:
        max_per_query = prob.max(dim=-1).values
    entropy_norm_per_query = entropy_per_query / norm_denom.clamp_min(eps)

    if query_mask is not None:
        query_mask = query_mask.to(device=device, dtype=dtype)
        if query_mask.shape != (batch, queries):
            raise ValueError(f"query_mask must have shape {(batch, queries)}, got {tuple(query_mask.shape)}")
        query_weight = query_mask[:, None, :]
        denom = (query_weight.sum() * float(heads)).clamp_min(1.0)
        entropy = (entropy_per_query * query_weight).sum() / denom
        max_prob = (max_per_query * query_weight).sum() / denom
        entropy_norm = (entropy_norm_per_query * query_weight).sum() / denom
    else:
        entropy = entropy_per_query.mean()
        max_prob = max_per_query.mean()
        entropy_norm = entropy_norm_per_query.mean()
    return entropy, max_prob, entropy_norm
