from __future__ import annotations

import torch
from torch import nn


class TimePatchEncoder(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        num_channels: int,
        patch_len: int,
        d_model: int,
        *,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.num_channels = int(num_channels)
        self.patch_len = int(patch_len)
        self.num_patches = (self.sequence_length + self.patch_len - 1) // self.patch_len
        self.proj = nn.Linear(self.patch_len * self.num_channels, d_model)
        self.pos = nn.Parameter(torch.randn(self.num_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        if base.ndim != 3:
            raise ValueError(f"base must be [B, L, C], got {tuple(base.shape)}")
        batch, length, channels = base.shape
        if length != self.sequence_length or channels != self.num_channels:
            raise ValueError(f"Expected [*, {self.sequence_length}, {self.num_channels}], got {tuple(base.shape)}")
        padded_length = self.num_patches * self.patch_len
        if padded_length > length:
            pad = base.new_zeros(batch, padded_length - length, channels)
            base = torch.cat([base, pad], dim=1)
        x = base.reshape(batch, self.num_patches, self.patch_len, channels)
        x = x.reshape(batch, self.num_patches, self.patch_len * channels)
        tokens = self.proj(x) + self.pos[None, :, :]
        return self.norm(self.encoder(tokens))


class ChannelEncoder(nn.Module):
    """Patch-aware channel encoder with shared temporal weights and channel mixing.

    The first stage follows the PatchTST idea of learning local temporal patch
    tokens with shared weights across channels. The second stage mixes channel
    tokens so the mapper can identify both independent and coupled variables.
    """

    def __init__(
        self,
        sequence_length: int,
        num_channels: int,
        d_model: int,
        *,
        patch_len: int | None = None,
        stride: int | None = None,
        temporal_layers: int = 1,
        channel_layers: int = 1,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.num_channels = int(num_channels)
        self.patch_len = int(patch_len or min(self.sequence_length, 8))
        self.patch_len = max(1, min(self.patch_len, self.sequence_length))
        self.stride = int(stride or self.patch_len)
        self.stride = max(1, min(self.stride, self.patch_len))
        if self.sequence_length <= self.patch_len:
            self.num_patches = 1
        else:
            tail = self.sequence_length - self.patch_len
            self.num_patches = 1 + (tail + self.stride - 1) // self.stride
        if d_model % heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by channel encoder heads={heads}")

        self.patch_proj = nn.Linear(self.patch_len, d_model)
        self.temporal_pos = nn.Parameter(torch.randn(self.num_patches, d_model) * 0.02)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=max(1, int(temporal_layers)))
        self.channel_emb = nn.Parameter(torch.randn(self.num_channels, d_model) * 0.02)
        channel_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.channel_mixer = nn.TransformerEncoder(channel_layer, num_layers=max(1, int(channel_layers)))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        if base.ndim != 3:
            raise ValueError(f"base must be [B, L, C], got {tuple(base.shape)}")
        batch, length, channels = base.shape
        if length != self.sequence_length or channels != self.num_channels:
            raise ValueError(f"Expected [*, {self.sequence_length}, {self.num_channels}], got {tuple(base.shape)}")
        x = base.transpose(1, 2)
        padded_length = (self.num_patches - 1) * self.stride + self.patch_len
        if padded_length > length:
            pad = x.new_zeros(batch, channels, padded_length - length)
            x = torch.cat([x, pad], dim=-1)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patches = patches.reshape(batch * channels, self.num_patches, self.patch_len)
        patch_tokens = self.patch_proj(patches) + self.temporal_pos[None, :, :]
        patch_tokens = self.temporal_encoder(patch_tokens)
        channel_tokens = patch_tokens.mean(dim=1).reshape(batch, channels, -1)
        channel_tokens = channel_tokens + self.channel_emb[None, :, :]
        return self.norm(self.channel_mixer(channel_tokens))
