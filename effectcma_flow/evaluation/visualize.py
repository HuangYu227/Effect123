from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def dump_sample_plot(path: str | Path, base: torch.Tensor, target: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, *, channel: int = 0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = base.detach().cpu()
    target = target.detach().cpu()
    pred = pred.detach().cpu()
    mask = mask.detach().cpu()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(base[:, channel], label="B")
    ax.plot(target[:, channel], label="Y")
    ax.plot(pred[:, channel], label="Y_hat")
    active = mask[:, channel] > 0.5
    if active.any():
        idx = active.nonzero(as_tuple=False).flatten()
        ax.axvspan(int(idx.min()), int(idx.max()), color="tab:orange", alpha=0.15)
    ax.legend(loc="best")
    ax.set_title(f"channel {channel}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def dump_text2ts_plot(path: str | Path, target: torch.Tensor, pred: torch.Tensor, *, channel: int = 0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = target.detach().cpu()
    pred = pred.detach().cpu()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(target[:, channel], label="target")
    ax.plot(pred[:, channel], label="generated")
    ax.legend(loc="best")
    ax.set_title(f"channel {channel}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def dump_generated_plot(path: str | Path, pred: torch.Tensor, *, channel: int = 0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = pred.detach().cpu()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(pred[:, channel], label="generated")
    ax.legend(loc="best")
    ax.set_title(f"channel {channel}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
