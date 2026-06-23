from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class VerbalTSCTTPDataset(Dataset):
    """Raw VerbalTS dataset split for CTTP training.

    This intentionally mirrors VerbalTS' own `CustomSplit`: raw time series are
    used directly, and one caption is selected from each sample.
    """

    def __init__(self, root: str | Path, split: str, *, caption_policy: str = "random", seed: int = 0) -> None:
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be one of {'train', 'valid', 'test'}")
        if caption_policy not in {"random", "first", "cyclic"}:
            raise ValueError("caption_policy must be one of {'random', 'first', 'cyclic'}")
        self.root = Path(root)
        self.split = split
        self.caption_policy = caption_policy
        self.seed = int(seed)
        self.ts = np.load(self.root / f"{split}_ts.npy", allow_pickle=False).astype(np.float32)
        self.caps = np.load(self.root / f"{split}_text_caps.npy", allow_pickle=True)
        if self.ts.ndim == 2:
            self.ts = self.ts[..., None]
        if self.ts.ndim != 3:
            raise ValueError(f"{split}_ts.npy must have shape [N, L, C], got {self.ts.shape}")
        if self.caps.ndim != 2 or self.caps.shape[0] != self.ts.shape[0]:
            raise ValueError(f"{split}_text_caps.npy must have shape [N, J], got {self.caps.shape}")
        self.epoch = 0

    def __len__(self) -> int:
        return int(self.ts.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.ts.shape[1])

    @property
    def num_channels(self) -> int:
        return int(self.ts.shape[2])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed + int(index) + 1_000_003 * self.epoch)
        cap_idx = self._caption_index(index, rng)
        return {
            "ts": torch.from_numpy(self.ts[index]).float(),
            "ts_len": torch.tensor(self.ts.shape[1], dtype=torch.long),
            "cap": str(self.caps[index, cap_idx]),
            "cap_idx": int(cap_idx),
        }

    def _caption_index(self, index: int, rng: np.random.Generator) -> int:
        if self.caption_policy == "first":
            return 0
        if self.caption_policy == "cyclic":
            return int(index) % int(self.caps.shape[1])
        return int(rng.integers(0, self.caps.shape[1]))


def collate_cttp(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ts": torch.stack([s["ts"] for s in samples]),
        "ts_len": torch.stack([s["ts_len"] for s in samples]),
        "cap": [s["cap"] for s in samples],
        "cap_idx": torch.tensor([s["cap_idx"] for s in samples], dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VerbalTS-style CTTP evaluator.")
    parser.add_argument("--verbalts-root", required=True, help="Path to the VerbalTS repository.")
    parser.add_argument("--dataset-root", required=True, help="Path containing train/valid/test *_ts.npy and *_text_caps.npy.")
    parser.add_argument("--output-dir", required=True, help="Output folder, e.g. save/synth-u_cttp.")
    parser.add_argument("--longclip-root", required=True, help="Local LongCLIP directory used by VerbalTS CTTP.")
    parser.add_argument("--template-cttp-folder", default=None, help="Optional existing CTTP folder whose model config is reused.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--caption-policy", default="random", choices=["random", "first", "cyclic"])
    parser.add_argument("--loss-type", default="Contrastive", choices=["CE", "Contrastive"])
    parser.add_argument("--coemb-dim", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--text-hidden-dim", type=int, default=None)
    parser.add_argument("--patch-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--padding", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast. Disabled by default for reproducibility.")
    parser.add_argument("--save-every", type=int, default=0, help="Optional epoch checkpoint interval; 0 disables.")
    args = parser.parse_args()

    _set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    verbalts_root = Path(args.verbalts_root).resolve()
    if str(verbalts_root) not in sys.path:
        sys.path.insert(0, str(verbalts_root))
    from models.cttp.cttp_model import CTTP

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ds = VerbalTSCTTPDataset(args.dataset_root, "train", caption_policy=args.caption_policy, seed=args.seed)
    valid_ds = VerbalTSCTTPDataset(args.dataset_root, "valid", caption_policy="cyclic", seed=args.seed + 10_000)
    eval_batch_size = int(args.eval_batch_size or args.batch_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_cttp,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_cttp,
    )
    if len(train_loader) == 0:
        raise ValueError("Training loader is empty. Use a smaller batch size.")

    model_config = build_model_config(
        args=args,
        seq_len=train_ds.sequence_length,
        n_var=train_ds.num_channels,
        template_folder=args.template_cttp_folder,
        device=str(device),
    )
    model = CTTP(model_config).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = max(0, args.warmup_epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _warmup_cosine_factor(step, warmup_steps=warmup_steps, total_steps=total_steps),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    write_yaml(output_dir / "model_configs.yaml", _cpu_config(model_config))
    train_config = vars(args).copy()
    train_config.update(
        {
            "dataset_shape": {
                "train": [len(train_ds), train_ds.sequence_length, train_ds.num_channels],
                "valid": [len(valid_ds), valid_ds.sequence_length, valid_ds.num_channels],
            },
            "steps_per_epoch": len(train_loader),
            "total_steps": total_steps,
            "selected_metric": "valid_loss",
        }
    )
    write_yaml(output_dir / "train_configs.yaml", train_config)

    best_loss = math.inf
    best_record: dict[str, float] | None = None
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            grad_clip=args.grad_clip,
            amp=bool(args.amp),
            desc=f"train {epoch + 1}/{args.epochs}",
        )
        valid_metrics = evaluate(model=model, loader=valid_loader, device=device, amp=bool(args.amp))
        record = {
            "epoch": float(epoch + 1),
            **{f"train_{k}": float(v) for k, v in train_metrics.items()},
            **{f"valid_{k}": float(v) for k, v in valid_metrics.items()},
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        append_jsonl(output_dir / "metrics.jsonl", record)
        print(json.dumps(record, sort_keys=True))

        if valid_metrics["loss"] < best_loss:
            best_loss = float(valid_metrics["loss"])
            best_record = record
            torch.save(model.state_dict(), output_dir / "clip_model_best.pth")
            write_yaml(output_dir / "best_metrics.yaml", record)
            print(f"[cttp] saved best at epoch {epoch + 1}: valid_loss={best_loss:.6f}")
        torch.save(model.state_dict(), output_dir / "clip_model_last.pth")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), output_dir / f"clip_model_epoch_{epoch + 1:04d}.pth")

    if best_record is None:
        raise RuntimeError("Training finished without producing a best checkpoint.")
    print(f"[cttp] done. best valid_loss={best_loss:.6f}. output={output_dir}")


def build_model_config(
    *,
    args: argparse.Namespace,
    seq_len: int,
    n_var: int,
    template_folder: str | None,
    device: str,
) -> dict[str, Any]:
    if template_folder:
        config_path = Path(template_folder) / "model_configs.yaml"
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"Template config must be a mapping: {config_path}")
    else:
        cfg = default_model_config(seq_len=seq_len, n_var=n_var)

    cfg["device"] = device
    cfg["clip_type"] = "clip_patchtst"
    cfg["loss_type"] = args.loss_type
    cfg.setdefault("text", {})
    cfg.setdefault("ts", {})
    cfg["text"].update(
        {
            "pretrain_model_path": str(Path(args.longclip_root).resolve()),
            "pretrain_model_dim": int(cfg["text"].get("pretrain_model_dim", 768)),
            "textemb_hidden_dim": int(args.text_hidden_dim or cfg["text"].get("textemb_hidden_dim", 512)),
            "coemb_dim": int(args.coemb_dim or cfg["text"].get("coemb_dim", 512)),
            "output_type": str(cfg["text"].get("output_type", "all")),
        }
    )
    patch_len, stride, padding = resolve_patch_params(
        seq_len=seq_len,
        patch_len=args.patch_len,
        stride=args.stride,
        padding=args.padding,
        cfg=cfg.get("ts", {}),
    )
    cfg["ts"].update(
        {
            "type": "patchtst_mae_pretrain",
            "seq_len": int(seq_len),
            "n_var": int(n_var),
            "d_model": int(args.d_model or cfg["ts"].get("d_model", 64)),
            "coemb_dim": int(args.coemb_dim or cfg["ts"].get("coemb_dim", 512)),
            "patch_len": int(patch_len),
            "stride": int(stride),
            "padding": int(padding),
            "dropout": float(args.dropout if args.dropout is not None else cfg["ts"].get("dropout", 0.1)),
            "pretrain_encoder_path": str(cfg["ts"].get("pretrain_encoder_path", "")),
            "factor": int(cfg["ts"].get("factor", 1)),
            "output_attention": bool(cfg["ts"].get("output_attention", False)),
            "n_heads": int(cfg["ts"].get("n_heads", cfg["ts"].get("nheads", 8))),
            "d_ff": int(cfg["ts"].get("d_ff", 256)),
            "e_layers": int(cfg["ts"].get("e_layers", 2)),
            "activation": str(cfg["ts"].get("activation", "gelu")),
        }
    )
    return cfg


def default_model_config(*, seq_len: int, n_var: int) -> dict[str, Any]:
    patch_len, stride, padding = resolve_patch_params(seq_len=seq_len, patch_len=None, stride=None, padding=None, cfg={})
    return {
        "clip_type": "clip_patchtst",
        "loss_type": "Contrastive",
        "device": "cuda:0",
        "text": {
            "pretrain_model_path": "save/Longclip",
            "pretrain_model_dim": 768,
            "textemb_hidden_dim": 512,
            "coemb_dim": 512,
            "output_type": "all",
        },
        "ts": {
            "type": "patchtst_mae_pretrain",
            "seq_len": int(seq_len),
            "n_var": int(n_var),
            "d_model": 64,
            "coemb_dim": 512,
            "patch_len": int(patch_len),
            "stride": int(stride),
            "padding": int(padding),
            "dropout": 0.1,
            "pretrain_encoder_path": "",
            "factor": 1,
            "output_attention": False,
            "n_heads": 8,
            "d_ff": 256,
            "e_layers": 2,
            "activation": "gelu",
        },
    }


def resolve_patch_params(
    *,
    seq_len: int,
    patch_len: int | None,
    stride: int | None,
    padding: int | None,
    cfg: dict[str, Any],
) -> tuple[int, int, int]:
    p = int(patch_len if patch_len is not None else cfg.get("patch_len", min(32, max(4, seq_len // 4))))
    s = int(stride if stride is not None else cfg.get("stride", p))
    pad = int(padding if padding is not None else cfg.get("padding", max(0, p - s)))
    p = max(2, min(p, seq_len))
    s = max(1, min(s, p))
    pad = max(0, pad)
    return p, s, pad


def train_one_epoch(
    *,
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    grad_clip: float,
    amp: bool,
    desc: str,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "acc_ts2text": 0.0, "acc_text2ts": 0.0}
    count = 0
    for batch in tqdm(loader, desc=desc, dynamic_ncols=True):
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=bool(amp and device.type == "cuda")):
            losses = model(batch["ts"], batch["ts_len"], batch["cap"], None)
            loss = losses["all"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        metrics = batch_retrieval_metrics(model, batch)
        bsz = int(batch["ts"].shape[0])
        totals["loss"] += float(loss.detach().item()) * bsz
        totals["acc_ts2text"] += metrics["acc_ts2text"] * bsz
        totals["acc_text2ts"] += metrics["acc_text2ts"] * bsz
        count += bsz
    return {k: v / max(1, count) for k, v in totals.items()}


@torch.no_grad()
def evaluate(*, model, loader, device: torch.device, amp: bool) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "acc_ts2text": 0.0, "acc_text2ts": 0.0, "diag_sim": 0.0}
    count = 0
    for batch in tqdm(loader, desc="valid", dynamic_ncols=True):
        batch = to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=bool(amp and device.type == "cuda")):
            losses = model(batch["ts"], batch["ts_len"], batch["cap"], None)
            loss = losses["all"]
        metrics = batch_retrieval_metrics(model, batch)
        bsz = int(batch["ts"].shape[0])
        totals["loss"] += float(loss.detach().item()) * bsz
        totals["acc_ts2text"] += metrics["acc_ts2text"] * bsz
        totals["acc_text2ts"] += metrics["acc_text2ts"] * bsz
        totals["diag_sim"] += metrics["diag_sim"] * bsz
        count += bsz
    return {k: v / max(1, count) for k, v in totals.items()}


@torch.no_grad()
def batch_retrieval_metrics(model, batch: dict[str, Any]) -> dict[str, float]:
    ts_emb = model.get_ts_coemb(batch["ts"], batch["ts_len"])
    text_emb = model.get_text_coemb(batch["cap"], None)
    sim = torch.mm(ts_emb, text_emb.t())
    labels = torch.arange(sim.shape[0], device=sim.device)
    return {
        "acc_ts2text": float((sim.argmax(dim=1) == labels).float().mean().item()),
        "acc_text2ts": float((sim.argmax(dim=0) == labels).float().mean().item()),
        "diag_sim": float(sim.diag().mean().item()),
    }


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def _warmup_cosine_factor(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-8, float(step + 1) / float(warmup_steps))
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(1e-4, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cpu_config(config: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(config))


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
