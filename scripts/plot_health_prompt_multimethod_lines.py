from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_COLORS = {
    "Ours": "#005F73",
    "VerbalTS": "#CA6702",
    "Text2Motion": "#7B6D8D",
    "TEdit": "#6A994E",
    "TimeVQVAE": "#B56576",
}

_PROMPT_EMBEDDING_CACHE: dict[tuple[str, str, int, str], torch.Tensor] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a single Health_US_96 prompt with Ours, VerbalTS, and selected "
            "ConTSG-Bench methods, then draw an overlaid line chart."
        )
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/health_prompt_lines"))
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--num-channels", type=int, default=None)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--aggregate", choices=("median", "mean", "first"), default="median")
    parser.add_argument(
        "--plot-normalize",
        choices=("none", "per-series-minmax", "global-minmax", "per-series-zscore"),
        default="per-series-minmax",
    )
    parser.add_argument("--dpi", type=int, default=600)

    parser.add_argument("--data-root", type=Path, required=True, help="Health_US_96 dataset root for shape inference.")

    parser.add_argument("--ours-config", type=Path, default=None)
    parser.add_argument("--ours-checkpoint", type=Path, default=None)
    parser.add_argument("--ours-text-encoder-model", default=None)

    parser.add_argument("--verbalts-root", type=Path, default=None)
    parser.add_argument("--verbalts-run-dir", type=Path, default=None)
    parser.add_argument("--verbalts-checkpoint", type=Path, default=None)
    parser.add_argument("--verbalts-diff-config", type=Path, default=None)
    parser.add_argument("--verbalts-cond-config", type=Path, default=None)
    parser.add_argument("--verbalts-longclip-root", type=Path, default=None)
    parser.add_argument("--verbalts-base-patch", type=int, default=None)
    parser.add_argument("--verbalts-multipatch-num", type=int, default=None)
    parser.add_argument("--verbalts-l-patch-len", type=int, default=None)

    parser.add_argument("--contsg-root", type=Path, default=None)
    parser.add_argument(
        "--contsg-method",
        action="append",
        default=[],
        help="ConTSG method in LABEL=EXPERIMENT_DIR format. Repeat for Text2Motion/TEdit/TimeVQVAE.",
    )
    parser.add_argument("--contsg-checkpoint", default="best")
    parser.add_argument("--contsg-sampler", default="ddim")
    parser.add_argument("--cttp-config", type=Path, default=None)
    parser.add_argument("--cttp-checkpoint", type=Path, default=None)
    parser.add_argument("--cttp-text-encoder-model", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    seq_len, num_channels = resolve_shape(args.data_root, args.sequence_length, args.num_channels)
    if args.channel < 0 or args.channel >= num_channels:
        raise ValueError(f"--channel must be in [0, {num_channels}), got {args.channel}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    series: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {
        "prompt": args.prompt,
        "sequence_length": seq_len,
        "num_channels": num_channels,
        "n_samples": int(args.n_samples),
        "aggregate": args.aggregate,
        "plot_normalize": args.plot_normalize,
        "methods": {},
    }

    if args.ours_config is not None and args.ours_checkpoint is not None:
        arr = generate_ours(args, device, seq_len, num_channels)
        save_method(args.output_dir, "Ours", arr)
        series["Ours"] = aggregate_samples(arr[0], args.aggregate)
        metadata["methods"]["Ours"] = {
            "config": str(args.ours_config),
            "checkpoint": str(args.ours_checkpoint),
        }

    if args.verbalts_root is not None and args.verbalts_run_dir is not None:
        arr = generate_verbalts(args, device, seq_len, num_channels)
        save_method(args.output_dir, "VerbalTS", arr)
        series["VerbalTS"] = aggregate_samples(arr[0], args.aggregate)
        metadata["methods"]["VerbalTS"] = {
            "run_dir": str(args.verbalts_run_dir),
            "checkpoint": str(args.verbalts_checkpoint or args.verbalts_run_dir / "ckpts" / "model_best_loss.pth"),
        }

    contsg_methods = parse_label_paths(args.contsg_method)
    if contsg_methods:
        missing = [
            name
            for name, value in {
                "contsg-root": args.contsg_root,
                "cttp-config": args.cttp_config,
                "cttp-checkpoint": args.cttp_checkpoint,
                "cttp-text-encoder-model": args.cttp_text_encoder_model,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError("ConTSG generation requires: " + ", ".join(missing))
        for label, experiment in contsg_methods:
            arr = generate_contsg(args, device, seq_len, num_channels, label, experiment)
            save_method(args.output_dir, label, arr)
            series[label] = aggregate_samples(arr[0], args.aggregate)
            metadata["methods"][label] = {"experiment": str(experiment), "checkpoint": str(args.contsg_checkpoint)}

    if not series:
        raise ValueError("No methods were enabled. Provide Ours, VerbalTS, or --contsg-method arguments.")

    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    figure = args.figure or args.output_dir / "health_prompt_multimethod_lines.png"
    plot_lines(
        series=series,
        prompt=args.prompt,
        channel=int(args.channel),
        output=figure,
        normalize=args.plot_normalize,
        dpi=int(args.dpi),
    )
    if figure.suffix.lower() != ".pdf":
        plot_lines(
            series=series,
            prompt=args.prompt,
            channel=int(args.channel),
            output=figure.with_suffix(".pdf"),
            normalize=args.plot_normalize,
            dpi=int(args.dpi),
        )
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "figure": str(figure.resolve())}, indent=2, ensure_ascii=False))


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def resolve_shape(data_root: Path, seq: int | None, channels: int | None) -> tuple[int, int]:
    if seq is not None and channels is not None:
        return int(seq), int(channels)
    arr = np.load(data_root / "train_ts.npy", mmap_mode="r", allow_pickle=False)
    if arr.ndim == 2:
        raw_seq, raw_channels = int(arr.shape[1]), 1
    elif arr.ndim == 3:
        raw_seq, raw_channels = int(arr.shape[1]), int(arr.shape[2])
    else:
        raise ValueError(f"{data_root / 'train_ts.npy'} must be [N,L,C] or [N,L], got {arr.shape}")
    return int(seq or raw_seq), int(channels or raw_channels)


def generate_ours(args: argparse.Namespace, device: torch.device, seq_len: int, num_channels: int) -> np.ndarray:
    from effectcma_flow.config import load_config
    from effectcma_flow.evaluation.sampler import sample_text2ts
    from effectcma_flow.evaluation.verbalts_metrics import _load_contsg_cttp
    from effectcma_flow.models import build_model
    from effectcma_flow.training import checkpoint_eval_config, checkpoint_stats_or_none, load_training_checkpoint
    from scripts.generate_text2ts import _denormalize_if_needed, _generate

    cfg = load_config(args.ours_config)
    cfg.setdefault("data", {})["root"] = str(args.data_root)
    cfg.setdefault("task", {})["mode"] = "text2ts"
    if args.ours_text_encoder_model is not None:
        text_cfg = cfg.setdefault("text_encoder", {})
        mode = str(text_cfg.get("mode", "")).lower()
        text_cfg["longclip_model_name" if mode == "longclip" else "hf_model_name"] = args.ours_text_encoder_model

    payload = load_training_checkpoint(args.ours_checkpoint, device)
    cfg = checkpoint_eval_config(cfg, payload, task_mode_override="text2ts")
    stats = checkpoint_stats_or_none(payload)
    model = build_model(cfg, sequence_length=seq_len, num_channels=num_channels).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    sample_cfg = cfg.get("sample", {})
    if str(cfg.get("text_encoder", {}).get("mode", "")).lower() == "precomputed":
        missing = [
            name
            for name, value in {
                "contsg-root": args.contsg_root,
                "cttp-config": args.cttp_config,
                "cttp-checkpoint": args.cttp_checkpoint,
                "cttp-text-encoder-model": args.cttp_text_encoder_model,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError("Ours uses precomputed text embeddings; provide " + ", ".join(missing))
        clip = _load_contsg_cttp(
            contsg_root=args.contsg_root.expanduser().resolve(),
            clip_config_path=args.cttp_config.expanduser().resolve(),
            clip_model_path=args.cttp_checkpoint.expanduser().resolve(),
            text_encoder_model=args.cttp_text_encoder_model.expanduser().resolve(),
            device=device,
        )
        expected_dim = int(cfg.get("text_encoder", {}).get("precomputed_dim", infer_training_cap_embedding_dim(args.data_root)))
        prompt_emb = prompt_embedding_in_training_space(args, clip, device, expected_dim)
        dtype = next(model.parameters()).dtype
        generator = torch.Generator(device=device)
        generator.manual_seed(int(args.seed))
        normalized = np.empty((1, int(args.n_samples), seq_len, num_channels), dtype=np.float32)
        for start in range(0, int(args.n_samples), int(args.batch_size)):
            width = min(int(args.batch_size), int(args.n_samples) - start)
            shape_like = torch.empty((width, seq_len, num_channels), device=device, dtype=dtype)
            text_condition = {
                "embeddings": prompt_emb.to(device=device, dtype=dtype).repeat(width, 1),
                "mask": torch.ones((width, 1), device=device, dtype=dtype),
            }
            generated, _ = sample_text2ts(
                model,
                shape_like,
                text_condition,
                solver=str(sample_cfg.get("solver", "rk4")),
                steps=int(sample_cfg.get("steps", 64)),
                noise_scale=float(sample_cfg.get("noise_scale", 1.0)),
                generator=generator,
                cfg_scale=float(sample_cfg.get("cfg_scale", 1.0)),
                guidance_t_lo=float(sample_cfg.get("guidance_t_lo", 0.0)),
                guidance_t_hi=float(sample_cfg.get("guidance_t_hi", 1.0)),
            )
            normalized[0, start : start + width] = generated.detach().cpu().float().numpy()
    else:
        normalized = _generate(
            model=model,
            captions=[[args.prompt]],
            sequence_length=seq_len,
            num_channels=num_channels,
            num_samples=int(args.n_samples),
            batch_size=int(args.batch_size),
            device=device,
            solver=str(sample_cfg.get("solver", "rk4")),
            steps=int(sample_cfg.get("steps", 64)),
            noise_scale=float(sample_cfg.get("noise_scale", 1.0)),
            cfg_scale=float(sample_cfg.get("cfg_scale", 1.0)),
            guidance_t_lo=float(sample_cfg.get("guidance_t_lo", 0.0)),
            guidance_t_hi=float(sample_cfg.get("guidance_t_hi", 1.0)),
            seed=int(args.seed),
        )
    return _denormalize_if_needed(normalized, cfg, stats).astype(np.float32)


def generate_verbalts(args: argparse.Namespace, device: torch.device, seq_len: int, num_channels: int) -> np.ndarray:
    verbalts_root = args.verbalts_root.expanduser().resolve()
    sys.path.insert(0, str(verbalts_root))

    from models.conditional_generator import ConditionalGenerator
    from scripts.eval_verbalts_official_contsg_metrics import configure_verbalts_model, read_yaml

    diff_config = (
        args.verbalts_diff_config
        or verbalts_root / "configs" / "synth-m" / "diff" / "model_text2ts_dep.yaml"
    ).expanduser().resolve()
    cond_config = (
        args.verbalts_cond_config
        or verbalts_root / "configs" / "synth-m" / "cond" / "text_msmdiffmv.yaml"
    ).expanduser().resolve()
    checkpoint = (
        args.verbalts_checkpoint
        or args.verbalts_run_dir / "ckpts" / "model_best_loss.pth"
    ).expanduser().resolve()
    longclip_root = (args.verbalts_longclip_root or verbalts_root / "save" / "Longclip").expanduser().resolve()

    diff_cfg = read_yaml(diff_config)
    cond_cfg = read_yaml(cond_config)
    configure_verbalts_model(
        diff_cfg,
        cond_cfg,
        device=device,
        longclip_root=longclip_root,
        cond_modal="text",
        text_output_type="all",
        text_pos_emb="none",
        diff_stage_num=3,
        base_patch=args.verbalts_base_patch,
        multipatch_num=args.verbalts_multipatch_num,
        l_patch_len=args.verbalts_l_patch_len,
    )
    model = ConditionalGenerator(diff_cfg, cond_cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    batch = {
        "ts": torch.zeros((1, seq_len, num_channels), device=device, dtype=torch.float32),
        "tp": torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0),
        "cap": [args.prompt],
    }
    with torch.no_grad():
        preds = model.generate(batch, int(args.n_samples), sampler="ddim")
    # VerbalTS returns [S, B, C, L]. Convert to [B, S, L, C].
    return preds.permute(1, 0, 3, 2).detach().cpu().float().numpy().astype(np.float32)


def generate_contsg(
    args: argparse.Namespace,
    device: torch.device,
    seq_len: int,
    num_channels: int,
    label: str,
    experiment: Path,
) -> np.ndarray:
    contsg_root = args.contsg_root.expanduser().resolve()
    sys.path.insert(0, str(contsg_root))

    from contsg.eval import Evaluator
    from effectcma_flow.evaluation.verbalts_metrics import _load_contsg_cttp

    evaluator = Evaluator.from_experiment(
        experiment.expanduser().resolve(),
        checkpoint=str(args.contsg_checkpoint),
        device=device,
        cache_only=False,
    )
    embedder = _load_contsg_cttp(
        contsg_root=contsg_root,
        clip_config_path=args.cttp_config.expanduser().resolve(),
        clip_model_path=args.cttp_checkpoint.expanduser().resolve(),
        text_encoder_model=args.cttp_text_encoder_model.expanduser().resolve(),
        device=device,
    )
    target_dim = infer_training_cap_embedding_dim(args.data_root)
    cap_emb = prompt_embedding_in_training_space(args, embedder, device, target_dim)
    batch = {
        "ts": torch.zeros((1, seq_len, num_channels), device=device, dtype=torch.float32),
        "tp": torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0),
        "cap": [args.prompt],
        "cap_emb": cap_emb.to(device),
    }
    with torch.no_grad():
        moved = evaluator._move_batch_to_device(batch)
        preds = evaluator._generate_predictions(moved, int(args.n_samples), str(args.contsg_sampler))
    # Evaluator normalizes to [S, B, L, C]. Convert to [B, S, L, C].
    return preds.permute(1, 0, 2, 3).detach().cpu().float().numpy().astype(np.float32)


def parse_label_paths(items: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError("--contsg-method must use LABEL=EXPERIMENT_DIR")
        label, path = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError("Empty method label")
        parsed.append((label, Path(path.strip())))
    return parsed


def cttp_text_embedding(clip: Any, texts: list[str]) -> torch.Tensor:
    """Return text embeddings from either ConTSG CLIPEmbedder or EffectCMA's CTTP adapter."""
    with torch.no_grad():
        if hasattr(clip, "get_text_embedding"):
            return clip.get_text_embedding({"cap": texts})
        if hasattr(clip, "get_text_coemb"):
            return clip.get_text_coemb(texts, None)
        embedder = getattr(clip, "embedder", None)
        if embedder is not None and hasattr(embedder, "get_text_embedding"):
            return embedder.get_text_embedding({"cap": texts})
    raise AttributeError(
        "CTTP object does not expose get_text_embedding, get_text_coemb, "
        "or embedder.get_text_embedding"
    )


def prompt_embedding_in_training_space(
    args: argparse.Namespace,
    clip: Any,
    device: torch.device,
    target_dim: int,
) -> torch.Tensor:
    """Map a free-text CTTP embedding into the dataset's precomputed cap_emb space."""
    cache_key = (str(args.data_root.expanduser().resolve()), args.prompt, int(target_dim), str(device))
    cached = _PROMPT_EMBEDDING_CACHE.get(cache_key)
    if cached is not None:
        return cached.to(device)

    prompt_raw = cttp_text_embedding(clip, [args.prompt]).detach().float()
    if int(prompt_raw.shape[-1]) == int(target_dim):
        out = prompt_raw.to(device)
        _PROMPT_EMBEDDING_CACHE[cache_key] = out.detach().cpu()
        return out

    train_caps = load_train_captions(args.data_root)
    target = load_train_cap_embeddings(args.data_root, int(target_dim))
    raw_batches = []
    encode_batch = max(1, min(int(args.batch_size), 128))
    for start in range(0, len(train_caps), encode_batch):
        raw = cttp_text_embedding(clip, train_caps[start : start + encode_batch]).detach().float().cpu()
        raw_batches.append(raw)
    source = torch.cat(raw_batches, dim=0)
    if source.shape[0] != target.shape[0]:
        raise ValueError(f"CTTP source rows {source.shape[0]} != target cap_emb rows {target.shape[0]}")

    mapped = fit_ridge_and_project(source, torch.from_numpy(target).float(), prompt_raw.cpu())
    out = mapped.to(device)
    _PROMPT_EMBEDDING_CACHE[cache_key] = out.detach().cpu()
    print(
        f"[embedding-map] CTTP dim {prompt_raw.shape[-1]} -> training cap_emb dim {target_dim} "
        f"using {source.shape[0]} Health train captions",
        flush=True,
    )
    return out


def load_train_captions(data_root: Path) -> list[str]:
    caps = np.load(data_root / "train_text_caps.npy", allow_pickle=True)
    if caps.ndim == 1:
        return [str(item).strip() for item in caps]
    if caps.ndim == 2:
        out = []
        for row in caps:
            parts = [str(item).strip() for item in row if str(item).strip()]
            out.append(" ".join(parts))
        return out
    raise ValueError(f"train_text_caps.npy must be [N] or [N,J], got {caps.shape}")


def load_train_cap_embeddings(data_root: Path, target_dim: int) -> np.ndarray:
    candidates = [
        data_root / "train_cap_emb.npy",
        data_root / f"train_text_caps_embeddings_{target_dim}.npy",
        data_root / "train_text_caps_embeddings_128.npy",
    ]
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Could not find train cap embeddings under {data_root}")
    values = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    if values.ndim == 3 and values.shape[1] == 1:
        values = values[:, 0, :]
    if values.ndim != 2 or int(values.shape[-1]) != int(target_dim):
        raise ValueError(f"{path} must have shape [N,{target_dim}] or [N,1,{target_dim}], got {values.shape}")
    return values


def infer_training_cap_embedding_dim(data_root: Path) -> int:
    values = load_train_cap_embeddings(data_root, 128)
    return int(values.shape[-1])


def fit_ridge_and_project(source: torch.Tensor, target: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    if source.ndim != 2 or target.ndim != 2 or query.ndim != 2:
        raise ValueError("source, target, and query must be rank-2 tensors")
    if source.shape[0] != target.shape[0]:
        raise ValueError(f"source rows {source.shape[0]} != target rows {target.shape[0]}")

    x_mean = source.mean(dim=0, keepdim=True)
    x_std = source.std(dim=0, keepdim=True).clamp_min(1e-6)
    x = (source - x_mean) / x_std
    q = (query - x_mean) / x_std

    ones = torch.ones((x.shape[0], 1), dtype=x.dtype)
    x_aug = torch.cat([x, ones], dim=1)
    q_aug = torch.cat([q, torch.ones((q.shape[0], 1), dtype=q.dtype)], dim=1)

    ridge = 1e-3
    xtx = x_aug.T @ x_aug
    reg = torch.eye(xtx.shape[0], dtype=xtx.dtype) * ridge
    reg[-1, -1] = 0.0
    rhs = x_aug.T @ target
    weights = torch.linalg.solve(xtx + reg, rhs)
    return (q_aug @ weights).float()


def save_method(output_dir: Path, label: str, values: np.ndarray) -> None:
    safe = label.lower().replace(" ", "_").replace("/", "_")
    np.save(output_dir / f"{safe}.npy", values.astype(np.float32, copy=False))
    print(f"[saved] {label}: {output_dir / f'{safe}.npy'} shape={values.shape}")


def aggregate_samples(values: np.ndarray, aggregate: str) -> np.ndarray:
    # values: [S, L, C]
    if aggregate == "first":
        return values[0]
    if aggregate == "mean":
        return values.mean(axis=0)
    if aggregate == "median":
        return np.median(values, axis=0)
    raise ValueError(aggregate)


def normalize_lines(lines: dict[str, np.ndarray], mode: str, channel: int) -> dict[str, np.ndarray]:
    raw = {label: arr[:, channel].astype(np.float64) for label, arr in lines.items()}
    if mode == "none":
        return raw
    if mode == "global-minmax":
        stacked = np.concatenate(list(raw.values()))
        lo, hi = float(stacked.min()), float(stacked.max())
        denom = hi - lo if hi > lo else 1.0
        return {label: (line - lo) / denom for label, line in raw.items()}
    out = {}
    for label, line in raw.items():
        if mode == "per-series-zscore":
            std = float(line.std())
            out[label] = (line - float(line.mean())) / (std if std > 1e-12 else 1.0)
        elif mode == "per-series-minmax":
            lo, hi = float(line.min()), float(line.max())
            out[label] = (line - lo) / (hi - lo if hi > lo else 1.0)
        else:
            raise ValueError(mode)
    return out


def plot_lines(
    *,
    series: dict[str, np.ndarray],
    prompt: str,
    channel: int,
    output: Path,
    normalize: str,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = normalize_lines(series, normalize, channel)
    fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=dpi)
    x = np.arange(next(iter(lines.values())).shape[0])
    for label, y in lines.items():
        ax.plot(x, y, color=DEFAULT_COLORS.get(label, "#333333"), linewidth=1.8, label=label)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Normalized Value" if normalize != "none" else "Value")
    ax.set_title("\n".join(textwrap.wrap(prompt, width=96)), fontsize=8.8, pad=8)
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.legend(loc="upper right", ncol=3, fontsize=8, frameon=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] figure: {output}")


if __name__ == "__main__":
    main()
