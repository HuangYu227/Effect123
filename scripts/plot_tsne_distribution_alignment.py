from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


GROUND_TRUTH_COLOR = "#2F4F4F"
OURS_COLOR = "#B71C1C"
VERBALTS_COLOR = "#1A237E"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    real_root: Path
    ours_path: Path
    verbalts_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw t-SNE distribution alignment for Ground Truth, Ours, and VerbalTS. "
            "Each dataset is embedded once using Real+Ours+VerbalTS, then shown as "
            "two panels: Real vs Ours and Real vs VerbalTS."
        )
    )
    parser.add_argument(
        "--synthm-real-root",
        type=Path,
        default=Path(r"C:\Users\蜂窝煤\Desktop\npy\verbalts\synth-m"),
        help="Directory containing Synth-M test_ts.npy.",
    )
    parser.add_argument(
        "--health-real-root",
        type=Path,
        default=Path(r"E:\Research\TSG\EffectCMA\AddDataset\Health_US_96"),
        help="Directory containing Health_US_96 test_ts.npy.",
    )
    parser.add_argument("--synthm-ours", type=Path, required=True, help="Ours generated Synth-M .npy/.npz.")
    parser.add_argument("--synthm-verbalts", type=Path, required=True, help="VerbalTS generated Synth-M .npy/.npz.")
    parser.add_argument("--health-ours", type=Path, required=True, help="Ours generated Health_US_96 .npy/.npz.")
    parser.add_argument("--health-verbalts", type=Path, required=True, help="VerbalTS generated Health_US_96 .npy/.npz.")
    parser.add_argument("--real-split", default="test", choices=("train", "valid", "test"), help="Real split to plot.")
    parser.add_argument("--max-real", type=int, default=4000, help="Max real samples per dataset.")
    parser.add_argument("--max-gen", type=int, default=4000, help="Max generated samples per method per dataset.")
    parser.add_argument(
        "--generated-aggregate",
        default="median",
        choices=("median", "mean", "first", "all"),
        help=(
            "How to reduce generated samples if array shape is [N,S,L,C] or [S,N,L,C]. "
            "'all' flattens N and S."
        ),
    )
    parser.add_argument("--pca-dim", type=int, default=50, help="PCA dimension before t-SNE; disabled if <=0.")
    parser.add_argument("--perplexity", type=float, default=35.0, help="Requested t-SNE perplexity.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling and t-SNE.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"E:\Research\TSG\EffectCMA\outputs\figures\tsne_synthm_health_ours_verbalts.png"),
        help="Output image path. A PDF with the same stem is also saved unless --no-pdf is set.",
    )
    parser.add_argument("--no-pdf", action="store_true", help="Do not save a PDF copy.")
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args()


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            preferred = ("generated", "samples", "ts", "arr_0")
            for key in preferred:
                if key in data:
                    return np.asarray(data[key])
            keys = list(data.keys())
            if not keys:
                raise ValueError(f"No arrays found in {path}")
            return np.asarray(data[keys[0]])
    return np.asarray(np.load(path, allow_pickle=False))


def coerce_series(
    array: np.ndarray,
    *,
    expected_length: int,
    expected_channels: int,
    aggregate: Literal["median", "mean", "first", "all"],
    label: str,
) -> np.ndarray:
    x = np.asarray(array)
    x = np.squeeze(x)

    if x.ndim == 2:
        if x.shape[1] == expected_length * expected_channels:
            x = x.reshape(x.shape[0], expected_length, expected_channels)
        elif expected_channels == 1 and x.shape[1] == expected_length:
            x = x[:, :, None]
        else:
            raise ValueError(f"{label}: cannot interpret 2D shape {x.shape}")

    if x.ndim == 3:
        if x.shape[1:] == (expected_length, expected_channels):
            return x.astype(np.float32, copy=False)
        if x.shape[:2] == (expected_length, expected_channels):
            return np.moveaxis(x, -1, 0).astype(np.float32, copy=False)
        raise ValueError(
            f"{label}: expected [N,{expected_length},{expected_channels}], got {x.shape}"
        )

    if x.ndim == 4:
        if x.shape[-2:] != (expected_length, expected_channels):
            raise ValueError(
                f"{label}: expected last two dims ({expected_length},{expected_channels}), got {x.shape}"
            )
        # Common cases:
        #   [N, S, L, C] from conditional generation.
        #   [S, N, L, C] when sample dimension was stacked first.
        if x.shape[0] <= 64 and x.shape[1] > x.shape[0]:
            x = np.transpose(x, (1, 0, 2, 3))
        if aggregate == "median":
            x = np.median(x, axis=1)
        elif aggregate == "mean":
            x = np.mean(x, axis=1)
        elif aggregate == "first":
            x = x[:, 0]
        elif aggregate == "all":
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        else:
            raise ValueError(f"Unsupported aggregate: {aggregate}")
        return x.astype(np.float32, copy=False)

    raise ValueError(f"{label}: unsupported shape {x.shape}")


def load_real(root: Path, split: str) -> np.ndarray:
    path = root / f"{split}_ts.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing real split file: {path}")
    x = np.asarray(np.load(path, allow_pickle=False))
    if x.ndim == 2:
        x = x[:, :, None]
    if x.ndim != 3:
        raise ValueError(f"Real time series must be [N,L,C], got {x.shape} from {path}")
    return x.astype(np.float32, copy=False)


def subsample(x: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if max_n <= 0 or x.shape[0] <= max_n:
        return x
    idx = rng.choice(x.shape[0], size=max_n, replace=False)
    return x[np.sort(idx)]


def flatten_features(x: np.ndarray) -> np.ndarray:
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.reshape(x.shape[0], -1).astype(np.float32, copy=False)


def embed_dataset(
    *,
    real: np.ndarray,
    ours: np.ndarray,
    verbalts: np.ndarray,
    pca_dim: int,
    perplexity: float,
    seed: int,
) -> dict[str, np.ndarray]:
    labels = np.concatenate([
        np.full(real.shape[0], "real", dtype=object),
        np.full(ours.shape[0], "ours", dtype=object),
        np.full(verbalts.shape[0], "verbalts", dtype=object),
    ])
    features = np.concatenate([flatten_features(real), flatten_features(ours), flatten_features(verbalts)], axis=0)
    features = StandardScaler().fit_transform(features)

    n_samples, n_features = features.shape
    if pca_dim > 0 and n_features > pca_dim and n_samples > pca_dim + 2:
        features = PCA(n_components=pca_dim, random_state=seed).fit_transform(features)

    safe_perplexity = min(float(perplexity), max(5.0, (n_samples - 1) / 3.0))
    z = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        metric="euclidean",
    ).fit_transform(features)

    lo = z.min(axis=0, keepdims=True)
    hi = z.max(axis=0, keepdims=True)
    z = (z - lo) / np.maximum(hi - lo, 1e-8)
    return {
        "real": z[labels == "real"],
        "ours": z[labels == "ours"],
        "verbalts": z[labels == "verbalts"],
    }


def process_dataset(
    spec: DatasetSpec,
    *,
    real_split: str,
    max_real: int,
    max_gen: int,
    aggregate: Literal["median", "mean", "first", "all"],
    pca_dim: int,
    perplexity: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    rng = np.random.default_rng(seed)
    real = load_real(spec.real_root, real_split)
    length, channels = real.shape[1], real.shape[2]
    ours = coerce_series(
        load_array(spec.ours_path),
        expected_length=length,
        expected_channels=channels,
        aggregate=aggregate,
        label=f"{spec.name}/ours",
    )
    verbalts = coerce_series(
        load_array(spec.verbalts_path),
        expected_length=length,
        expected_channels=channels,
        aggregate=aggregate,
        label=f"{spec.name}/verbalts",
    )

    real = subsample(real, max_real, rng)
    ours = subsample(ours, max_gen, rng)
    verbalts = subsample(verbalts, max_gen, rng)
    embedding = embed_dataset(
        real=real,
        ours=ours,
        verbalts=verbalts,
        pca_dim=pca_dim,
        perplexity=perplexity,
        seed=seed,
    )
    meta = {
        "name": spec.name,
        "real_root": str(spec.real_root),
        "ours_path": str(spec.ours_path),
        "verbalts_path": str(spec.verbalts_path),
        "real_shape": list(real.shape),
        "ours_shape": list(ours.shape),
        "verbalts_shape": list(verbalts.shape),
        "sequence_length": length,
        "num_channels": channels,
    }
    return embedding, meta


def scatter_pair(ax: plt.Axes, real: np.ndarray, generated: np.ndarray, *, method: str, color: str) -> None:
    ax.scatter(
        real[:, 0],
        real[:, 1],
        s=7,
        c=GROUND_TRUTH_COLOR,
        alpha=0.20,
        marker="o",
        linewidths=0,
        label="Ground Truth",
        rasterized=True,
    )
    ax.scatter(
        generated[:, 0],
        generated[:, 1],
        s=12,
        c=color,
        alpha=0.72,
        marker="^",
        linewidths=0,
        label=method,
        rasterized=True,
    )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(True, color="#B8B8B8", linewidth=0.7, alpha=0.75)
    ax.tick_params(labelsize=8, width=0.8, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#303030")


def draw_figure(embeddings: list[dict[str, np.ndarray]], dataset_names: list[str], output: Path, *, dpi: int, save_pdf: bool) -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), constrained_layout=False)
    col_titles = ["Ours", "VerbalTS"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12, fontweight="bold", pad=7)

    for i, (embedding, name) in enumerate(zip(embeddings, dataset_names)):
        scatter_pair(axes[i, 0], embedding["real"], embedding["ours"], method="Ours", color=OURS_COLOR)
        scatter_pair(axes[i, 1], embedding["real"], embedding["verbalts"], method="VerbalTS", color=VERBALTS_COLOR)
        axes[i, 0].set_ylabel(name, fontsize=11, fontweight="bold")
        for j in range(2):
            axes[i, j].legend(
                loc="upper left",
                frameon=True,
                framealpha=0.86,
                facecolor="white",
                edgecolor="#D9D9D9",
                fontsize=8,
                handletextpad=0.3,
                borderpad=0.3,
            )

    fig.subplots_adjust(left=0.09, right=0.985, top=0.91, bottom=0.09, wspace=0.16, hspace=0.23)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    if save_pdf:
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    specs = [
        DatasetSpec("Synth-M", args.synthm_real_root, args.synthm_ours, args.synthm_verbalts),
        DatasetSpec("Health-US", args.health_real_root, args.health_ours, args.health_verbalts),
    ]
    embeddings: list[dict[str, np.ndarray]] = []
    metadata: list[dict[str, object]] = []
    for idx, spec in enumerate(specs):
        embedding, meta = process_dataset(
            spec,
            real_split=args.real_split,
            max_real=args.max_real,
            max_gen=args.max_gen,
            aggregate=args.generated_aggregate,
            pca_dim=args.pca_dim,
            perplexity=args.perplexity,
            seed=args.seed + idx,
        )
        embeddings.append(embedding)
        metadata.append(meta)
        print(f"[embedded] {spec.name}: real={meta['real_shape']} ours={meta['ours_shape']} verbalts={meta['verbalts_shape']}")

    draw_figure(
        embeddings,
        [spec.name for spec in specs],
        args.output,
        dpi=args.dpi,
        save_pdf=not args.no_pdf,
    )
    meta_path = args.output.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {args.output}")
    if not args.no_pdf:
        print(f"[saved] {args.output.with_suffix('.pdf')}")
    print(f"[saved] {meta_path}")


if __name__ == "__main__":
    main()
