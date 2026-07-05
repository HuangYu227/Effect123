import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate external generated time-series arrays with the same ConTSG/CTTP 11-metric protocol."
    )
    parser.add_argument("--generated", required=True, type=Path, help="Generated .npy, shape [N,L,C] or [N,L].")
    parser.add_argument("--data-root", required=True, type=Path, help="VerbalTS raw dataset root.")
    parser.add_argument("--contsg-root", required=True, type=Path)
    parser.add_argument("--cttp-config", required=True, type=Path)
    parser.add_argument("--cttp-checkpoint", required=True, type=Path)
    parser.add_argument("--cttp-text-encoder-model", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--caption-index", type=int, default=0)
    parser.add_argument(
        "--reshape-generated",
        default="",
        help="Optional T,C shape for flattened outputs, e.g. 36,21 to reshape [N,756,1] into [N,36,21].",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from effectcma_flow.evaluation import contsg_metrics
    from effectcma_flow.evaluation.verbalts_metrics import (
        _load_contsg_cttp,
        calculate_frechet_distance,
    )

    seed_everything(args.seed)
    device = resolve_device(args.device)

    data_root = args.data_root.expanduser().resolve()
    generated = load_generated(args.generated.expanduser().resolve(), reshape=args.reshape_generated)
    train_ts, train_caps = load_split(data_root, "train", args.caption_index)
    test_ts, test_caps = load_split(data_root, "test", args.caption_index)

    if generated.shape[1:] != test_ts.shape[1:]:
        raise ValueError(f"generated shape {generated.shape} is incompatible with test data {test_ts.shape}")
    if generated.shape[0] != test_ts.shape[0]:
        n = min(generated.shape[0], test_ts.shape[0])
        print(f"[warn] generated/test count mismatch: {generated.shape[0]} vs {test_ts.shape[0]}; using first {n}")
        generated = generated[:n]
        test_ts = test_ts[:n]
        test_caps = test_caps[:n]

    clip = _load_contsg_cttp(
        contsg_root=args.contsg_root.expanduser().resolve(),
        clip_config_path=args.cttp_config.expanduser().resolve(),
        clip_model_path=args.cttp_checkpoint.expanduser().resolve(),
        text_encoder_model=args.cttp_text_encoder_model.expanduser().resolve(),
        device=device,
    )

    with torch.no_grad():
        ref_ts_feat, ref_joint_feat, ref_raw = collect_embeddings(
            clip=clip,
            ts=train_ts,
            caps=train_caps,
            batch_size=args.batch_size,
            device=device,
            desc="ref-train",
        )
        real_ts_feat, real_text_feat, real_raw = collect_real_embeddings(
            clip=clip,
            ts=test_ts,
            caps=test_caps,
            batch_size=args.batch_size,
            device=device,
            desc="real-test",
        )
        gen_ts_feat, gen_joint_feat, cttp, gen_raw = collect_generated_embeddings(
            clip=clip,
            ts=generated,
            caps=test_caps,
            batch_size=args.batch_size,
            device=device,
            desc="gen-external",
        )

    ref_ts_mean, ref_ts_cov = mean_cov(ref_ts_feat)
    gen_ts_mean, gen_ts_cov = mean_cov(gen_ts_feat)
    ref_joint_mean, ref_joint_cov = mean_cov(ref_joint_feat)
    gen_joint_mean, gen_joint_cov = mean_cov(gen_joint_feat)

    out = {
        "contsg_fid": calculate_frechet_distance(ref_ts_mean, ref_ts_cov, gen_ts_mean, gen_ts_cov),
        "contsg_jftsd": calculate_frechet_distance(ref_joint_mean, ref_joint_cov, gen_joint_mean, gen_joint_cov),
        "contsg_cttp": float(cttp),
        "contsg_reference_count": float(ref_ts_feat.shape[0]),
        "contsg_generated_count": float(gen_ts_feat.shape[0]),
        "CTTPScore": float(cttp),
    }
    stat = contsg_metrics.compute_statistical_metrics(ref_raw, gen_raw, train_reference=ref_raw)
    stat["ACD"] = contsg_metrics.acd(real_raw, gen_raw, max_lag=50)
    out.update(stat)
    out["FID"] = contsg_metrics.frechet_distance(ref_ts_feat, gen_ts_feat)
    precision, recall = contsg_metrics.precision_recall(real_ts_feat, gen_ts_feat, device=device)
    out["Precision"] = precision
    out["Recall"] = recall
    out["J-FTSD"] = contsg_metrics.frechet_distance(ref_joint_feat, gen_joint_feat)
    jprecision, jrecall = contsg_metrics.joint_precision_recall(
        real_ts_feat,
        real_text_feat,
        gen_ts_feat,
        real_text_feat,
        device=device,
    )
    out["JointPrecision"] = jprecision
    out["JointRecall"] = jrecall

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[saved] {args.output_json}")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_generated(path: Path, *, reshape: str = "") -> np.ndarray:
    arr = np.load(path, allow_pickle=False).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim == 4:
        arr = np.median(arr, axis=1).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"generated array must have shape [N,L,C], [N,L], or [N,S,L,C], got {arr.shape}")
    if reshape:
        parts = [int(value) for value in reshape.replace("x", ",").split(",") if value.strip()]
        if len(parts) != 2:
            raise ValueError("--reshape-generated must be formatted as T,C, e.g. 36,21")
        t, c = parts
        flat = arr.reshape(arr.shape[0], -1)
        if flat.shape[1] != t * c:
            raise ValueError(f"cannot reshape generated {arr.shape} into [N,{t},{c}]")
        arr = flat.reshape(arr.shape[0], t, c)
    return arr


def load_split(root: Path, split: str, caption_index: int) -> tuple[np.ndarray, np.ndarray]:
    ts = np.load(root / f"{split}_ts.npy", allow_pickle=False).astype(np.float32)
    if ts.ndim == 2:
        ts = ts[..., None]
    caps = np.load(root / f"{split}_text_caps.npy", allow_pickle=True)
    if caps.ndim == 1:
        caps = caps[:, None]
    cap_idx = min(caption_index, caps.shape[1] - 1)
    text = np.asarray([str(value) for value in caps[:, cap_idx].tolist()], dtype=object)
    return ts, text


def iter_batches(ts: np.ndarray, caps: np.ndarray, batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for start in range(0, ts.shape[0], batch_size):
        end = min(start + batch_size, ts.shape[0])
        yield ts[start:end], caps[start:end]


def collect_embeddings(*, clip, ts: np.ndarray, caps: np.ndarray, batch_size: int, device: torch.device, desc: str):
    ts_embeddings = []
    joint_embeddings = []
    raw_series = []
    for batch_ts, batch_caps in tqdm(list(iter_batches(ts, caps, batch_size)), desc=desc, dynamic_ncols=True):
        tensor = torch.from_numpy(batch_ts).to(device).float()
        ts_len = torch.full((tensor.shape[0],), tensor.shape[1], device=device, dtype=torch.int32)
        text = [str(value) for value in batch_caps.tolist()]
        ts_emb = clip.get_ts_coemb(tensor, ts_len)
        text_emb = clip.get_text_coemb(text, None)
        ts_embeddings.append(ts_emb.cpu())
        joint_embeddings.append(torch.cat([ts_emb, text_emb], dim=-1).cpu())
        raw_series.append(tensor.cpu())
    return cat_np(ts_embeddings), cat_np(joint_embeddings), cat_np(raw_series)


def collect_real_embeddings(*, clip, ts: np.ndarray, caps: np.ndarray, batch_size: int, device: torch.device, desc: str):
    ts_embeddings = []
    text_embeddings = []
    raw_series = []
    for batch_ts, batch_caps in tqdm(list(iter_batches(ts, caps, batch_size)), desc=desc, dynamic_ncols=True):
        tensor = torch.from_numpy(batch_ts).to(device).float()
        ts_len = torch.full((tensor.shape[0],), tensor.shape[1], device=device, dtype=torch.int32)
        text = [str(value) for value in batch_caps.tolist()]
        ts_emb = clip.get_ts_coemb(tensor, ts_len)
        text_emb = clip.get_text_coemb(text, None)
        ts_embeddings.append(ts_emb.cpu())
        text_embeddings.append(text_emb.cpu())
        raw_series.append(tensor.cpu())
    return cat_np(ts_embeddings), cat_np(text_embeddings), cat_np(raw_series)


def collect_generated_embeddings(*, clip, ts: np.ndarray, caps: np.ndarray, batch_size: int, device: torch.device, desc: str):
    ts_embeddings = []
    joint_embeddings = []
    raw_series = []
    cttp_sum = 0.0
    count = 0
    for batch_ts, batch_caps in tqdm(list(iter_batches(ts, caps, batch_size)), desc=desc, dynamic_ncols=True):
        tensor = torch.from_numpy(batch_ts).to(device).float()
        ts_len = torch.full((tensor.shape[0],), tensor.shape[1], device=device, dtype=torch.int32)
        text = [str(value) for value in batch_caps.tolist()]
        ts_emb = clip.get_ts_coemb(tensor, ts_len)
        text_emb = clip.get_text_coemb(text, None)
        ts_embeddings.append(ts_emb.cpu())
        joint_embeddings.append(torch.cat([ts_emb, text_emb], dim=-1).cpu())
        raw_series.append(tensor.cpu())
        cttp_sum += (ts_emb * text_emb).sum(dim=-1).sum().item()
        count += int(ts_emb.shape[0])
    if count == 0:
        raise ValueError("No generated samples were evaluated")
    return cat_np(ts_embeddings), cat_np(joint_embeddings), cttp_sum / count, cat_np(raw_series)


def cat_np(tensors: list[torch.Tensor]) -> np.ndarray:
    if not tensors:
        raise ValueError("Cannot concatenate an empty tensor list")
    return torch.cat(tensors, dim=0).detach().cpu().numpy()


def mean_cov(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(features, axis=0), np.cov(features, rowvar=False)


if __name__ == "__main__":
    main()
