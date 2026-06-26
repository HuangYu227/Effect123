from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate released VerbalTS text2ts checkpoints with a ConTSG CTTP checkpoint."
    )
    parser.add_argument("--verbalts-root", required=True, type=Path)
    parser.add_argument("--contsg-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path, help="VerbalTS released run folder, e.g. .../text2ts_msmdiffmv/0.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to RUN_DIR/ckpts/model_best_loss.pth.")
    parser.add_argument("--diff-config", type=Path, default=None)
    parser.add_argument("--cond-config", type=Path, default=None)
    parser.add_argument("--cttp-config", required=True, type=Path)
    parser.add_argument("--cttp-checkpoint", required=True, type=Path)
    parser.add_argument("--cttp-text-encoder-model", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--text-output-type", default="all")
    parser.add_argument("--text-pos-emb", default="none")
    parser.add_argument("--diff-stage-num", type=int, default=3)
    parser.add_argument("--base-patch", type=int, default=None)
    parser.add_argument("--multipatch-num", type=int, default=None)
    parser.add_argument("--l-patch-len", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    verbalts_root = args.verbalts_root.expanduser().resolve()
    contsg_root = args.contsg_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = (args.checkpoint or run_dir / "ckpts" / "model_best_loss.pth").expanduser().resolve()
    diff_config = (
        args.diff_config
        or verbalts_root / "configs" / "synth-m" / "diff" / "model_text2ts_dep.yaml"
    ).expanduser().resolve()
    cond_config = (
        args.cond_config
        or verbalts_root / "configs" / "synth-m" / "cond" / "text_msmdiffmv.yaml"
    ).expanduser().resolve()
    cttp_config = args.cttp_config.expanduser().resolve()
    cttp_checkpoint = args.cttp_checkpoint.expanduser().resolve()
    cttp_text_encoder = args.cttp_text_encoder_model.expanduser().resolve()

    for path in (
        verbalts_root,
        contsg_root,
        data_root,
        checkpoint,
        diff_config,
        cond_config,
        cttp_config,
        cttp_checkpoint,
        cttp_text_encoder,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(verbalts_root))

    from data import GenerationDataset
    from effectcma_flow.evaluation import contsg_metrics
    from effectcma_flow.evaluation.verbalts_metrics import (
        _load_contsg_cttp,
        calculate_frechet_distance,
    )
    from models.conditional_generator import ConditionalGenerator

    seed_everything(args.seed)
    device = resolve_device(args.device)

    diff_cfg = read_yaml(diff_config)
    cond_cfg = read_yaml(cond_config)
    configure_verbalts_model(
        diff_cfg,
        cond_cfg,
        device=device,
        longclip_root=cttp_text_encoder,
        text_output_type=args.text_output_type,
        text_pos_emb=args.text_pos_emb,
        diff_stage_num=args.diff_stage_num,
        base_patch=args.base_patch,
        multipatch_num=args.multipatch_num,
        l_patch_len=args.l_patch_len,
    )

    dataset = GenerationDataset({"name": "custom", "folder": str(data_root)})
    train_loader = dataset.get_loader(
        "train",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        include_self=False,
    )
    test_loader = dataset.get_loader(
        "test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        include_self=False,
    )

    model = ConditionalGenerator(diff_cfg, cond_cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    clip = _load_contsg_cttp(
        contsg_root=contsg_root,
        clip_config_path=cttp_config,
        clip_model_path=cttp_checkpoint,
        text_encoder_model=cttp_text_encoder,
        device=device,
    )

    metrics = evaluate(
        model=model,
        clip=clip,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        n_samples=args.n_samples,
        sampler=args.sampler,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[saved] {args.output_json}")


def configure_verbalts_model(
    diff_cfg: dict[str, Any],
    cond_cfg: dict[str, Any],
    *,
    device: torch.device,
    longclip_root: Path,
    text_output_type: str,
    text_pos_emb: str,
    diff_stage_num: int,
    base_patch: int | None,
    multipatch_num: int | None,
    l_patch_len: int | None,
) -> None:
    diff_cfg["device"] = str(device)
    diff_cfg["generator_pretrain_path"] = ""
    diffusion = diff_cfg.setdefault("diffusion", {})
    if base_patch is not None:
        diffusion["base_patch"] = int(base_patch)
    else:
        diffusion.setdefault("base_patch", 1)
    if multipatch_num is not None:
        diffusion["multipatch_num"] = int(multipatch_num)
    if l_patch_len is not None:
        diffusion["L_patch_len"] = int(l_patch_len)

    cond_cfg["device"] = str(device)
    cond_cfg["cond_modal"] = "text"
    text_cfg = cond_cfg.setdefault("text", {})
    text_cfg["device"] = str(device)
    text_cfg["pretrain_model_path"] = str(longclip_root)
    text_cfg["tokenizer_path"] = str(longclip_root)
    text_cfg["output_type"] = text_output_type
    text_cfg["num_stages"] = int(diff_stage_num)
    text_cfg["pos_emb"] = text_pos_emb


@torch.no_grad()
def evaluate(
    *,
    model,
    clip,
    train_loader,
    test_loader,
    device: torch.device,
    n_samples: int,
    sampler: str,
) -> dict[str, float]:
    from effectcma_flow.evaluation import contsg_metrics
    from effectcma_flow.evaluation.verbalts_metrics import calculate_frechet_distance

    ref_ts_feat, ref_joint_feat, ref_raw = collect_reference_embeddings(
        clip=clip,
        loader=train_loader,
        device=device,
    )
    gen_ts_feat, gen_joint_feat, cttp, gen_raw, real_ts_feat, eval_text_feat, real_raw = collect_generated_embeddings(
        model=model,
        clip=clip,
        loader=test_loader,
        device=device,
        n_samples=n_samples,
        sampler=sampler,
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
        eval_text_feat,
        gen_ts_feat,
        eval_text_feat,
        device=device,
    )
    out["JointPrecision"] = jprecision
    out["JointRecall"] = jrecall
    return out


@torch.no_grad()
def collect_reference_embeddings(*, clip, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts_embeddings = []
    joint_embeddings = []
    raw_series = []
    for batch in tqdm(loader, desc="ref-train", dynamic_ncols=True):
        ts = batch["ts"].to(device).float()
        ts_len = batch["ts_len"].to(device).int()
        text = [str(value) for value in batch["cap"]]
        ts_emb = clip.get_ts_coemb(ts, ts_len)
        text_emb = clip.get_text_coemb(text, None)
        ts_embeddings.append(ts_emb.cpu())
        joint_embeddings.append(torch.cat([ts_emb, text_emb], dim=-1).cpu())
        raw_series.append(ts.cpu())
    return cat_np(ts_embeddings), cat_np(joint_embeddings), cat_np(raw_series)


@torch.no_grad()
def collect_generated_embeddings(
    *,
    model,
    clip,
    loader,
    device: torch.device,
    n_samples: int,
    sampler: str,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ts_embeddings = []
    joint_embeddings = []
    raw_series = []
    real_ts_embeddings = []
    text_embeddings = []
    real_raw_series = []
    cttp_sum = 0.0
    count = 0
    for batch in tqdm(loader, desc="gen-test", dynamic_ncols=True):
        ts = batch["ts"].to(device).float()
        ts_len = batch["ts_len"].to(device).int()
        text = [str(value) for value in batch["cap"]]

        multi_preds = model.generate(batch, int(n_samples), sampler=sampler)
        multi_preds = multi_preds.permute(0, 1, 3, 2)
        pred = multi_preds.median(dim=0).values.float()

        ts_gen_emb = clip.get_ts_coemb(pred, ts_len)
        text_emb = clip.get_text_coemb(text, None)
        ts_real_emb = clip.get_ts_coemb(ts, ts_len)

        ts_embeddings.append(ts_gen_emb.cpu())
        joint_embeddings.append(torch.cat([ts_gen_emb, text_emb], dim=-1).cpu())
        raw_series.append(pred.cpu())
        real_ts_embeddings.append(ts_real_emb.cpu())
        text_embeddings.append(text_emb.cpu())
        real_raw_series.append(ts.cpu())
        cttp_sum += (ts_gen_emb * text_emb).sum(dim=-1).sum().item()
        count += int(ts_gen_emb.shape[0])
    if count == 0:
        raise ValueError("No generated samples were evaluated")
    return (
        cat_np(ts_embeddings),
        cat_np(joint_embeddings),
        cttp_sum / count,
        cat_np(raw_series),
        cat_np(real_ts_embeddings),
        cat_np(text_embeddings),
        cat_np(real_raw_series),
    )


def cat_np(tensors: list[torch.Tensor]) -> np.ndarray:
    if not tensors:
        raise ValueError("Cannot concatenate an empty tensor list")
    return torch.cat(tensors, dim=0).detach().cpu().numpy()


def mean_cov(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(features, axis=0), np.cov(features, rowvar=False)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


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


if __name__ == "__main__":
    main()
