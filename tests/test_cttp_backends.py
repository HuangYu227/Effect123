from __future__ import annotations

import argparse
from pathlib import Path
import sys
import types

import pytest
import torch
import yaml

from effectcma_flow.evaluation import verbalts_metrics
from scripts.eval_verbalts_metrics import _resolve_cttp_paths


class _FakeConTSGEmbedder:
    loaded_config: dict | None = None

    def __init__(
        self,
        clip_config_path,
        clip_model_path,
        device,
        use_longalign,
        normalize_embeddings,
    ) -> None:
        self.__class__.loaded_config = yaml.safe_load(
            Path(clip_config_path).read_text(encoding="utf-8")
        )
        self.model = torch.nn.Linear(1, 1)
        self.device = device
        assert Path(clip_model_path).name == "clip_model_best.pth"
        assert use_longalign is False
        assert normalize_embeddings is None

    def get_ts_embedding(self, ts, ts_len=None):
        assert ts_len is not None
        return torch.ones(ts.shape[0], 4, device=ts.device)

    def get_text_embedding(self, batch):
        return torch.full((len(batch["cap"]), 4), 2.0)


def _install_fake_contsg(monkeypatch) -> None:
    contsg = types.ModuleType("contsg")
    contsg.__path__ = []
    contsg_eval = types.ModuleType("contsg.eval")
    contsg_eval.__path__ = []
    embedder = types.ModuleType("contsg.eval.embedder")
    embedder.CLIPEmbedder = _FakeConTSGEmbedder
    monkeypatch.setitem(sys.modules, "contsg", contsg)
    monkeypatch.setitem(sys.modules, "contsg.eval", contsg_eval)
    monkeypatch.setitem(sys.modules, "contsg.eval.embedder", embedder)


def test_default_cttp_backend_preserves_verbalts_loader(monkeypatch, tmp_path):
    sentinel = object()
    called = {}

    def fake_loader(**kwargs):
        called.update(kwargs)
        return sentinel

    monkeypatch.setattr(verbalts_metrics, "_load_verbalts_cttp", fake_loader)
    computer = verbalts_metrics.VerbalTSMetricComputer(
        verbalts_root=tmp_path,
        clip_config_path=tmp_path / "model_configs.yaml",
        clip_model_path=tmp_path / "clip_model_best.pth",
        device=torch.device("cpu"),
        stats=None,
    )

    assert computer.cttp_backend == "verbalts"
    assert computer.clip is sentinel
    assert called["verbalts_root"] == tmp_path


def test_contsg_backend_uses_official_embedder_contract(monkeypatch, tmp_path):
    _install_fake_contsg(monkeypatch)
    root = tmp_path / "ConTSG-Bench"
    root.mkdir()
    clip_folder = tmp_path / "cttp"
    clip_folder.mkdir()
    config_path = clip_folder / "model_configs.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "clip_type": "clip_patchtst",
                "text": {
                    "pretrain_model_path": "${LONGCLIP_ROOT}",
                    "pretrain_model_dim": 768,
                },
                "ts": {"seq_len": 128, "n_var": 2},
            }
        ),
        encoding="utf-8",
    )
    model_path = clip_folder / "clip_model_best.pth"
    model_path.write_bytes(b"checkpoint-placeholder")
    text_model = tmp_path / "LongCLIP-GmP"
    text_model.mkdir()

    computer = verbalts_metrics.VerbalTSMetricComputer(
        verbalts_root=None,
        clip_config_path=config_path,
        clip_model_path=model_path,
        device=torch.device("cpu"),
        stats=None,
        cttp_backend="contsg",
        contsg_root=root,
        cttp_text_encoder_model=text_model,
    )
    ts_emb, text_emb = computer._embed(torch.zeros(2, 128, 2), ["a", "b"])

    assert computer.cttp_backend == "contsg"
    assert ts_emb.shape == (2, 4)
    assert text_emb.shape == (2, 4)
    assert _FakeConTSGEmbedder.loaded_config["text"]["pretrain_model_path"] == str(
        text_model.resolve()
    )
    assert not any(
        parameter.requires_grad for parameter in computer.clip.embedder.model.parameters()
    )


def test_contsg_backend_requires_explicit_text_encoder(tmp_path):
    args = argparse.Namespace(
        cttp_backend="contsg",
        contsg_root=str(tmp_path),
        cttp_text_encoder_model=None,
        clip_folder=str(tmp_path),
        clip_config=None,
        clip_model=None,
    )
    with pytest.raises(ValueError, match="cttp-text-encoder-model"):
        _resolve_cttp_paths(args)


def test_cttp_path_resolver_keeps_legacy_default(tmp_path):
    args = argparse.Namespace(
        verbalts_root=str(tmp_path / "VerbalTS"),
        clip_folder=str(tmp_path / "cttp"),
        clip_config=None,
        clip_model=None,
    )
    root, config, model = _resolve_cttp_paths(args)
    assert root == Path(args.verbalts_root)
    assert config == Path(args.clip_folder) / "model_configs.yaml"
    assert model == Path(args.clip_folder) / "clip_model_best.pth"
