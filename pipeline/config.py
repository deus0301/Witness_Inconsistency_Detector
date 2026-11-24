"""Configuration loading, seeding and model-path resolution shared by every module."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
SAMPLE_CONFIG = REPO_ROOT / "configs" / "sample.yaml"


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (in place) and return it."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    path: str | Path | None = None,
    sample: bool = False,
    overrides: dict | None = None,
) -> dict[str, Any]:
    """Load ``configs/default.yaml``, then layer the sample config and overrides on top."""
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if sample:
        with open(SAMPLE_CONFIG, "r", encoding="utf-8") as fh:
            deep_update(cfg, yaml.safe_load(fh) or {})
        cfg["sample_mode"] = True
        # keep sample artefacts apart from full-run artefacts
        cfg["paths"]["output_dir"] = "outputs/sample"
        cfg["paths"]["figures_dir"] = "outputs/sample/figures"
    else:
        cfg["sample_mode"] = False
    if overrides:
        deep_update(cfg, copy.deepcopy(overrides))
    if cfg.get("finetune", {}).get("use_finetuned"):
        apply_finetuned(cfg)
    return cfg


OPERATING_POINT_FILE = "operating_point.json"
FINETUNE_META_FILE = "finetune_meta.json"


def finetuned_model_name(cfg: dict, checkpoint: str) -> str:
    """Name of a fine-tuned checkpoint as ``resolve_model`` expects it.

    Relative to ``paths.models_dir`` when the checkpoint lives under it (for example
    ``finetuned/nli-deberta-v3-base-pheme-all``), so ``resolve_model`` finds it unchanged
    and ``slug`` gives it a cache file of its own.
    """
    path = resolve_path(cfg, "finetuned_dir") / checkpoint
    try:
        return path.relative_to(resolve_path(cfg, "models_dir")).as_posix()
    except ValueError:
        return str(path)


def read_operating_point(model_dir: str | Path) -> dict | None:
    path = Path(model_dir) / OPERATING_POINT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def apply_finetuned(cfg: dict) -> dict:
    """Point ``models.nli`` at the configured fine-tuned checkpoint and use its saved
    direction rule and threshold (mode ``finetune.operating_mode``).

    If the checkpoint is missing, warn and keep the zero-shot model and settings.
    """
    ft = cfg["finetune"]
    name = finetuned_model_name(cfg, ft["checkpoint"])
    local = resolve_model(name, cfg)
    point = read_operating_point(local) if Path(local).is_dir() else None
    if point is None:
        print(f"[config] fine-tuned NLI checkpoint '{name}' not found (or has no "
              f"{OPERATING_POINT_FILE}); using the zero-shot model {cfg['models']['nli']}",
              file=sys.stderr, flush=True)
        cfg["finetune"]["active"] = False
        return cfg
    chosen = point["modes"][ft["operating_mode"]]
    cfg["models"]["nli"] = name
    cfg["nli"]["aggregation"] = chosen["aggregation"]
    cfg["disagreement"]["contradiction_threshold"] = float(chosen["threshold"])
    cfg["finetune"]["active"] = True
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """Return ``cfg['paths'][key]`` as an absolute path anchored at the repo root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_model(name: str, cfg: dict) -> str:
    """Use a local ``models/<name>`` snapshot when present, else the HuggingFace name.

    Both ``models/cross-encoder/nli-deberta-v3-base`` and the flat
    ``models/cross-encoder__nli-deberta-v3-base`` layouts are accepted.
    """
    models_dir = resolve_path(cfg, "models_dir")
    for candidate in (models_dir / name, models_dir / name.replace("/", "__"),
                      models_dir / name.split("/")[-1]):
        if candidate.is_dir() and any(candidate.iterdir()):
            return str(candidate)
    return name


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional for the mock path
        pass


def pick_device(cfg: dict | None = None) -> str:
    """``cuda`` when a usable GPU exists and ``cfg['device']`` allows it, else ``cpu``."""
    wanted = (cfg or {}).get("device", "auto")
    if wanted == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.cuda.get_device_name(0)  # raises if the device is hidden / unusable
            return "cuda"
    except Exception:  # noqa: BLE001 - any CUDA problem means CPU
        pass
    return "cpu"


def slug(name: str) -> str:
    """Filesystem-safe version of a model name."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def text_hash(*parts: str) -> str:
    """Stable sha1 over several strings (used as cache key)."""
    h = hashlib.sha1()
    for part in parts:
        h.update(part.encode("utf-8", errors="replace"))
        h.update(b"\x1f")
    return h.hexdigest()


def fix_console_encoding() -> None:
    """Make printing tweets safe on Windows consoles that are not UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
