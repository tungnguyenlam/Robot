"""Small YAML config system with ``_base_`` inheritance and dotted CLI overrides.

    cfg = load_config("configs/experiment/example.yaml", ["data.clip_len=32"])

A config file may contain ``_base_: [relative/path.yaml, ...]``. The bases are merged
in order, then the file itself is merged on top (dicts merge recursively, all other
values are replaced).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    bases = raw.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    cfg: dict = {}
    for b in bases:
        cfg = _merge(cfg, _load((path.parent / b).resolve()))
    return _merge(cfg, raw)


def _parse(value: str) -> Any:
    v = yaml.safe_load(value)
    if isinstance(v, str):  # YAML 1.1 reads "1e-4" as a string
        try:
            return float(v)
        except ValueError:
            pass
    return v


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        key, _, value = ov.partition("=")
        node: Any = cfg
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node[int(p)] if isinstance(node, list) else node.setdefault(p, {})
        node[int(leaf) if isinstance(node, list) else leaf] = _parse(value)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    return apply_overrides(_load(Path(path).resolve()), overrides)
