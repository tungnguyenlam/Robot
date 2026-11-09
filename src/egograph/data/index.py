"""Cached sequence index. Each dataset gets one JSONL file of ``SequenceMeta`` records.

Scanning the raw data (walking ~1.8M HOI4D annotation files, probing videos) is slow, so
we run it once and store the result:

    egograph index --dataset hoi4d      # writes $EGO_INDEX_ROOT/hoi4d.jsonl

The loader and the viewer then read only the index.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from egograph.data.adapters import get_adapter
from egograph.data.schema import SequenceMeta
from egograph.paths import index_root


def index_path(dataset: str) -> Path:
    return index_root() / f"{dataset}.jsonl"


def build_index(dataset: str, log: Callable[[str], None] = print, **adapter_kwargs: Any) -> Path:
    adapter = get_adapter(dataset, **adapter_kwargs)
    if not adapter.available():
        raise FileNotFoundError(f"{dataset}: no data at {adapter.root}")
    out = index_path(dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    t0, n = time.time(), 0
    with tmp.open("w") as f:
        for meta in adapter.scan():
            f.write(json.dumps(meta.to_dict(), ensure_ascii=False) + "\n")
            n += 1
            if n % 500 == 0:
                log(f"{dataset}: {n} sequences ({time.time() - t0:.0f}s)")
    tmp.replace(out)  # atomic, so readers never see a half-written index
    log(f"{dataset}: indexed {n} sequences in {time.time() - t0:.0f}s -> {out}")
    return out


class SequenceIndex:
    """In-memory view of one dataset's index, with simple filtering."""

    def __init__(self, dataset: str, metas: Iterable[SequenceMeta]):
        self.dataset = dataset
        self.metas = list(metas)
        self._by_id = {m.sequence_id: m for m in self.metas}

    @classmethod
    def load(cls, dataset: str, path: Path | None = None) -> "SequenceIndex":
        p = path or index_path(dataset)
        if not p.exists():
            raise FileNotFoundError(f"no index for {dataset} at {p}; run `egograph index --dataset {dataset}`")
        with p.open() as f:
            return cls(dataset, (SequenceMeta.from_dict(json.loads(line)) for line in f if line.strip()))

    def __len__(self) -> int:
        return len(self.metas)

    def __getitem__(self, sequence_id: str) -> SequenceMeta:
        return self._by_id[sequence_id]

    def __contains__(self, sequence_id: str) -> bool:
        return sequence_id in self._by_id

    def filter(self, query: str | None = None, require: Iterable[str] = (), **tags: Any) -> list[SequenceMeta]:
        """Keep sequences whose id contains ``query``, whose boolean flags in ``require``
        (for example "has_hand_pose") are true, and whose tags equal ``tags``. Values may be lists."""
        out = []
        q = (query or "").lower()
        for m in self.metas:
            if q and q not in m.sequence_id.lower() and q not in json.dumps(m.tags, ensure_ascii=False).lower():
                continue
            if any(not getattr(m, r) for r in require):
                continue
            ok = True
            for k, v in tags.items():
                allowed = v if isinstance(v, (list, tuple, set)) else [v]
                if m.tags.get(k) not in allowed:
                    ok = False
                    break
            if ok:
                out.append(m)
        return out
