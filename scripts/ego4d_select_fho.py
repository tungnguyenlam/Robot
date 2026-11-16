#!/usr/bin/env python
"""Pick the Ego4D full-scale videos to download for the hand-object part of the project.

Ego4D has no 3D labels, so the only real hand-object labels it adds are FHO's: per-action
critical frames (pre / contact / PNR / post) with hand and object boxes. This script keeps the
videos whose ``fho_main.json`` actions carry at least one box (so test-split videos, whose labels
are withheld, drop out), ranks them by boxed actions per GB and keeps the best ones until the
size budget is used. Videos already on disk count toward the budget but are always kept.

Inputs are the metadata already downloaded with lab22: ``ego4d.json``,
``v2/annotations/fho_main.json`` and lab22's saved manifest (for file sizes).

    python scripts/ego4d_select_fho.py --budget-gb 150 --out configs/data/ego4d_fho_uids.txt
    lab22 download ego4d -o data/Ego4D --parts full_scale --uid-file configs/data/ego4d_fho_uids.txt --no-metadata
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


def boxed_actions(video: dict) -> tuple[int, int]:
    """(valid actions, valid actions with at least one box on a critical frame)."""
    n = n_box = 0
    for interval in video["annotated_intervals"]:
        for a in interval["narrated_actions"]:
            if a.get("is_rejected") or a.get("is_invalid_annotation"):
                continue
            n += 1
            if any(f.get("boxes") for f in a.get("frames") or []):
                n_box += 1
    return n, n_box


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/Ego4D"))
    ap.add_argument("--budget-gb", type=float, default=150.0)
    ap.add_argument("--out", type=Path, default=Path("configs/data/ego4d_fho_uids.txt"))
    args = ap.parse_args()

    sizes = {}
    with gzip.open(args.root / ".lab22/manifest.csv.gz", "rt") as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            if row["part"] == "full_scale":
                sizes[Path(row["path"]).name.split(".")[0]] = int(row["size"] or 0)
    have = {p.stem for p in (args.root / "v2/full_scale").glob("*.mp4")}
    fho = json.loads((args.root / "v2/annotations/fho_main.json").read_text())

    rows = []
    for v in fho["videos"]:
        uid = v["video_uid"]
        n, n_box = boxed_actions(v)
        if n_box and sizes.get(uid):
            rows.append((uid, n, n_box, sizes[uid], uid in have))
    rows.sort(key=lambda r: (not r[4], -r[2] / r[3]))  # on disk first, then boxed actions per byte

    budget, used, keep = args.budget_gb * 1e9, 0, []
    for r in rows:
        if r[4] or used + r[3] <= budget:
            keep.append(r)
            used += r[3]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    head = (f"Ego4D FHO selection: scripts/ego4d_select_fho.py --budget-gb {args.budget_gb:g}\n"
            f"{len(keep)} videos, {used / 1e9:.1f} GB, {sum(r[2] for r in keep)} of "
            f"{sum(r[2] for r in rows)} boxed FHO actions ({len(rows)} candidate videos)\n")
    args.out.write_text("".join(r[0] + "\n" for r in sorted(keep)))  # lab22 rejects comment lines
    print(head.strip())
    print(f"new to download: {sum(not r[4] for r in keep)} videos, "
          f"{sum(r[3] for r in keep if not r[4]) / 1e9:.1f} GB -> {args.out}")


if __name__ == "__main__":
    main()
