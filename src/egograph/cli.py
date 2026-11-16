"""``egograph`` command line.

    egograph status                                   # what is downloaded and indexed
    egograph index --dataset all                      # (re)build the sequence indexes
    egograph prepare --dataset epic_kitchens          # pre-parse slow modalities (EPIC GoPro IMU)
    egograph modalities                               # per-dataset modality coverage
    egograph export --dataset hoi4d --limit 2 --out exports/demo --clip-len 32 --resize 512
    egograph viewer --host 0.0.0.0 --port 8765        # dataset web viewer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from egograph.data.adapters import ADAPTERS, get_adapter
from egograph.data.index import SequenceIndex, build_index, index_path


def cmd_status(args: argparse.Namespace) -> None:
    for name in ADAPTERS:
        st = get_adapter(name).status()
        ip = index_path(name)
        st["indexed_sequences"] = len(SequenceIndex.load(name)) if ip.exists() else None
        print(f"== {name}\n{json.dumps(st, indent=2, ensure_ascii=False)}")


def cmd_index(args: argparse.Namespace) -> None:
    names = list(ADAPTERS) if args.dataset == "all" else [args.dataset]
    for name in names:
        ad = get_adapter(name)
        if not ad.available():
            print(f"{name}: skipped (no data at {ad.root})")
            continue
        build_index(name)


def cmd_prepare(args: argparse.Namespace) -> None:
    """Fill per-sequence caches that are slow to build on first access (EPIC IMU parsing reads
    the whole video file). Safe to interrupt and rerun: finished sequences are skipped."""
    from concurrent.futures import ThreadPoolExecutor

    ad = get_adapter(args.dataset)
    metas = [m for m in SequenceIndex.load(args.dataset).metas if "imu" in m.modalities]

    def one(m):
        t = ad.load_imu(m)
        return m.sequence_id, (len(t.t) if t is not None else 0)

    with ThreadPoolExecutor(args.workers) as pool:
        for k, (sid, n) in enumerate(pool.map(one, metas)):
            print(f"[{k + 1}/{len(metas)}] {sid}: {n} IMU samples", flush=True)


def cmd_modalities(args: argparse.Namespace) -> None:
    from collections import Counter

    from egograph.data.modalities import MODALITIES

    names = list(MODALITIES)
    print(f"{'modality':12s}" + "".join(f"{n:>15s}" for n in ADAPTERS))
    counts, totals = {}, {}
    for name in ADAPTERS:
        ip = index_path(name)
        ms = SequenceIndex.load(name).metas if ip.exists() else []
        counts[name] = Counter(x for m in ms for x in m.modalities)
        totals[name] = len(ms)
    for mod in names:
        print(f"{mod:12s}" + "".join(f"{counts[n][mod]:>9d}/{totals[n]:<5d}" for n in ADAPTERS))


def cmd_export(args: argparse.Namespace) -> None:
    from egograph.data.loader import EgoLoader, sample_windows
    from egograph.data.transforms import build_transforms

    loader = EgoLoader(transform=build_transforms(
        {"resize": args.resize, "center_crop": args.center_crop}), modalities=args.modalities)
    idx = loader.index(args.dataset)
    metas = idx.filter(query=args.query, require=args.require)[: args.limit]
    windows = sample_windows(metas, args.clip_len, fps=args.fps, stride_s=args.stride_s,
                             mode=args.sampling, adapters={args.dataset: loader.adapter(args.dataset)})
    if args.max_clips:
        windows = windows[: args.max_clips]
    out = Path(args.out)
    for k, w in enumerate(windows):
        clip = loader.load_window(w)
        d = clip.save(out / w.dataset / w.sequence_id / f"{w.frame_indices[0]:07d}")
        print(f"[{k + 1}/{len(windows)}] {d}")


def cmd_viewer(args: argparse.Namespace) -> None:
    import uvicorn

    from egograph.viewer.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="egograph")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(fn=cmd_status)

    s = sub.add_parser("index")
    s.add_argument("--dataset", default="all", choices=["all", *ADAPTERS])
    s.set_defaults(fn=cmd_index)

    s = sub.add_parser("export", help="write clips in the standard on-disk format")
    s.add_argument("--dataset", required=True, choices=list(ADAPTERS))
    s.add_argument("--out", required=True)
    s.add_argument("--query", default=None, help="substring filter on sequence id or tags")
    s.add_argument("--require", nargs="*", default=[], help="flags, e.g. has_hand_pose has_object_pose")
    s.add_argument("--limit", type=int, default=1, help="max sequences")
    s.add_argument("--max-clips", type=int, default=0)
    s.add_argument("--clip-len", type=int, default=32)
    s.add_argument("--fps", type=float, default=None, help="resample rate (default: native)")
    s.add_argument("--stride-s", type=float, default=None)
    s.add_argument("--sampling", choices=["sliding", "actions"], default="sliding")
    s.add_argument("--resize", type=int, default=None, help="short side in pixels")
    s.add_argument("--center-crop", type=int, nargs=2, default=None, metavar=("W", "H"))
    s.add_argument("--modalities", nargs="*", default=["rgb"],
                   help="on-request modalities to export, e.g. rgb depth seg audio imu gaze scene")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("prepare", help="pre-build slow per-sequence caches (EPIC GoPro IMU)")
    s.add_argument("--dataset", default="epic_kitchens", choices=list(ADAPTERS))
    s.add_argument("--workers", type=int, default=2, help="parallel files (the data disk is an HDD)")
    s.set_defaults(fn=cmd_prepare)

    sub.add_parser("modalities", help="which modalities each indexed dataset has").set_defaults(fn=cmd_modalities)

    s = sub.add_parser("viewer")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(fn=cmd_viewer)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
