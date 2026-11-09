"""``egograph`` command line.

    egograph status                                   # what is downloaded and indexed
    egograph index --dataset all                      # (re)build the sequence indexes
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


def cmd_export(args: argparse.Namespace) -> None:
    from egograph.data.loader import EgoLoader, sample_windows
    from egograph.data.transforms import build_transforms

    loader = EgoLoader(transform=build_transforms(
        {"resize": args.resize, "center_crop": args.center_crop}))
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
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("viewer")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(fn=cmd_viewer)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
