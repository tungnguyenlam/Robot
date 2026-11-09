# Codebase architecture

```
Robot/
├── pyproject.toml               package "egograph" (pip install -e ".[torch,web,dev]")
├── configs/
│   ├── data/base.yaml           shared data defaults (clip_len, fps, transforms)
│   └── experiment/*.yaml        one file per experiment; inherits with `_base_`
├── src/egograph/
│   ├── paths.py                 EGO_DATA_ROOT / EGO_INDEX_ROOT / EGO_MANO_DIR / EGO_RUNS_ROOT
│   ├── config.py                YAML loading, `_base_` inheritance, `a.b.0.c=value` overrides
│   ├── cli.py                   `egograph status | index | export | viewer`
│   ├── data/
│   │   ├── schema.py            egoformat-v1 (SequenceMeta, Clip, Annotations, HandTrack, ObjectTrack)
│   │   ├── adapters/            one file per dataset (the only code that knows raw layouts)
│   │   │   ├── base.py          DatasetAdapter interface and registry
│   │   │   ├── hoi4d.py         3D GT: MANO hands, 6-DoF objects, camera trajectory, actions
│   │   │   ├── epic_kitchens.py videos and narrated action segments
│   │   │   └── ego4d.py         videos (and FHO later)
│   │   ├── index.py             cached JSONL sequence index per dataset
│   │   ├── video.py             frame-exact PyAV reader
│   │   ├── geometry.py          projection, transforms, intrinsics resize/crop
│   │   ├── transforms.py        resize/crop that keep K and 2D labels consistent
│   │   ├── loader.py            EgoLoader + sample_windows (sliding / action-centred)
│   │   └── torch_dataset.py     EgoClipDataset and the fixed-shape tensor layout
│   ├── hands/mano.py            MANO forward kinematics to 21 OpenPose joints
│   ├── extractors/base.py       PoseExtractor interface (Month 2 baselines plug in here)
│   ├── utils/run.py             runs/<timestamp>-<name>/ with a config and git snapshot
│   └── viewer/                  FastAPI app and static frontend (no build step)
├── scripts/                     example_dataloader.py, viewer.sh, sync_to_server.sh
├── tests/                       unit tests run anywhere; dataset tests skip without data
├── docs/                        DATA_FORMAT.md, DATASETS_GUIDE.md, this file
├── runs/                        experiment outputs (heavy files gitignored)
└── data -> /hdd/Robot/data      raw datasets (on amdserver; gitignored)
```

## Data flow

```
raw files ──adapter.scan()──► index (JSONL) ──sample_windows()──► ClipWindow list
                                                                      │
             adapter.load_annotations() + VideoReader.read() ◄────────┘
                                   │
                             Clip (egoformat-v1) ──transforms──► Clip
                                   │
        ┌──────────────────────────┼───────────────────────────┐
   Clip.save() (export)    clip_to_tensors() (training)    viewer overlays
```

## Where the later months plug in

| Month | Work | Where |
|---|---|---|
| 2 | StableHand / EgoGrasp / WHOLE inference | `egograph/extractors/<name>.py` subclasses `PoseExtractor`. Each baseline keeps its own conda env and repo; the wrapper calls it and converts the output to `Annotations`. Predictions are cached with `Clip.save` under `runs/<run>/predictions/` |
| 2 | failure profiling (occlusion, penetration, jitter) | `egograph/eval/` metrics on `Annotations` vs HOI4D GT; the viewer overlays predictions next to GT |
| 3 | spatio-temporal hand-object graph | `egograph/graph/` builds nodes from `HandTrack.joints_3d` (edges from `HAND_PARENTS`) and `ObjectTrack` boxes/keypoints, with contact and proximity edges |
| 4 | graph-conditioned diffusion / flow matching | `egograph/models/`, `egograph/losses/`, trained from `configs/experiment/*.yaml` with `utils/run.py` |
| 5 | retargeting and VLA distillation | `egograph/retarget/`, `egograph/policy/` |
| 6 | simulation (Isaac Lab / SAPIEN) | `egograph/sim/` |

Each new module consumes `Clip`/`Annotations` rather than raw files. To add a dataset (for
example ARCTIC, HOT3D, or DexYCB), write a single adapter.
