# Robot: dexterous manipulation from egocentric video, informed by graph geometry

Roadmap: [`user-work/TOPIC_PLAN.md`](user-work/TOPIC_PLAN.md). This repository contains the
`egograph` package. Its first piece (Month 1) is a unified data pipeline that parses Ego4D,
EPIC-KITCHENS-100, and HOI4D into one standard format for 3D hand and object pose
extraction, plus a web viewer for the datasets.

## Quick start (amdserver, where the data lives)

```bash
conda activate egograph                 # torch 2.14 + ROCm; made by cloning the stablehand env
pip install -e ".[torch,web,dev]"
egograph status                         # what is downloaded and indexed
egograph index --dataset all            # build sequence indexes (about 1 min)
scripts/viewer.sh start                 # http://100.121.118.8:8765 over Tailscale
pytest -q                               # unit tests and data checks
python scripts/example_dataloader.py    # one mixed HOI4D + EPIC torch batch
```

From the Mac or thinkbook: edit locally, then run `scripts/sync_to_server.sh` and run on the server.

## Using the loader

```python
from egograph.data import EgoLoader
loader = EgoLoader()
meta = loader.index("hoi4d").filter(require=["has_hand_pose"], category_name="Mug")[0]
clip = loader.load_clip("hoi4d", meta.sequence_id, loader.frames_at(meta, start_s=2.0, num=16, fps=15))
clip.frames                              # (16, 1080, 1920, 3) uint8
clip.annotations.hands["right"].joints_3d   # (16, 21, 3) camera frame, metres
clip.annotations.objects[0].T_cam_obj       # (16, 4, 4)
clip.annotations.T_world_cam                # (16, 4, 4)
```

For training, use `egograph.data.torch_dataset.EgoClipDataset.from_config(cfg["data"])`
(see `configs/experiment/example_dataloader.yaml`).

## Docs

- [Data format (egoformat-v1)](docs/DATA_FORMAT.md): schema, conventions, on-disk layout, tensor layout
- [Architecture](docs/ARCHITECTURE.md): module map and where Months 2–6 plug in
- [Experiments](docs/EXPERIMENTS.md): configs, run directories, logging
- [Dataset acquisition](docs/DATASETS_GUIDE.md)
