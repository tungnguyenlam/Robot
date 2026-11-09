# Running experiments

1. **Config.** Copy `configs/experiment/example_dataloader.yaml` to
   `configs/experiment/<name>.yaml` and inherit shared defaults with `_base_: ../data/base.yaml`.
   Override on the command line with dotted keys, for example `data.clip_len=32` or
   `data.sources.0.max_sequences=10`.
2. **Run directory.** Entry points call
   `egograph.utils.run.create_run_dir(name, cfg)`. This creates `runs/<YYYYMMDD-HHMMSS>-<name>/`
   with `config.yaml` (fully resolved) and `git.json` (commit and dirty flag).
3. **Outputs** go into that directory. `report.md`, `metrics.json`, and the config are tracked
   in git. Checkpoints, frames, `*.pt`, `*.npz`, and `*.mp4` are gitignored.
4. **Log** every result in `WORKLOG.md`: run id, config change, metrics, and what it rules in
   or out. Link to `runs/<id>/report.md`.

## Where to run

| Job | Machine |
|---|---|
| data indexing, viewer, export, tests | amdserver (`~/miniforge3/envs/egograph`) |
| training up to 16 GB VRAM (RX 9060 XT, ROCm) | amdserver |
| larger training | Colab or Kaggle (see the `colab-gpu` and `kaggle-gpu` skills); export clips first with `egograph export` |
| editing code | any machine; push to the server with `scripts/sync_to_server.sh` or with git |

## Reproducibility checklist

- Fix seeds through the config (`seed:`) and record them in the run config.
- Build the window list deterministically: `sample_windows` is deterministic, so shuffle only in the DataLoader.
- Rebuild indexes (`egograph index`) whenever data changes, and note it in the WORKLOG.
