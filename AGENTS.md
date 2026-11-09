## Machine

- Read `CURRENT_MACHINE.md` first. It's a per-machine file, untracked via
  the shared `.gitignore`, that gives this machine's hardware, its role, and the fleet layout
  (macmini, thinkbook, amdserver). Datasets live on amdserver.

## Worklog

- Append decisions and results to the bottom of `WORKLOG.md` with `cat >> WORKLOG.md`
  (oldest entry first). Start each entry with `## YYYY-MM-DD - title`, keep it under
  ~20 lines, and put long detail in a report under the run directory (`runs/<id>/`) or
  `results/` that the entry links to.
- Log every decision that changes code, defaults, or the experiment plan, and every
  experiment result (run id, config delta, WER, what it rules in/out).
- Never read `WORKLOG.md` or any file over ~300 lines in full. Read the newest entries
  from the bottom up:

```bash
grep -n '^## ' WORKLOG.md | tail -15                                  # index of recent entries
tac WORKLOG.md | awk '/^## /{if(++n==2){print; exit}} {print}' | tac  # last 2 entries in full
sed -n '120,150p' WORKLOG.md                                         # one entry by line range
grep -n -i 'projector' WORKLOG.md docs/worklog_archive/*.md | tail   # older history by keyword
```

## Code

- Package `egograph` (`src/egograph`); layout in `docs/ARCHITECTURE.md`, data schema in
  `docs/DATA_FORMAT.md`. Data, GPU, and the `egograph` conda env are on amdserver: edit
  anywhere, push with `scripts/sync_to_server.sh`, then run `pytest -q` there.
- Dataset raw layouts are known only to `src/egograph/data/adapters/`; everything else uses
  `Clip`/`Annotations`. Rebuild indexes (`egograph index`) after data changes.

