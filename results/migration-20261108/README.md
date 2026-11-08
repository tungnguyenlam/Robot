# Dataset evacuation — 2026-11-08

Status at setup: **transfers running**, source deletion and P04 restart pending.

## What runs

- macmini tmux `epic_migration`: `migrate.py macmini` under caffeinate.
- thinkbook tmux `epic_migration`: `~/.local/share/epic-migration-20261108.py thinkbook`.
- amdserver tmux `epic_migration_audit`: `/hdd/Robot/migration-20261108/finalize.py`.
- Linux `epic_migration_awake` sessions hold idle inhibitors until migration/download completion.
  These do not override explicit suspend or closing the laptop lid.

The server is canonical at `/hdd/Robot/data`. The downloader itself already lives on amdserver.
The transfers preserve the current P01–P05 train selection; they do not expand to the full dataset.

Each helper copies dataset files and archives its complete downloader directory, excluding its
`.verified` cache from the server dataset destination. Completed video digests come from each
source cache and are checked against official MD5 metadata. Other files, including partial
P04_04 and its aria2 control file, receive a source MD5 snapshot. The server independently
hashes every copied file. Only validated video entries enter the server cache. The helper
then removes inventoried files only if every source size and modification time is unchanged.
Tools and their bin symlinks are removed after the archive passes validation. Small migration
state/log files remain for auditing and recovery. Other project/user files are untouched.

After P04 migration validation, the server downloader changes its assigned participants to
P04 and starts its existing retry-until-verified loop. This avoids touching P02/P03 while their
transfer runs. Expected final selection: 156 videos (29 P01/P05, 84 P02/P03, 43 P04).

## Status and audit

On each helper:

```
cat ~/.local/share/epic-migration-20261108-<node>/status
 tail -20 ~/.local/share/epic-migration-20261108-<node>/run.log
```

On amdserver:

```
cat /hdd/Robot/migration-20261108/status
cat /hdd/Robot/migration-20261108/download-status
```

Per-source server directories contain `manifest.json`, `source.verified`, archived `tools/`,
`VERIFIED`, and finally `SOURCE_CLEANED`. Server-wide `MIGRATION_COMPLETE` means both helpers
were cleaned; `DOWNLOADS_COMPLETE` means all 156 selected videos are verified. Neither marker
existed at setup. The finalizer updates amdserver's machine note as these milestones happen.

## Recovery

Copy and destination-verification failures retain sources and retry every 120 seconds.
Unexpected source modification stops the job before deletion, leaving the source for inspection.
The migration survives disconnecting the agent, but not rebooting a helper; relaunch the same
command from its repo in tmux to resume from its saved state. Keep the helpers online until
`MIGRATION_COMPLETE`. The existing server downloader handles download retries in tmux.

The original `epic-sync` was bypassed because it merges remote verification entries even after
rsync errors. Its original script and amdserver node.env are backed up in the audit directory.
No dataset has been deleted merely because a transfer was started.

Checks: destination verifier accepted a valid fixture, preserved old cache entries, avoided
duplicates across repeated verification, and rejected corruption. Python compiled successfully;
`git diff --check` passed. Actual destination hashes are checked by the running pipelines.

## Update: macmini downloading again (2026-11-08 12:46 local)

At the user's request, paused the Mac's migration and resumed its single P04 downloader
(16 connections), starting with the existing P04_04 partial. amdserver has no downloader
active while this source is changing. Thinkbook's separate migration is unaffected.
The old Mac manifest is retained as `manifest.before-download-resume.json` in its state directory.

Tmux `epic_download_then_migrate` runs `download-then-migrate.sh` under caffeinate.
After `epic_download` exits, it checks all 43 P04 files are verified before rebuilding the
migration inventory, copying to amdserver, independently hashing there, and cleaning the Mac.
If downloading is manually stopped before completion, migration stays paused with sources intact.
Both Mac sessions survive terminal disconnects, but need restarting after a reboot.

## Update: split P04 downloads (2026-11-08 12:56 local)

Supersedes the previous full-P04 Mac assignment. Mac finishes only P04_04 in tmux
`epic_download_then_migrate`, then migrates its P04_01–04 files with independent
server MD5 verification and source cleanup. The old completion watcher was stopped.

amdserver tmux `epic_download` now runs `/hdd/Robot/migration-20261108/download-assigned.py`
with `--exclude P04_01,P04_02,P04_03,P04_04`, assigning it the other 39 train videos.
Both assignments retain the existing downloader's 16 connections, resumability, retry
loop, and official MD5 checks. Server log: `/hdd/Robot/migration-20261108/p04-server-download.log`.
This log is outside the source transfer tree. Do not start a second server downloader.
If rebooted, restart the same assignment command in tmux rather than the full-P04 launcher
until the Mac migration completes.

Confirmed server aria2 downloading P04_05 and Mac aria2 resuming P04_04, one process each.

## Update: ThinkBook transfer resumed (2026-11-08 13:12 local)

ThinkBook rebooted around 12:50 and lost its previous tmux migration. P02/P03 remain
84/84 locally verified. Relaunched the existing direct-SSH migration in tmux `epic_migration`;
confirmed rsync writing P02_108 on amdserver. No additional dataset downloads were started
on ThinkBook because its assigned downloads are complete. Destination MD5 and cleanup
continue automatically after copying. Keep the machine awake and online; tmux does not survive reboot.
