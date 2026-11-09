## 2026-11-08 - Consolidate dataset downloads on amdserver
- User authorized transferring all helper dataset files to amdserver, continuing P04 there,
  stopping helper download jobs, and deleting helper leftovers after verification.
- Stopped macmini EPIC tmux job and confirmed aria2/Python/caffeinate exited; thinkbook already stopped.
- Source inventory: thinkbook P02/P03 (84 verified videos, ~97 GiB); macmini P04_01–03
  verified plus resumable P04_04 and its .aria2 control file (~14 GiB allocated).
- Preserve resumable state and logs; independently hash destination videos before source deletion.
- Existing epic-sync trusts remote .verified entries even after failed rsync; bypass it for this migration.
- Scope remains the assigned P01–P05 training-video selection, with remaining P04 downloads on amdserver.

## 2026-11-08 - Persistent migration and cleanup pipelines launched
- Pipelines run in `epic_migration` tmux sessions on macmini and thinkbook; stopped all old copy jobs.
- Direct amdserver↔thinkbook Tailscale SSH stalled; thinkbook→amdserver public IPv6 works.
  Mac transfer now jumps through thinkbook to that IPv6 endpoint, bypassing the slow relay.
- Source snapshots: macmini 15,977,586,055 bytes including partial P04_04; thinkbook 84 videos.
- Each pipeline retries transfer/verification failures, validates destination MD5, merges only
  validated cache entries, archives helper tools, then removes unchanged source files/tools.
- P04 alone starts on amdserver after its migration check, avoiding P02/P03 transfer collisions.
- Server `epic_migration_audit` records cleanup/download completion in `/hdd/Robot/migration-20261108/`.
- Verifier checks passed: valid data accepted; old cache preserved; duplicates avoided; corruption rejected.
- CURRENT_MACHINE remains gitignored; corrected AGENTS.md wording to match shared `.gitignore`.
- Bulk transfers still running; cleanup and P04 restart are queued, not yet complete.
- Details and recovery: [migration report](results/migration-20261108/README.md).

## 2026-11-08 - Resume P04 downloading on macmini at user request
- No downloader was active on either macmini or amdserver; Mac network work was outbound rsync.
- Paused macmini migration before restarting downloads, avoiding writes during immutable inventory transfer.
- Archived the old manifest; rebuild it after all 43 P04 train videos pass official MD5.
- Keep one downloader with 16 connections on macmini, then automatically resume migration/verification/cleanup.
- Thinkbook migration remains independent; amdserver remains canonical storage.
- Started `epic_download`: resuming P04_04 with one aria2 process and 16 connections.
- Measured Wi-Fi receive 1.84 MB/s (14.7 Mbps), send 0.08 MB/s over 10 seconds after restart.
- `epic_download_then_migrate` checks downloader completion before rebuilding inventory; shell syntax passed.
- Recovery and revised sequencing: [migration report](results/migration-20261108/README.md).

## 2026-11-08 - Split remaining P04 downloads across macmini and amdserver
- User requested remaining P04 downloads start on amdserver now; server downloader was idle.
- Mac P04_04 was 93% complete, so retain it on Mac; assign the other 39 videos exclusively to amdserver.
- Stopped the old Mac full-P04 downloader and completion watcher before launching disjoint assignments.
- Added an assignment wrapper around the existing official-MD5 downloader, preserving its resume/retry behavior.
- Mac finishes P04_04, then inventories and migrates P04_01–04; server downloads exclude these four.
- Server download log stays outside the migration data tree to avoid an rsync/log collision.
- Confirmed amdserver tmux `epic_download` started P04_05 with 16 connections; Mac resumed only P04_04.
- Verified assignment wrapper Python syntax, shell syntax, and `git diff --check`; one aria2 per node.
- Server incoming traffic measured 0.23 MB/s across physical interfaces over 8s (includes other transfers).

## 2026-11-08 - Resume ThinkBook migration after reboot
- User asked to continue ThinkBook work; no tmux sessions or downloader/migration processes survived reboot.
- Its assigned P02/P03 download status is already 84/84 verified; no replacement downloads needed.
- Direct ThinkBook→amdserver SSH works; resumed existing migration in tmux `epic_migration`.
- Confirmed rsync writing `.P02_108.MP4.*` on amdserver; independent destination MD5 and source cleanup remain queued.
- P04 assignments on Mac and amdserver are unchanged; keep ThinkBook awake and online until cleanup completes.

## 2026-11-08 - Mac mini evacuation completed
- Server copies of all P04 source files and helper tools passed independent MD5 checks.
- Resumed remaining P04 downloads on amdserver; removed verified local dataset/tool files.
- Audit and manifests: `/hdd/Robot/migration-20261108/macmini/`.
