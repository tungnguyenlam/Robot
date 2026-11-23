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

## 2026-11-09 - Month 1 data pipeline: egograph package and dataset viewer
- Added `src/egograph`: adapters for HOI4D, EPIC-100, and Ego4D, parsing into egoformat-v1
  (`docs/DATA_FORMAT.md`). Also: frame-exact PyAV reader, K-aware transforms, torch `EgoClipDataset`,
  `egograph export`, and a `PoseExtractor` interface for the Month 2 baselines. Layout: `docs/ARCHITECTURE.md`.
- amdserver: extracted `HOI4D_release.zip` (2,973 videos), cloned EPIC-100 annotations into
  `data/epic_kitchens/annotations`, and created the `egograph` conda env (a clone of stablehand: torch 2.14 + ROCm).
- Indexes in `data/_index/`: HOI4D 2,973 seqs / 16.5 h (1,685 with MANO hands; all have obj pose,
  camera, and actions); EPIC 156 videos / 22.2 h (P01–P05). Ego4D not downloaded.
- HOI4D conventions measured, not assumed. Objects use intrinsic XYZ Euler and (l,w,h) dims, with
  projected-box IoU 0.945 vs 0.71 for the next-best convention. MANO uses flat_hand_mean=True and the
  original MANO_LEFT (no smplx shapedirs fix): reprojection 0.7 px right and 0.85 px left, vs 29–32 px otherwise.
- Some HOI4D objpose files are zero-padded (`00000.json`); the adapter handles both naming styles.
- Viewer: `scripts/viewer.sh start` serves http://100.121.118.8:8765 (Tailscale only). It shows video with
  GT hand keypoints, MANO→2D, 3D/2D object boxes, the action timeline, the camera trajectory, and a standard-clip preview.
- Tests: 11 pass on amdserver (`pytest -q`). A mixed batch of 4 HOI4D+EPIC clips at 1080p decode takes about 12 s
  (first batch). For training, `egograph export` downsized clips first.
- Next: download Ego4D (needs LAB22_API_KEY); then Month 2 baseline wrappers in `egograph/extractors/`.
## 2026-11-10 - Commit and push all changes
- Committed all staged changes (47 files) with message "chore: commit all changes".
- Pushed to origin/main after rebasing to incorporate remote updates.
- Report: runs/commit-push-2026-11-10/report.md

## 2026-11-16 - All core-paper modalities in the loader; Ego4D in index + viewer
- Schema egoformat-v1.1 (v1 still loads): `Clip.depth/seg/audio/scene`, `Annotations.gaze_2d/imu`,
  `HandTrack.bbox_2d`, `SequenceMeta.modalities`. `EgoLoader(modalities=[...])` loads on request;
  `EgoClipDataset` emits fixed-shape keys with `*_valid` (missing -> zeros) so mixed batches collate.
  Config: `data.modalities` (default [rgb]); example `configs/experiment/all_modalities.yaml`.
- Sources: HOI4D depth (FFV1 16-bit mm AVI, pixel-aligned), 2D masks (VOC ids), 3Dseg scene cloud
  (verified in the T_world_cam frame: median |depth - projected z| 1-3.5 cm); EPIC audio + GoPro
  GPMF IMU (new `gpmf.py`, mapped to OpenCV camera frame); Ego4D audio, IMU CSV (device frame),
  gaze (v flipped; checked by eye), narrations, FHO actions + hand/object boxes on key frames.
- Contact proxy (`contact.py`): hand joints to object box < 1 cm; calibrated on 40 rigid HOI4D seqs
  (carry/pick/put 98-99%, grasp 92%, reach 26%, rest 0%).
- Fixes: HOI4D `_load_actions` KeyError on 17 `markResult` files; Ego4D adapter root ("ego4d" vs
  "Ego4D") was why Ego4D never indexed. Ego4D: 278 videos indexed, v2 annotations (6 GB) + IMU/gaze
  downloaded into data/Ego4D. HOI4D depth extraction still running (log data/HOI4D/extract_depth.log);
  HOI4D auto-reindexes when it prints DONE.
- CLI: `egograph prepare` (EPIC IMU cache), `egograph modalities`, `export --modalities`.
  Viewer: Ego4D tab, modality badges/filters, hand boxes, gaze, IMU plot, depth/mask overlays.
- Tests: 21 pass (new tests/test_modalities.py + 3 real-data tests). Docs: docs/MODALITIES.md.
- Gemini (3 T3 threads) failed to summarize the papers (sandbox denied / stuck on approval);
  the paper analysis was done directly.

## 2026-11-17 - Modalities scoped to TOPIC_PLAN; HOI4D CAD + Ego4D FHO-only download
- Decision: real labels only, no pseudo-labelling (Depth-Anything/WiLoR/SAM2) to fill gaps.
- Plan-relevant: RGB, intrinsics, camera traj, 3D hands, object 6-DoF + meshes, contact, action text.
  Secondary: IMU/gravity, 2D boxes, depth, masks. Not used: audio, gaze (off by default). See docs/MODALITIES.md.
- Ego4D modality counts are low by design (IMU 23%, gaze 31 videos of 9821); our 278 match that.
- HOI4D depth extraction done: depth 2973/2973 after reindex.
- HOI4D CAD models added to data/HOI4D/download.py (1.5 GB zip, 2581 obj) and extracted to
  data/HOI4D/HOI4D_CAD_Model_for_release/. Loader does not read meshes yet.
- New scripts/ego4d_select_fho.py -> configs/data/ego4d_fho_uids.txt: FHO videos with boxes, ranked by
  boxed actions/GB, 150 GB budget = 586 videos, 64k/106k boxed actions (full set would be 1.24 TB).
  Downloading with lab22 --parts full_scale --uid-file (log data/Ego4D/fho_download.log); then checksum
  verify + egograph index --dataset ego4d.

## 2026-11-17 - viewer: label annotation flags, disable unavailable overlays
- Trigger: the 4 unlabeled sidebar dots were misread (P01_02's yellow dot = actions, not pose); EPIC-KITCHENS has actions only.
- Sidebar dots → lettered flags H/O/C/A with a legend; sequence badges coloured to match.
- Overlay toggles with no data for the sequence are now disabled and struck through. An "only action labels" note is shown, the empty pose lane is hidden from the timeline, and the empty trajectory SVG is collapsed.
- Fixed a stale `S.trajXY` that carried over between sequences. Files: `src/egograph/viewer/static/{app.js,index.html,style.css}`.

## 2026-11-17 - Uncommitted work wiped by a macmini sync; recovered
- 00:37: `scripts/sync_to_server.sh` run from macmini (rsync --delete-after, .git excluded) replaced
  amdserver's working tree with macmini's, deleting all uncommitted 11-16/11-17 work. uv.lock lost too.
- Recovered by replaying every Write/Edit/heredoc edit from the Claude transcript onto HEAD (40 ops, all
  asserts passed), then 3-way merged with macmini's viewer edits (KINDS legend, updateLayers) and its
  WORKLOG change. Backup outside the repo: ~/egograph-wip-backup-20261117/.
- pytest: 20 pass, 1 fail: test_hoi4d_depth_seg_scene now picks a different first sequence (depth is
  complete) where scene cloud vs depth is 14 cm off; to investigate.
- HOI4D CAD meshes: rigid meshes are centred at the object origin with extents = box (l,w,h) in metres;
  articulated parts use objs/new-k-align.obj per part. Loader + viewer integration not started.

## 2026-11-17 - HOI4D CAD meshes in loader, tensors and viewer; scene misregistration found
- New `mesh` modality (HOI4D, 2967/2973 seqs): `Clip.meshes` {object_id: ObjectMesh}, saved as
  `meshes/<oid>.npz`; tensors `object_points (O,P,6)` + `object_points_valid`, config `mesh_points: 1024`.
  `egograph/data/mesh.py` parses each .obj once into a cache (~3k-face draw mesh + 8192 surface
  samples with normals); `egograph prepare --dataset hoi4d` built all 1140 (217 MB).
- Rigid: `rigid/<Cat>/<NNN>.obj`, already in the objpose frame (extents = box dims to mm).
  Articulated: `-align` part meshes paired to part labels by assignment on box-size gap + name
  keywords; median gap 0.3–2 cm per part type (lockers/trash cans 3–5 cm: loose boxes).
- Checks: rigid silhouette vs mask IoU median 0.65–0.8, identity orientation beats 180° flips;
  figures in `results/mesh_check/`. Viewer: layer "object mesh (CAD)" + `/api/mesh/{ds}` (path-guarded).
- `test_hoi4d_depth_seg_scene` failure explained: ~1 in 5 HOI4D sequences has a constant 10–60 cm
  scene/pose misregistration in the source data (not scale, not a frame offset). Test now
  requires 4 of 6 sequences to agree; documented in docs/MODALITIES.md.
- Fixed temp-file races in the mesh and EPIC IMU caches (parallel writers). Ego4D FHO
  checksum verify: 588/588 ok. Tests: 24 pass.

## 2026-11-18 - Viewer: MANO hand mesh, contact, scene cloud; Ego4D FHO contact/PNR key frames
- Audit of raw data/modalities vs viewer. Plan-relevant gaps closed; audio skipped (not in the study).
- Viewer layers: "hand mesh (MANO)" (`/api/.../handmesh`, int16 0.1 mm; one depth sort with CAD
  meshes), "contact (derived)" (same 1 cm proxy as tensors: label, pink box, timeline lane),
  "scene cloud" (HOI4D 3Dseg projected per frame, z-buffered; misregistered seqs are obvious),
  "contact / PNR frames" (Ego4D timeline ticks + banner). Chunks now carry contact dist and T_cam_world.
- New modality `keyframes` (Ego4D 587/831): FHO `contact_frame` (~half the actions) and `pnr_frame`
  → tensors `keyframe (2,T)`, `keyframe_valid`; `contact.fho_keyframes`. Ego4D reindexed.
- Decision: `fho_oscc-pnr` not loaded: its 65k state-change clips on our videos repeat fho_main PNR
  frames (98% exact); only 12k "no state change" negatives are new. sta/lta/goalstep/nlq/vq/av skipped.
- Tests: 27 pass (+keyframe mapping, Ego4D keyframes, viewer hand mesh/contact/scene endpoints).
  Screenshots: results/viewer_check/. Viewer restarted on :8765.

## 2026-11-18 - viewer: split view for depth / masks
- New "⧉ split view" button (key `v`) puts the paused depth or mask PNG in its own pane next to
  the video at full opacity, with a 0.2-2.0 m turbo colour bar for depth. With depth and masks
  both on, depth goes to the pane and masks stay overlaid on the video. Turning split on with
  neither layer active switches depth on. Still paused-only, same `/api/modality` endpoint (w=1280).
- Files: `src/egograph/viewer/static/{index.html,app.js,style.css}`. Checked in headless
  Chromium on HOI4D ZY20210800001-H1-C1-N19-S100-s02-T1 frame 120.

## 2026-11-19 - EPIC VISOR annotations downloaded; docs refreshed
- VISOR (data.bris `2v6cgv1x04ol22qp9rm9x2j6a7`) restricted to our P01–P05 videos: sparse GT JSON +
  dense interpolation zips for 62 of our 156 EPIC videos, plus `frame_mapping.json`, noun classes v2,
  README. 127 files, 4.1 GB, in `data/epic_kitchens/VISOR/` (`MANIFEST.txt` = path + bytes).
  6 more of ours are in VISOR test (no annotations); 88 are not in VISOR. RGB frame zips skipped.
- Check: all 127 sizes match, 62/62 zips pass `unzip -t`, 62 JSONs parse. Content: 15,948 frames,
  82,291 polygon masks; hand masks carry `in_contact_object` (20,925 name the touched mask), i.e.
  real EPIC contact labels.
- Not loaded yet: the EPIC adapter needs `seg` (polygons → masks) and contact, mapped through
  `frame_mapping.json` and checked against our decoded frames.
- Docs: MODALITIES.md (coverage rows, VISOR source section, viewer split view), DATASETS_GUIDE.md
  (VISOR step), ARCHITECTURE.md (mesh.py, FHO key frames, ego4d_select_fho.py), README.md
  (egoformat-v1.1, Modalities link).

## 2026-11-19 - EPIC VISOR masks + hand contact in the adapter and viewer (`seg_sparse`)
- Correction to the entry above: 53 of our 156 EPIC videos have VISOR (45 train, 8 val), not 62;
  the other 9 downloaded (P01_14, P02_12, P03_22–24, P04_24–26, P04_33) have no local video.
- Frame mapping: VISOR frame n = last video frame with timestamp < n/R, R = round(fps) (60 for
  59.94 fps, 50 for 50 fps). `visor.frame_map` uses packet timestamps → VideoReader index.
  The closed form ceil(n·fps/R)−1 fails after a timestamp gap (P05_08: off by 1 after n≈35k).
  Checked vs VISOR's own JPEGs: 52/52 sampled frames exact on P01_03, P05_08, P01_103.
  `frame_mapping.json` (→ old rgb_frames) not used. Duplicate landings deduped, manual wins.
- New: `adapters/visor.py`, `PolygonMasks` + `CONTACT_*` in schema, modality `seg_sparse` (53),
  `adapter.load_polygon_masks`, polygons Douglas–Peucker-simplified (~1.7 px @1080p, 4× fewer
  points; cv2 if installed, numpy DP was 50x slower), cached per video (662 MB) via `egograph
  prepare`: 48 min incl. 89 IMU caches that were missing. 14,214 manual + 866,904 interpolated
  frames, 18,588 hand contacts (= raw counts). Not in tensors yet.
- Viewer: layers "masks (VISOR)" + "mask names", hand→object contact on manual frames (pink),
  manual/interpolated banner, timeline lane, `,`/`.` jump to GT frames, "Masks (VISOR)" card,
  gzip on API. Busiest chunk 0.5 MB gz. Screenshots: results/viewer_check/epic_visor_*.jpg.
- Tests: 31 pass (+4 test_visor.py). EPIC reindexed; viewer restarted on :8765.

## 2026-11-19 - Rename package from egograph to robot
- Renamed `src/egograph` to `src/robot`, updated `pyproject.toml` (`name = "robot"`, CLI `robot = "robot.cli:main"`), and updated imports, tests, scripts, and docs (`AGENTS.md`, `README.md`, `docs/*.md`).
- Kept the conda environment name `egograph` (`~/miniforge3/envs/egograph`) unchanged on amdserver.
- Reinstalled editable (`pip install --no-build-isolation --no-deps -e .`); `pytest -q` passes (31 passed) and `robot status` runs cleanly.

## 2026-11-20 - Viewer video toggle; all work moves to main
- Viewer: new "▣ video" button (key `b`) hides the RGB video so overlays (pose, masks,
  depth, scene cloud) draw on black; a depth/mask image on the main stage then shows at
  full opacity. Documented in `docs/MODALITIES.md`.
- Git: `data-modalities` fast-forwarded into `main`. From now on we work only on `main`
  (rule added to `AGENTS.md`); no feature branches.

## 2026-11-21 - DataLoader-only dataset statistics notebooks + no-frame transform fix
- Added 3 DataLoader-only analysis notebooks under `src/notebooks/` using exclusively `EgoClipDataset.from_config` + `torch.utils.data.DataLoader` (no raw files, indexes, or adapters): `01_hoi4d_statistics.ipynb`, `02_epic_kitchens_statistics.ipynb`, and `03_ego4d_statistics.ipynb`. All 3 executed cleanly end-to-end with the `egograph` kernel (installed `matplotlib`, `pandas`, `seaborn`, `ipykernel`, `nbformat`, `nbclient` into `egograph`).
- Fixed a transform bug exposed by the Ego4D annotation-only DataLoader pass: when `frames`/`depth`/`seg` are `None`, `Resize` followed by `CenterCrop` computed the crop offset from the unresized `meta.(height, width)`. `transforms.py` now tracks `_grid_hw` on the `Clip` without mutating `SequenceMeta` (also respected by `clip_to_tensors` in `torch_dataset.py`), with regression test `tests/test_geometry.py::test_transforms_without_frames_match_with_frames`. `pytest -q`: 32 passed.
