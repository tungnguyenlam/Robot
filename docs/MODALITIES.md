# Modalities: what the core papers need and what the loader gives

The loader carries every input modality used by the three Month 2 baselines in
`user-work/core-papers` (StableHand, EgoGrasp, WHOLE), as far as our datasets
(HOI4D, EPIC-KITCHENS-100, Ego4D) contain them. This page lists:

1. which modalities each paper uses,
2. which dataset provides each one, and how the source data was decoded and checked,
3. how to request them in code, and the tensor keys you get back.

The registry in code is [`src/robot/data/modalities.py`](../src/robot/data/modalities.py).
For the in-memory and on-disk layout, see [DATA_FORMAT.md](DATA_FORMAT.md).

## 1. What each paper consumes

| Modality | StableHand | EgoGrasp | WHOLE | Month 3 graph |
|---|---|---|---|---|
| RGB video | ✓ | ✓ | ✓ | ✓ |
| Depth | – | ✓ (Depth-Anything3 prediction) | – | object/scene geometry |
| Intrinsics | ✓ | ✓ (predicted) | ✓ | ✓ |
| Camera trajectory | ✓ (DROID-SLAM) | ✓ (predicted extrinsics) | ✓ (metric SLAM) | world-frame nodes |
| Gravity / IMU | ✓ gravity-aligned world frame | – | ✓ gravity-aware frame | world "up" |
| 2D masks | – | ✓ (SAM2 hands+object) | ✓ (2D object masks) | – |
| Object mesh / template | – | ✓ (SAM3D templates) | ✓ (templates) | object nodes |
| 2D hand detection / pose | ✓ (WiLoR/HaMeR) | ✓ (WiLoR) | ✓ | – |
| 3D hand pose (MANO) | output / supervision | output | output | hand nodes |
| Object 6-DoF | – | output | output | object nodes |
| Hand–object contact | – | ✓ contact loss | ✓ per-hand binary contact (VLM) | contact edges |
| Action / narration | – | – | – | conditioning |
| Audio, gaze | – | – | – | optional cues (not used by the baselines) |

The evaluation datasets of these papers (HOT3D, ARCTIC, H2O) are **not downloaded**. HOI4D is
the overlap with EgoGrasp; we use it as our 3D ground truth.

## 2. Coverage per dataset

Run `robot modalities` for live counts. As of 2026-11-18:

| Modality | HOI4D (2973) | EPIC-KITCHENS (156) | Ego4D (831 downloaded) |
|---|---|---|---|
| `rgb` | all | all | all |
| `depth` | 2973 | – | – |
| `seg` | 2972 | – (see `seg_sparse`) | – |
| `seg_sparse` (polygons, some frames) | – | 53 (VISOR: manual + interpolated) | – |
| `audio` | – (silent videos) | 156 | 541 |
| `imu` | – | 142 (GoPro, camera frame) | 67 (device frame) |
| `gaze` | – | – | 1 |
| `scene` | 2973 | – | – |
| `mesh` | 2967 (CAD; every rigid object, all articulated parts) | – | – |
| `intrinsics`, `camera_pose`, `object_pose` | all | – | – |
| `hand_pose` | 1685 | – | – |
| `hand_box`, `object_box` | object_box: all | – (hand-object detections not downloaded) | 586 (FHO key frames) |
| `contact` (derived) | 1685 | – (VISOR labels in `seg_sparse`, manual frames) | – |
| `keyframes` (contact onset, PNR) | – | – | 587 (FHO labels) |
| `actions` | all | all | 807 (narrations and/or FHO) |

HOI4D CAD models are in `data/HOI4D/HOI4D_CAD_Model_for_release/` and load as the `mesh`
modality (section 3). EPIC VISOR masks and hand contact load as `seg_sparse` (53 videos, viewer
and adapter; not in tensors yet). Still missing: EPIC hand-object detections, and Ego4D
`fho_hands` for train/val (not in v2).

### Which modalities the project needs

Judged against `user-work/TOPIC_PLAN.md`. Real labels only: we do not pseudo-label gaps.

- **Core**: RGB + timestamps; intrinsics and camera trajectory (world frame); 3D hand joints;
  object 6-DoF + boxes; object meshes (penetration loss in Month 4, simulation in Month 6);
  hand–object contact (graph edges, contact loss); action/narration text (VLA language, Month 5).
  3D labels come from HOI4D only.
- **Secondary**: gravity/IMU (world "up"), 2D hand/object boxes, depth and 2D masks (Month 2
  baseline evaluation).
- **Not used by the plan**: audio, gaze. They stay in the loader but are off by default.

Ego4D's role is in-the-wild RGB, narrations and FHO contact timing and boxes. We download only
FHO videos that have boxes: `scripts/ego4d_select_fho.py` ranks them by boxed actions per GB
and writes `configs/data/ego4d_fho_uids.txt` (150 GB budget: 586 videos, 64k of 106k boxed
actions). Fetch with
`lab22 download ego4d -o data/Ego4D --parts full_scale --uid-file configs/data/ego4d_fho_uids.txt --no-metadata`,
then `robot index --dataset ego4d`.

## 3. Sources, conventions and checks

**HOI4D depth.** `HOI4D_depth_video/<rel>/align_depth/depth_video.avi` is FFV1 `gray16le`,
1920×1080, 15 fps, in millimetres, already aligned to RGB pixels. The loader returns metres,
with 0 meaning no reading. Check: depth sampled at the 2D hand keypoints matches the 3D joint
depth to within a few cm (`tests/test_datasets.py::test_hoi4d_depth_seg_scene`).

**HOI4D masks (`seg`).** `2Dseg/mask/*.png` (palette) or `2Dseg/shift_mask/*.png` (RGB), both
using the PASCAL VOC palette. The loader converts both to VOC indices (`mask_ids`); colours
outside the palette become 255. Ids are per sequence: 1 is the main object, 2 the right hand,
3 the left hand in two-hand sequences. Articulated parts also use 3–5, so read ids together
with the sequence's object list.

**HOI4D scene cloud (`scene`).** `3Dseg/label.pcd` (~150k points, xyz rgb label). It is in
the same world frame as `T_world_cam` (from `3Dseg/output.log`). Check: projected into a view,
its points agree with the depth map: median gap 0.4–4 cm in 20 of 25 random sequences. The
other ~1 in 5 (e.g. `ZY20210800001-H1-C1-N19-S100-s02-T1`, two Lamp sequences) have a constant
10–60 cm misregistration in the source data. No other frame's pose fixes it and it is not a
scale error, so treat the scene cloud as approximate for those; per-frame depth is unaffected.

**HOI4D object meshes (`mesh`).** Rigid objects: `rigid/<Category>/<NNN>.obj`, where NNN is the
instance number in the sequence id (`N12` → `012`). Articulated objects: one
`articulated/<Category>/<NNN>/objs/new-<k>-align.obj` per part, named in `result.json`. All
meshes are in metres in the frame of `T_cam_obj`, centred on the annotated box. Rigid mesh
extents match the box `dimensions` to a few mm. Articulated parts are paired with annotated
part labels (`Laptopdisplay`, `Safedoor`, …) by `_match_parts` in the HOI4D adapter: an
assignment on box-size gap plus a name-keyword check (`PART_WORDS`). Over all ~1600 articulated
sequences, the median gap between box and part extents is 0.3–2 cm per part type. Lockers and
trash cans run 3–5 cm because their part boxes are loose, but each has one mesh per part, so
the pairing is unambiguous. Checks:
- Rigid silhouettes vs masks: IoU median 0.65–0.8 (Knife 0.4, thin).
- The identity orientation beats 180° flips; mugs and kettles show the handle on the correct side
  (`results/mesh_check/hoi4d_mesh_orientation.jpg`).
- Part overlays land on the correct parts (`results/mesh_check/hoi4d_articulated_parts.jpg`).
- Missing: 6 sequences (5 Lamp, 1 Pliers) whose instance has no CAD folder, and 10 Pliers parts
  with no `-align` file.

Each `.obj` (50k–100k triangles) is parsed once into
`$EGO_INDEX_ROOT/cache/hoi4d/meshes/*.npz` (`robot/data/mesh.py`):
- a ~3k-face simplified mesh for drawing;
- 8192 area-weighted surface points with normals from the full mesh, for penetration and contact
  losses.
Run `robot prepare --dataset hoi4d` once to build them all (~1.5 GB of CAD → small npz files).

**EPIC audio.** Stereo AAC at 48 kHz in the video file. `read_audio` resamples it (default
16 kHz mono) and aligns it to the first decoded video frame. A clip of frames
`[t0, tN]` gets the audio of `[t0, tN + one frame period)`.

**EPIC IMU.** The GoPro `gpmd` telemetry track (GPMF: ACCL and GYRO at ~200 Hz). Parsing
lives in `robot/data/gpmf.py`. Channels are reordered by `ORIN` (Hero7 "YxZ"; Hero5 has no
ORIN, so "ZXY" is assumed) and then mapped to the OpenCV camera frame. Check: over 10 videos,
the mean accelerometer is about (0, −8.3, −4.8) m/s² in OpenCV axes, i.e. gravity "up" is −y
with a ~30° head tilt. That is only consistent with the camera frame. The first read parses the whole
file (seconds per video), so run `robot prepare --dataset epic_kitchens` once to cache all
of them under `$EGO_INDEX_ROOT/cache/epic_kitchens/imu/`.

**EPIC VISOR masks (`seg_sparse`).** Manual polygon masks from
[VISOR](https://github.com/epic-kitchens/VISOR) (data.bris DOI `2v6cgv1x04ol22qp9rm9x2j6a7`,
CC BY-NC 4.0), in `data/epic_kitchens/VISOR/` (4.1 GB, `MANIFEST.txt` lists every file). 53 of
our 156 EPIC videos have them (45 VISOR train, 8 val); the other 103 are not in VISOR. The
download also holds 9 VISOR videos of P01–P05 that we have no video for (0.6 GB, unused).
- `GroundTruth-SparseAnnotations/annotations/{train,val}/<vid>.json`: manual masks on about 2
  frames per action (14,214 frames, 73,191 masks on our 53 videos). Each mask has an
  open-vocabulary `name`, an EPIC noun `class_id` (`EPIC_100_noun_classes_v2.csv` adds left/right
  hand and glove) and polygons in 1920×1080 pixels. Hand masks carry `in_contact_object`: the id
  of the touched mask (18,588 hand masks), `hand-not-in-contact`, `none-of-the-above` or
  `inconclusive`. These are real contact labels, unlike the derived HOI4D ones.
- `Interpolations-DenseAnnotations/{train,val}/<vid>_interpolations.zip`: masks interpolated
  between pairs of manual frames (polygons in 854×480), kept only where the interpolation scored
  high J&F. No contact labels. Up to 1.4 GB of JSON per video.
- **Frame numbers.** VISOR frame `n` (1-based, in the file names) was extracted by time at
  `R = round(nominal fps)` (60 for the 59.94 fps EPIC-55 videos, 50 for the 50 fps ones): it is
  the last video frame whose timestamp is before `n / R` s. `visor.frame_map` finds it from the
  packet timestamps and returns our `VideoReader` index. That is `ceil(n·fps/R) − 1` on most
  videos, but not after a timestamp gap (P05_08 has one and would be off by a frame after
  n ≈ 35k). Two VISOR frames can land on one video frame (60 fps sampling of 59.94 fps video);
  the manual one is kept. Check: VISOR's own JPEGs (the sparse `rgb_frames` zips, not kept)
  match the mapped decoded frame on 52/52 sampled frames of P01_03, P05_08 and P01_103 (mean
  abs. error 1–2 vs 5–30 for neighbours), and overlays sit on the objects late in P05_08.
  `frame_mapping.json` maps to the older `rgb_frames` release and is not used.
- **Loading.** `adapter.load_polygon_masks(meta)` returns a `PolygonMasks` (schema.py): entity
  names (stable per video; VISOR mask ids are not), sorted frame indices with a manual flag, and
  per mask its entity, contact (an entity id or a `CONTACT_*` code) and polygons (int16 pixels
  at video resolution). Polygons are simplified (Douglas–Peucker, ~1.7 px at 1080p; 4× fewer
  points; OpenCV's `approxPolyDP` when installed, else a numpy version). Built once per video into
  `$EGO_INDEX_ROOT/cache/epic_kitchens/visor/` (662 MB for 53) by `robot prepare --dataset
  epic_kitchens`: ~20 min of the 48 min run on 2026-11-19, which also built 89 missing IMU caches.
  All 53: 14,214 manual frames (18,588 hand contacts) and 866,904 interpolated frames.
  Not in `Clip` or the training tensors yet: `seg_valid` would have to become per frame.
- Viewer frame numbers come from `currentTime × meta.fps`, where `meta.fps` is EPIC's nominal
  rate. On a video with a timestamp gap that can differ from the reader index by one frame near
  the end.

**Ego4D IMU.** `v2/imu/<uid>.csv` with `canonical_timestamp_ms` on the video timeline. The axes
are the device's own frame (`ImuTrack.frame == "device"`), and its rotation to the camera is
unknown, so `gravity_cam` is zero for these clips.

**Ego4D gaze.** `v2/gaze/<uid>.csv` (Pupil Invisible), `norm_pos_x/y` with the origin at the
bottom-left: `u = x·W`, `v = (1 − y)·H`. Rows with confidence < 0.6 are dropped. Check (by eye,
3 frames of the one video with gaze): the flipped points land on the person or the cards being
looked at, and the unflipped points do not.

**Ego4D narrations and FHO.** At index time, `narration.json` (both annotation passes) and
`fho_main.json` are split into one file per video under `$EGO_INDEX_ROOT/cache/ego4d/ann/`.
Narrations become 1 s segments centred on each timestamp. FHO actions keep their critical
frames (`pre_45 … pre_frame, contact_frame, pnr_frame, post_frame`) and boxes. FHO frame
numbers count frames of the full-scale video, and boxes are xywh pixels at full resolution
(checked on two videos; an overlay on a 1440×1080 video puts both boxes on the hands). Boxes exist
only on key frames: the loader places each on the clip frame with that exact source index
and leaves all other frames NaN. Rejected FHO actions are skipped.

**Contact (derived).** Only EPIC VISOR ships contact labels, on its manual frames (`seg_sparse`,
above). For HOI4D, `robot/data/contact.py`
computes the signed distance from each hand's 21 3D joints to each object's oriented box.
`contact = distance < 1 cm`. The threshold was calibrated on 40 rigid HOI4D sequences against
their action labels: carry/pick-up/put-down 98–99% in contact, grasp 92%, reach-out 26%,
stop 32%, rest 0%. Boxes are amodal and joints lie under the skin, so this is a proxy.

**Ego4D contact timing (`keyframes`).** FHO's critical frames are real human labels: the
frame where the hand first touches the object (`contact_frame`, on about half of the actions)
and the point of no return where the object changes state (`pnr_frame`, on all of them).
`fho_keyframes` in `contact.py` puts each on the nearest clip frame within half a frame step,
as `keyframe (2,T)` [contact onset, PNR]. `keyframe_valid` is False when no action in the window
has critical frames, so an all-False row then means unknown, not "no contact".
`fho_oscc-pnr_*.json` is not loaded: on our 831 videos its 65k state-change clips repeat
`fho_main`'s PNR frames (98% exact match); only its 12k "no state change" negatives are new.

## 4. Using them

```python
from robot.data import EgoLoader
L = EgoLoader(modalities=["rgb", "depth", "seg", "audio", "imu", "gaze", "scene"])
clip = L.load_clip("hoi4d", seq_id, frame_indices)
clip.depth          # (T,H,W) metres      clip.seg   (T,H,W) uint8 ids
clip.audio          # (S,C) float32       clip.audio_sr
clip.scene          # ScenePoints(xyz, rgb, label)
clip.meshes         # {object_id: ObjectMesh(vertices, faces, points, normals, source)} ("mesh")
clip.annotations.imu      # ImuTrack(t, accel, gyro, frame); .at(timestamps) -> (T,6), valid
clip.annotations.gaze_2d  # (T,2) pixels, NaN when missing
clip.annotations.hands["right"].bbox_2d  # (T,4) xyxy (Ego4D FHO)
```

Labels (hands, objects, camera, actions, hand boxes, gaze) always load. The `modalities`
list only controls the heavy, on-request ones. A modality that a sequence lacks is simply
`None`. Transforms (`Resize`, `CenterCrop`) resize depth and masks with nearest-neighbour
sampling and move hand boxes and gaze with the image.

For training, set `data.modalities` in the config (see
[`configs/experiment/all_modalities.yaml`](../configs/experiment/all_modalities.yaml)).
Sources can be filtered with `require_modalities: [depth]`. `EgoClipDataset` then adds these
keys, always with the same shape so mixed-dataset batches collate:

| key | shape | note |
|---|---|---|
| `hand_bbox_2d`, `hand_bbox_valid` | (B,2,T,4), (B,2,T) | always present |
| `depth`, `depth_valid` | (B,T,1,H,W) metres, (B,) | |
| `seg`, `seg_valid` | (B,T,H,W) uint8, (B,) | |
| `audio`, `audio_valid` | (B,C,S), (B,) | S = clip_len / fps × sample_rate |
| `imu`, `imu_valid` | (B,T,6), (B,T) | accel then gyro, interpolated to frame times |
| `imu_in_camera_frame`, `gravity_cam` | (B,), (B,3) | unit gravity (down) in the camera frame; zero unless the IMU is in the camera frame |
| `gaze_2d`, `gaze_valid` | (B,T,2), (B,T) | |
| `scene_points`, `scene_valid` | (B,N,6), (B,) | xyz (world) + rgb/255, N = `scene_points` |
| `object_points`, `object_points_valid` | (B,O,P,6), (B,O) | surface xyz + normal in each object's frame (map with `obj_T_cam_obj`), P = `mesh_points` (1024) |
| `contact`, `contact_dist`, `contact_valid` | (B,2,O,T) | hand × object × time |
| `keyframe`, `keyframe_valid` | (B,2,T), (B,) | Ego4D FHO [contact onset, PNR] on their clip frames |

Missing values are 0 in tensors. Always mask with the `*_valid` flags.

```bash
python scripts/example_dataloader.py configs/experiment/all_modalities.yaml   # one mixed batch
robot modalities                                                           # coverage table
robot export --dataset hoi4d --modalities rgb depth seg scene --limit 1 --out exports/demo
```

## 5. In the viewer

`scripts/viewer.sh start` (http://100.121.118.8:8765). There is an Ego4D tab now. Per
sequence it shows modality badges, dashed modality filter chips in the sidebar, hand boxes
(Ego4D FHO), a gaze ring, an IMU plot with a playhead, depth or mask overlays when paused
(layer buttons "depth (paused)" / "masks (paused)"), and for HOI4D the CAD mesh of each object
or part, posed by `T_cam_obj` and shaded (layer "object mesh (CAD)", from `/api/mesh/hoi4d`).

Added 2026-11-18:
- **hand mesh (MANO)**: the 778-vertex MANO surface from the labelled parameters
  (`/api/sequence/{ds}/{sid}/handmesh`, int16 0.1 mm), painted in one depth sort with the CAD
  meshes so hands occlude objects correctly. It turns pink while the hand is in contact.
- **contact (derived)**: the same distance and 1 cm threshold as the `contact` tensors. A hand
  in contact is labelled with the object and distance, the object's box turns pink, and a pink
  lane under the timeline marks contact frames.
- **contact / PNR frames** (Ego4D): yellow (contact onset) and red (PNR) ticks on the timeline,
  and a banner within ±2 frames of each.
- **scene cloud** (HOI4D, off by default): 60k points of the 3Dseg cloud projected with the
  frame's camera pose, z-buffered, coloured by semantic label. The ~1 in 5 misregistered
  sequences (section 3) are obvious here, e.g. `ZY20210800001-H1-C1-N19-S100-s02-T1`.
Added 2026-11-19:
- **masks (VISOR)** and **mask names** (EPIC, 53 videos): filled polygons per entity (hands in
  the usual left cyan / right orange), drawn while playing on interpolated runs and on manual
  frames. On manual frames a hand's label names what it touches ("right hand ● knife") and the
  hand and touched mask get pink outlines. A banner says "manual (GT)" or "interpolated". The
  bottom timeline lane shows interpolated runs (purple) and manual frames (white ticks); keys
  `,` / `.` and the buttons in the "Masks (VISOR)" card jump between manual frames, which are
  single frames and invisible at playback speed. Sidebar chip `seg_sparse` filters to them.
- **⧉ split view** (key `v`, paused only): the depth or mask image in its own pane beside the
  video at full opacity, with a 0.2–2.0 m turbo colour bar for depth. With both layers on, depth
  goes to the pane and masks stay on the video.
