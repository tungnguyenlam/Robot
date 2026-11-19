# egoformat-v1.1: the standard egocentric hand-object format

v1.1 (2026-11-16) adds optional modalities: depth, masks, audio, IMU, gaze, a scene point
cloud, and hand 2D boxes; 2026-11-17 adds object meshes (`Clip.meshes`). v1 clips still load. What each modality is, where it comes from,
and how it was checked: [MODALITIES.md](MODALITIES.md).

Every dataset adapter, pose extractor (Month 2), and model (Month 3+) reads and writes this
format. It is defined in [`src/robot/data/schema.py`](../src/robot/data/schema.py),
which is the source of truth. This page summarizes it.

## Conventions

| Item | Convention |
|---|---|
| Units | metres, seconds, radians; 2D in pixels (origin at the top-left, x right, y down) |
| Camera frame | OpenCV: x right, y down, z forward |
| Transforms | `T_a_b` is 4×4 and maps frame b to frame a. `T_world_cam` is the camera pose; `T_cam_obj` is the object pose in the camera frame |
| Pixel centres | integer coordinates are pixel centres. Resizing maps `u' = (u + 0.5)·s − 0.5`, and K is updated to match |
| Hand joints | 21 joints in OpenPose order: wrist, then thumb, index, middle, ring, pinky (base to tip). `HAND_PARENTS` gives the kinematic tree |
| MANO | 48-D axis-angle (3 global + 45 finger), no PCA, `flat_hand_mean=True`. Left-hand betas use the original `MANO_LEFT.pkl` (no smplx shapedirs fix). `robot.hands.mano.ManoModel` implements this exactly |
| Missing data | NaN, plus `valid[t] = False`. Arrays are never ragged in time |

The HOI4D conventions were checked against the data (`tests/test_datasets.py::test_hoi4d_conventions`):

* **Objects:** intrinsic `XYZ` Euler angles and dimensions `(length, width, height)` along the
  object x/y/z axes. The projected 3D box matches the annotated 2D box with mean IoU 0.945;
  the next-best convention scores 0.71.
* **Hands:** projected MANO joints are 0.7 px (right) and 0.85 px (left) from the annotated `kps2D`.
  `flat_hand_mean=False` gives 32 px, and the smplx left-shapedirs fix gives 29 px.

## In memory: `Clip`

```
Clip
├── meta: SequenceMeta      dataset, sequence_id, video_path, fps, num_frames, width, height,
│                           duration_s, intrinsics (3×3 or None), has_* flags, tags{},
│                           modalities[] (names from MODALITIES; meta.has("depth"))
├── frames: (T,H,W,3) uint8 RGB, or None
├── K: (3,3) float32, or None (uncalibrated: EPIC, Ego4D)
├── depth: (T,H,W) float32 metres on the RGB grid (0 = no reading), or None
├── seg:   (T,H,W) uint8 instance ids (0 = background), or None
├── audio: (S,C) float32 from the first frame's time, audio_sr: int, or None
├── scene: ScenePoints(xyz (N,3) world, rgb (N,3), label (N,), instance), or None
├── meshes: {object_id: ObjectMesh}, or None    each in its object's frame (apply T_cam_obj)
│     vertices (V,3), faces (F,3) simplified (~3k faces, for drawing);
│     points (8192,3), normals (8192,3) area-weighted samples of the full mesh; source (path)
└── annotations: Annotations
    ├── frame_indices (T,) int64      source frame numbers
    ├── timestamps    (T,) float64    seconds
    ├── T_world_cam   (T,4,4) or None
    ├── hands{left,right}: HandTrack
    │     valid (T,), joints_2d (T,21,2), joints_3d (T,21,3),
    │     mano_pose (T,48), mano_betas (T,10), mano_trans (T,3), bbox_2d (T,4) xyxy
    ├── objects[]: ObjectTrack
    │     object_id, label, valid (T,), T_cam_obj (T,4,4), bbox_2d (T,4) xyxy,
    │     dimensions (3,), mesh_path (relative to the dataset root, or None)
    ├── actions[]: ActionSegment(start_s, end_s, label, verb, noun, extra)
    ├── gaze_2d (T,2) pixels, or None
    └── imu: ImuTrack(t (N,), accel (N,3), gyro (N,3), frame "camera"|"device"), or None
          native rate, clip span ± 50 ms; .at(timestamps) -> ((T,6), valid)
```

## On disk: `Clip.save(dir)` / `Clip.load(dir)`

```
<out>/<dataset>/<sequence_id>/<first_frame:07d>/
├── meta.json          {"schema": "egoformat-v1.1", "source": SequenceMeta, "K", "frame_size",
│                       "frame_format", "depth", "seg", "audio_sr", "imu_frame",
│                       "objects": [...], "actions": [...]}
├── annotations.npz    frame_indices, timestamps, T_world_cam,
│                      hand_{left,right}_{valid,joints_2d,joints_3d,mano_pose,mano_betas,mano_trans,bbox_2d},
│                      obj{i}_{valid,T_cam_obj,bbox_2d,dimensions}, gaze_2d, imu_{t,accel,gyro}
├── extra.npz          audio, scene_{xyz,rgb,label,instance}      (only if present)
├── meshes/<object_id>.npz  vertices, faces, points, normals, source (only if present)
├── frames/000000.jpg …
├── depth/000000.png … 16-bit millimetres (HOI4D's native encoding)
└── seg/000000.png …   8-bit instance ids
```

Produce clips with:

```bash
robot export --dataset hoi4d --require has_hand_pose --limit 3 --clip-len 32 \
    --fps 15 --resize 512 --out exports/hoi4d_demo
robot export --dataset epic_kitchens --sampling actions --max-clips 20 --clip-len 32 \
    --fps 30 --resize 512 --out exports/epic_demo
```

## As torch batches: `EgoClipDataset`

`robot.data.torch_dataset.clip_to_tensors` gives every sample the same keys and shapes, so
the default collate works. B is the batch size, T the frame count, and O = `max_objects`:

| key | shape | note |
|---|---|---|
| `frames` | (B,T,3,H,W) float in [0,1] | |
| `K`, `K_valid` | (B,3,3), (B,) | |
| `hand_valid` | (B,2,T) | index 0 = left, 1 = right |
| `hand_joints_2d` / `hand_joints_3d` | (B,2,T,21,2) / (B,2,T,21,3) | |
| `hand_mano` | (B,2,T,61) | pose 48, then betas 10, then trans 3 |
| `obj_valid`, `obj_T_cam_obj`, `obj_bbox_2d`, `obj_dims` | (B,O,T), (B,O,T,4,4), (B,O,T,4), (B,O,3) | padded to O |
| `cam_valid`, `T_world_cam` | (B,T), (B,T,4,4) | |
| `hand_bbox_2d`, `hand_bbox_valid` | (B,2,T,4), (B,2,T) | Ego4D FHO key frames |
| `dataset`, `sequence_id`, `action` | lists of str | |

With `data.modalities` set, the batch also has `depth`, `seg`, `audio`, `imu`, `gravity_cam`,
`gaze_2d`, `scene_points`, `object_points`, `contact*` and `keyframe`, each with a `*_valid` flag. The table is in
[MODALITIES.md](MODALITIES.md#4-using-them). `frames` is present only when `rgb` is requested.

NaNs are replaced with 0 in tensors, so always mask with the `*_valid` flags.
