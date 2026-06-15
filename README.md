# Motion Detection for Time Study

Reduce manual stopwatch-based time studies and support MTM (Methods-Time Measurement) automation through computer-vision-driven motion analysis.

---

## Goal

Traditional time studies require an analyst to stand on the shop floor with a stopwatch, manually recording each motion element. This project aims to replace that process with a camera-based pipeline that:

- Detects hands and tools in the scene
- Tracks body keypoints throughout each work cycle
- Segments motions automatically
- Estimates cycle times without human intervention

---

## Target Pipeline

```
Camera feed
    │
    ▼
RF-DETR ─────────── hand & tool detection (bounding boxes, class labels)
    │
    ▼
RTMPose ─────────── body keypoints (33-point skeleton per person)
    │
    ▼
Tracking + motion segmentation ── assign IDs across frames, split cycles
    │
    ▼
Automatic cycle-time estimation ── output MTM-ready time data
```

---

## Current Phase: POC

Validating individual detection components before integrating them into the full pipeline.

### POC scripts

| File | Purpose |
|---|---|
| `poc/hand_detect.py` | MediaPipe Hand Landmarker standalone — 21-point hand skeleton on webcam |
| `poc/object_and_hand.py` | RF-DETR object detection + MediaPipe hand landmarks combined |
| `poc/pose_estimation_pipeline.py` | Full multi-model pipeline: RF-DETR + body pose + hands + face |

---

### `poc/hand_detect.py`

Baseline hand tracking using MediaPipe Tasks API.

- Detects up to 2 hands, draws 21 landmarks and skeleton connections
- Tries MediaPipe GPU delegate; falls back to CPU automatically
- Displays live FPS on screen
- Press `q` to quit

```bash
python poc/hand_detect.py
```

---

### `poc/object_and_hand.py`

Combines RF-DETR (COCO object detection) with MediaPipe hand landmarks in a single frame loop.

- RF-DETR runs on CUDA if available, CPU otherwise
- Annotates detected objects with bounding boxes and confidence scores (80 COCO classes)
- Overlays hand skeleton on top of object annotations
- Supports reading from a video file via `--file`
- Displays FPS and active device on screen
- Press `q` to quit

```bash
# Webcam
python poc/object_and_hand.py

# Video file
python poc/object_and_hand.py --file path/to/video.mp4
```

---

### `poc/pose_estimation_pipeline.py`

The most complete POC — runs all four detectors in a single loop.

| Detector | Model | Output | Color |
|---|---|---|---|
| RF-DETR | `rf-detr-base.pth` (COCO) | Bounding boxes + labels | supervision default |
| Body pose | `pose_landmarker_full.task` | 33-point skeleton | Orange |
| Hands | `hand_landmarker.task` | 21-point skeleton × 2 hands | Green |
| Face | `face_landmarker.task` | 478-point mesh (oval, eyes, lips) | Yellow |

- Auto-downloads all MediaPipe `.task` model files on first run into `poc/`
- CUDA auto-selected for RF-DETR; MediaPipe tries GPU delegate per task with CPU fallback
- Optional video output via `--output`
- Press `q` to quit

```bash
# Webcam only
python poc/pose_estimation_pipeline.py

# Read from file, save annotated output
python poc/pose_estimation_pipeline.py --file input.mp4 --output result.mp4
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:** `opencv-python`, `mediapipe`, `rfdetr`, `supervision`, `Pillow`, `torch`

> RF-DETR weights (~355 MB) are downloaded automatically on first run to `~/.roboflow/models/`.
> MediaPipe `.task` files (~25–30 MB each) are downloaded to `poc/` on first run.

---

## Roadmap

- [x] Hand landmark detection (MediaPipe)
- [x] Object / tool detection (RF-DETR)
- [x] Body pose estimation (MediaPipe PoseLandmarker)
- [x] Face landmark detection (MediaPipe FaceLandmarker)
- [ ] Integrate RTMPose for higher-accuracy body keypoints
- [ ] Multi-person tracking across frames
- [ ] Motion segmentation (idle vs. active element)
- [ ] Cycle-time extraction and MTM mapping
