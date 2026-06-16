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
| `poc/llm_description.py` | Pose pipeline + local VLM (qwen2.5vl:7b) narrating operator actions |

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

### `poc/llm_description.py`

Same RF-DETR + pose/hand/face pipeline as above, extended with a local vision-language model that narrates what the operator is doing — a step toward automatic motion/element labeling instead of manual notes.

**How it works:**

- A background thread periodically grabs the latest raw camera frame, downsamples it to 640px wide (keeps inference fast on 8 GB VRAM cards like an RTX 3060 Ti), and sends it to a local [Ollama](https://ollama.com) server running `qwen2.5vl:7b`
- The first available frame is described immediately on startup; afterward it repeats on a configurable interval
- Runs in its own thread so the LLM call (slow, ~1–3s+ on a 7B local model) never blocks the video loop or drops the displayed FPS
- The prompt asks the model to focus on **hands, tools, objects, assembly operations, and sequence of actions** — the same categories a manual time-study analyst would record

**Output:**

- Every description is printed to the terminal as `[LLM] <description>`
- Optionally overlaid on the video as a movie-style subtitle (semi-transparent bar, centered, word-wrapped)

**Prerequisites:**

```bash
ollama pull qwen2.5vl:7b
pip install ollama
```

**Usage:**

```bash
# Webcam, description printed to terminal every 3s (default)
python poc/llm_description.py

# Also show the description as an on-screen subtitle
python poc/llm_description.py --show-description

# Slower cadence to reduce GPU load
python poc/llm_description.py --llm-interval 5

# Read from file, save annotated output, no LLM (detection only)
python poc/llm_description.py --file input.mp4 --output result.mp4 --no-llm
```

| Argument | Default | Purpose |
|---|---|---|
| `--file` | webcam | Path to input video file |
| `--output` | none | Save annotated output video |
| `--llm-interval` | `3.0` | Seconds between scene descriptions |
| `--llm-model` | `qwen2.5vl:7b` | Ollama model name |
| `--no-llm` | off | Disable the LLM thread entirely |
| `--show-description` | off | Overlay description as a video subtitle (always printed to terminal regardless) |

> With RF-DETR + 3 MediaPipe models + a 7B VLM sharing one GPU, VRAM is tight. If you hit OOM, raise `--llm-interval` first — it's the easiest knob to throttle.

---

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:** `opencv-python`, `mediapipe`, `rfdetr`, `supervision`, `Pillow`, `torch`, `ollama`

> RF-DETR weights (~355 MB) are downloaded automatically on first run to `~/.roboflow/models/`.
> MediaPipe `.task` files (~25–30 MB each) are downloaded to `poc/` on first run.
> `llm_description.py` additionally requires a local [Ollama](https://ollama.com) server with `qwen2.5vl:7b` pulled (`ollama pull qwen2.5vl:7b`).

---

## Roadmap

- [x] Hand landmark detection (MediaPipe)
- [x] Object / tool detection (RF-DETR)
- [x] Body pose estimation (MediaPipe PoseLandmarker)
- [x] Face landmark detection (MediaPipe FaceLandmarker)
- [x] Local VLM scene narration (qwen2.5vl:7b via Ollama) as a manual-notes substitute
- [ ] Integrate RTMPose for higher-accuracy body keypoints
- [ ] Multi-person tracking across frames
- [ ] Motion segmentation (idle vs. active element)
- [ ] Cycle-time extraction and MTM mapping
