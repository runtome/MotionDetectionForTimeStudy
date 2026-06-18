# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Reduce manual stopwatch-based time studies by replacing them with a camera-based pipeline that detects hands/tools, tracks body keypoints, segments motions, and outputs MTM-ready cycle-time data automatically. Currently in POC phase — validating individual detection components.

## Running the POC scripts

All scripts are run from the repo root with `python poc/<script>.py`.

```bash
# Webcam input (default for all scripts)
python poc/hand_detect.py
python poc/object_and_hand.py
python poc/pose_estimation_pipeline.py
python poc/llm_description.py

# Video file input
python poc/llm_description.py --file input.mp4

# Save annotated output video
python poc/pose_estimation_pipeline.py --file input.mp4 --output result.mp4

# llm_description.py specific flags
python poc/llm_description.py --show-description     # overlay subtitles on video
python poc/llm_description.py --no-llm               # skip VLM, detection only
python poc/llm_description.py --llm-interval 5       # throttle VLM cadence (seconds)
python poc/llm_description.py --llm-model qwen2.5vl:7b
```

Press `q` in the OpenCV window to quit any script.

## Setup

```bash
pip install -r requirements.txt
# For llm_description.py only:
ollama pull qwen2.5vl:7b
# For action_recognition.py: pytorchvideo is in requirements.txt;
# X3D-S weights (~200 MB) and Kinetics-400 class names JSON auto-download on first run.
```

RF-DETR weights (~355 MB) auto-download to `~/.roboflow/models/` on first run.  
MediaPipe `.task` files (~25–30 MB each) auto-download into `poc/` on first run.

## Architecture

### `action_recognition.py` — temporal action classification

- **RF-DETR** detects persons per frame; highest-confidence person is the "primary subject"
- Primary person is cropped with 15% padding and added to a rolling frame buffer every 5th frame (subsampling webcam to ~6 fps to match X3D-S training cadence)
- **X3D-S** (via `torch.hub.load("facebookresearch/pytorchvideo", "x3d_s", pretrained=True)`) classifies the 13-frame clip → Kinetics-400 top-k action label
- Inference runs on the main thread (X3D-S is ~5–20 ms on GPU; acceptable for POC)
- Cyan bounding box highlights the primary person whose crops feed the action model
- `poc/kinetics400_classnames.json` auto-downloads on first run; if that fails, indices are shown instead

### Detection stack (per frame, in order)

1. **RF-DETR** (`rfdetr.RFDETRBase`) — COCO object detection. Returns 1-indexed class IDs (1–90 with gaps), so `COCO_CLASSES` is a dict, not a list. Always call `.optimize_for_inference()` after construction.
2. **MediaPipe PoseLandmarker** — 33-point body skeleton (orange).
3. **MediaPipe HandLandmarker** — 21-point hand skeleton × 2 (green).
4. **MediaPipe FaceLandmarker** — 478-point face mesh, drawn as oval + eyes + lips subsets (yellow).

All MediaPipe tasks use the new **Tasks API** (`mediapipe.tasks.python.vision`) — the old `mp.solutions.*` namespace was removed in MediaPipe 0.10 and must not be used. Each task is created with `RunningMode.IMAGE` and tries GPU delegate first, falls back to CPU silently via `_try_gpu()`.

### LLM narration thread (`llm_description.py` only)

- Background `llm_worker` thread calls Ollama (`qwen2.5vl:7b`) every `--llm-interval` seconds.
- First available frame is described immediately (polls every 50 ms until a frame exists).
- Frame is downsampled to ≤640 px wide before sending (`LLM_IMAGE_MAX_WIDTH`).
- Thread writes to `desc_state` dict under `desc_lock`; main loop reads it without blocking.
- Shutdown: `stop_event.set()` + `llm_thread.join(timeout=2)` in the `finally` block.

### Subtitle chunking

`split_description()` splits the LLM response on numbered-list markers first, then sentence boundaries. `compute_chunk_durations()` allocates display time proportionally to character length, with a `MIN_CHUNK_SECONDS = 0.8` floor. `current_chunk()` picks the active chunk from elapsed time and holds the last chunk instead of going blank if the next LLM call is slow.

### GPU handling

- RF-DETR: `device = "cuda" if torch.cuda.is_available() else "cpu"`.
- MediaPipe: per-task GPU delegate attempt; exception → CPU fallback.
- Development target: RTX 3060 Ti (8 GB VRAM). With RF-DETR + 3 MediaPipe models + 7B VLM all on one GPU, VRAM is tight — `--llm-interval` is the main throttle knob.

## Key invariants

- `*.mp4` files are git-ignored.
- COCO class IDs from RF-DETR are 1-indexed with gaps (no class 12, 26, etc.) — always use `.get(cid, f'cls{cid}')` for safe label lookup.
- MediaPipe landmark connections (HAND_CONNECTIONS, POSE_CONNECTIONS, FACE_CONNECTIONS) are defined manually in each script because the Tasks API does not expose them as constants.
- `pose_estimation_pipeline.py` and `llm_description.py` share the same detection loop structure; `llm_description.py` is the authoritative version with VLM added.

## LLM prompts

Two prompts exist in `llm_description.py`. The **detailed prompt** (`LLM_PROMPT`) describes hands, tools, objects, assembly operations, and sequence — currently active. A **short manufacturing-label prompt** (2–5 words like "Assemble motor", "Inspection") is commented out below it for switching to time-study logging mode.
