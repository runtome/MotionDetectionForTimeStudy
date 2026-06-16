# Requires a local Ollama server with the vision model pulled:
#   ollama pull qwen2.5vl:7b
#   pip install ollama
import argparse
import re
import textwrap
import threading
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
from PIL import Image
from rfdetr import RFDETRBase
import supervision as sv
import torch
import ollama

# RF-DETR uses 1-indexed COCO IDs (1-90) with gaps
COCO_CLASSES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup",
    48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
    53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot",
    58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
    70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock",
    86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}

# --- GPU detection ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Torch device: {device}")

# ── Model paths & downloads ──────────────────────────────────────────────────
_POC_DIR = os.path.dirname(__file__)

MODELS = {
    "hand": (
        os.path.join(_POC_DIR, "hand_landmarker.task"),
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    ),
    "pose": (
        os.path.join(_POC_DIR, "pose_landmarker_full.task"),
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    ),
    "face": (
        os.path.join(_POC_DIR, "face_landmarker.task"),
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    ),
}

for name, (path, url) in MODELS.items():
    if not os.path.exists(path):
        print(f"Downloading {name} model...")
        urllib.request.urlretrieve(url, path)
        print(f"  -> {path}")

HAND_MODEL, POSE_MODEL, FACE_MODEL = (MODELS[k][0] for k in ("hand", "pose", "face"))

# ── Skeleton / mesh connections ──────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),  # head
    (9,10),                                              # mouth
    (11,12),                                             # shoulders
    (11,13),(13,15),(15,17),(17,19),(19,15),(15,21),    # left arm
    (12,14),(14,16),(16,18),(18,20),(20,16),(16,22),    # right arm
    (11,23),(12,24),(23,24),                             # torso
    (23,25),(25,27),(27,29),(29,31),(31,27),             # left leg
    (24,26),(26,28),(28,30),(30,32),(32,28),             # right leg
]

FACE_OVAL = [
    (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),(356,454),
    (454,323),(323,361),(361,288),(288,397),(397,365),(365,379),(379,378),(378,400),
    (400,377),(377,152),(152,148),(148,176),(176,149),(149,150),(150,136),(136,172),
    (172,58),(58,132),(132,93),(93,234),(234,127),(127,162),(162,21),(21,54),
    (54,103),(103,67),(67,109),(109,10),
]
LEFT_EYE = [
    (263,249),(249,390),(390,373),(373,374),(374,380),(380,381),(381,382),(382,362),
    (362,398),(398,384),(384,385),(385,386),(386,387),(387,388),(388,466),(466,263),
]
RIGHT_EYE = [
    (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),(155,133),
    (133,173),(173,157),(157,158),(158,159),(159,160),(160,161),(161,246),(246,33),
]
LIPS = [
    (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),(405,321),(321,375),(375,291),
    (61,185),(185,40),(40,39),(39,37),(37,0),(0,267),(267,269),(269,270),(270,291),
    (78,95),(95,88),(88,178),(178,87),(87,14),(14,317),(317,402),(402,318),(318,324),(324,308),
    (78,191),(191,80),(80,81),(81,82),(82,13),(13,312),(312,311),(311,310),(310,415),(415,308),
]
FACE_CONNECTIONS = FACE_OVAL + LEFT_EYE + RIGHT_EYE + LIPS

# ── MediaPipe task factories ─────────────────────────────────────────────────
def _delegate(gpu: bool):
    return python.BaseOptions.Delegate.GPU if gpu else python.BaseOptions.Delegate.CPU

def _base(model_path: str, gpu: bool):
    return python.BaseOptions(model_asset_path=model_path, delegate=_delegate(gpu))

def _try_gpu(create_fn, gpu_opts, cpu_opts, name: str):
    if device == "cuda":
        try:
            task = create_fn(gpu_opts)
            print(f"MediaPipe {name}: GPU delegate")
            return task
        except Exception:
            print(f"MediaPipe {name}: GPU unavailable, falling back to CPU")
    print(f"MediaPipe {name}: CPU delegate")
    return create_fn(cpu_opts)

def create_hand_landmarker() -> vision.HandLandmarker:
    def opts(gpu):
        return vision.HandLandmarkerOptions(
            base_options=_base(HAND_MODEL, gpu),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _try_gpu(vision.HandLandmarker.create_from_options, opts(True), opts(False), "Hand")

def create_pose_landmarker() -> vision.PoseLandmarker:
    def opts(gpu):
        return vision.PoseLandmarkerOptions(
            base_options=_base(POSE_MODEL, gpu),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=4,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _try_gpu(vision.PoseLandmarker.create_from_options, opts(True), opts(False), "Pose")

def create_face_landmarker() -> vision.FaceLandmarker:
    def opts(gpu):
        return vision.FaceLandmarkerOptions(
            base_options=_base(FACE_MODEL, gpu),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=4,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _try_gpu(vision.FaceLandmarker.create_from_options, opts(True), opts(False), "Face")

# ── RF-DETR ──────────────────────────────────────────────────────────────────
print("Loading RF-DETR model...")
object_model = RFDETRBase(device=device)
object_model.optimize_for_inference()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# ── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default=None, help="Path to input video file (omit for webcam)")
parser.add_argument("--output", type=str, default=None, help="Path to save output video (omit to skip saving)")
parser.add_argument("--llm-interval", type=float, default=3.0, help="Seconds between LLM scene descriptions (default: 3.0)")
parser.add_argument("--llm-model", type=str, default="qwen2.5vl:7b", help="Ollama vision model name")
parser.add_argument("--no-llm", action="store_true", help="Disable LLM scene description overlay")
parser.add_argument("--show-description", action="store_true", help="Overlay the LLM description as subtitles on the video (always printed to terminal)")
args = parser.parse_args()

# ── Drawing helpers ──────────────────────────────────────────────────────────
def draw_landmarks(frame, landmarks, connections, dot_color, line_color, dot_r=3, thickness=1):
    h, w, _ = frame.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in connections:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], line_color, thickness)
    for x, y in pts:
        cv2.circle(frame, (x, y), dot_r, dot_color, -1)

def draw_subtitle(frame, text, max_chars_per_line=70, font_scale=0.35, line_height=14):
    if not text:
        return
    lines = textwrap.wrap(text, width=max_chars_per_line) or [text]
    h, w, _ = frame.shape
    box_height = line_height * len(lines) + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - box_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    y = h - box_height + line_height
    for line in lines:
        size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        x = max(10, (w - size[0]) // 2)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_height

# ── LLM scene description (runs in a background thread) ─────────────────────
LLM_IMAGE_MAX_WIDTH = 640  # downscale before sending to keep local 7B VLM inference fast on 8GB VRAM
# Saved for later use: detailed multi-aspect description of the operator's actions.
LLM_PROMPT = (
    "Describe the operator's actions in detail.\n"
    "Focus on:\n"
    "- hands\n"
    "- tools\n"
    "- objects\n"
    "- assembly operations\n"
    "- sequence of actions"
)

# Active prompt: short motion-element label for time-study logging
# (e.g. "Wait", "Assemble motor", "Inspection") instead of a full sentence.
# LLM_PROMPT = (
#     "Look at the operator in this image and classify the current action "
#     "with a short manufacturing motion label (2-5 words).\n"
#     "Common categories:\n"
#     "- Wait / Idle\n"
#     "- Walk / Move to station\n"
#     "- Pick up part\n"
#     "- Assemble motor\n"
#     "- Assemble component\n"
#     "- Use tool\n"
#     "- Inspection / Quality check\n"
#     "- Place / Position part\n"
#     "- Package part\n"
#     "If the action does not match these, describe it briefly in the same style.\n"
#     "Respond with only the short label, no extra explanation."
# )

MIN_CHUNK_SECONDS = 0.8  # floor so very short chunks aren't flashed instantly

def split_description(text: str) -> list:
    """Split an LLM response into subtitle-sized chunks.

    Handles numbered-list responses (e.g. "1. **Hands**: ...") by splitting on the
    numbering and stripping markdown bold markers; falls back to sentence splitting
    for plain prose, and returns the whole text as a single chunk if it's already short.
    """
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?m)^\s*\d+\.\s*", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        parts = re.split(r"(?<=[.!?])\s+", text)
        parts = [p.strip() for p in parts if p.strip()]
    cleaned = []
    for p in parts:
        p = re.sub(r"\*\*(.*?)\*\*", r"\1", p)
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            cleaned.append(p)
    return cleaned or [text]

def compute_chunk_durations(chunks: list, total_seconds: float) -> list:
    """Allocate display time per chunk, weighted by length, within total_seconds."""
    if not chunks:
        return []
    lengths = [max(len(c), 1) for c in chunks]
    total_len = sum(lengths)
    return [max(MIN_CHUNK_SECONDS, total_seconds * (l / total_len)) for l in lengths]

def current_chunk(chunks: list, durations: list, elapsed: float) -> str:
    """Pick which chunk should be on screen `elapsed` seconds after the update.

    Holds the last chunk if `elapsed` exceeds the total (e.g. the next LLM call
    is taking longer than --llm-interval), instead of going blank.
    """
    if not chunks:
        return ""
    t = 0.0
    for chunk, dur in zip(chunks, durations):
        t += dur
        if elapsed < t:
            return chunk
    return chunks[-1]


def _resize_for_llm(frame_bgr):
    h, w = frame_bgr.shape[:2]
    if w <= LLM_IMAGE_MAX_WIDTH:
        return frame_bgr
    scale = LLM_IMAGE_MAX_WIDTH / w
    return cv2.resize(frame_bgr, (LLM_IMAGE_MAX_WIDTH, int(h * scale)))

def describe_image(frame_bgr, model: str) -> str:
    small = _resize_for_llm(frame_bgr)
    ok, buf = cv2.imencode(".jpg", small)
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": LLM_PROMPT,
            "images": [buf.tobytes()],
        }],
    )
    return response["message"]["content"].strip()

def _describe_and_store(frame_lock, latest_frame, desc_lock, desc_state, model, interval) -> bool:
    with frame_lock:
        frame = latest_frame["frame"]
        frame = frame.copy() if frame is not None else None
    if frame is None:
        return False
    try:
        text = describe_image(frame, model=model)
        chunks = split_description(text)
        durations = compute_chunk_durations(chunks, interval)
        with desc_lock:
            desc_state["chunks"] = chunks
            desc_state["durations"] = durations
            desc_state["updated_at"] = time.time()
        print(f"[LLM] {text}")
    except Exception as e:
        print(f"[LLM] error: {e}")
    return True

def llm_worker(stop_event, frame_lock, latest_frame, desc_lock, desc_state, model, interval):
    # Describe the first available frame as soon as it arrives, instead of waiting a full interval
    while not stop_event.is_set():
        if _describe_and_store(frame_lock, latest_frame, desc_lock, desc_state, model, interval):
            break
        stop_event.wait(0.05)

    while not stop_event.is_set():
        start = time.time()
        _describe_and_store(frame_lock, latest_frame, desc_lock, desc_state, model, interval)
        elapsed = time.time() - start
        stop_event.wait(max(0.0, interval - elapsed))

# ── Open source & optional writer ───────────────────────────────────────────
source = args.file if args.file else 0
cap = cv2.VideoCapture(source)

writer = None
if args.output:
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w_src, h_src))
    print(f"Saving output to: {args.output}")

# ── Create all MediaPipe tasks ───────────────────────────────────────────────
hand_lm = create_hand_landmarker()
pose_lm = create_pose_landmarker()
face_lm = create_face_landmarker()
prev_time = time.time()

# ── Start LLM description thread ─────────────────────────────────────────────
stop_event = threading.Event()
frame_lock = threading.Lock()
latest_frame = {"frame": None}
desc_lock = threading.Lock()
desc_state = {"chunks": [], "durations": [], "updated_at": time.time()}

llm_thread = None
if not args.no_llm:
    llm_thread = threading.Thread(
        target=llm_worker,
        args=(stop_event, frame_lock, latest_frame, desc_lock, desc_state, args.llm_model, args.llm_interval),
        daemon=True,
    )
    llm_thread.start()
    print(f"LLM description: model={args.llm_model}, interval={args.llm_interval}s")
else:
    print("LLM description: disabled")

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        with frame_lock:
            latest_frame["frame"] = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # --- Object detection (RF-DETR) ---
        detections = object_model.predict(Image.fromarray(rgb), threshold=0.5)
        if len(detections) > 0:
            labels = [
                f"{COCO_CLASSES.get(cid, f'cls{cid}')} {conf:.2f}"
                for cid, conf in zip(detections.class_id, detections.confidence)
            ]
            frame = box_annotator.annotate(frame, detections)
            frame = label_annotator.annotate(frame, detections, labels=labels)

        # --- Body pose ---
        pose_results = pose_lm.detect(mp_image)
        if pose_results.pose_landmarks:
            for pose_landmarks in pose_results.pose_landmarks:
                draw_landmarks(frame, pose_landmarks, POSE_CONNECTIONS,
                               dot_color=(0, 165, 255), line_color=(0, 120, 255), dot_r=4, thickness=2)

        # --- Hands ---
        hand_results = hand_lm.detect(mp_image)
        if hand_results.hand_landmarks:
            for hand_landmarks in hand_results.hand_landmarks:
                draw_landmarks(frame, hand_landmarks, HAND_CONNECTIONS,
                               dot_color=(0, 255, 0), line_color=(0, 200, 0), dot_r=4, thickness=2)

        # --- Face ---
        face_results = face_lm.detect(mp_image)
        if face_results.face_landmarks:
            for face_landmarks in face_results.face_landmarks:
                draw_landmarks(frame, face_landmarks, FACE_CONNECTIONS,
                               dot_color=(255, 200, 0), line_color=(180, 140, 0), dot_r=1, thickness=1)

        # --- FPS & device overlay ---
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Device: {device.upper()}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # --- LLM subtitle overlay (terminal always gets it via _describe_and_store) ---
        if args.show_description:
            with desc_lock:
                chunks = desc_state["chunks"]
                durations = desc_state["durations"]
                updated_at = desc_state["updated_at"]
            elapsed = time.time() - updated_at
            draw_subtitle(frame, current_chunk(chunks, durations, elapsed))

        if writer:
            writer.write(frame)

        cv2.imshow("Pose + LLM Description", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    stop_event.set()
    if llm_thread is not None:
        llm_thread.join(timeout=2)
    hand_lm.close()
    pose_lm.close()
    face_lm.close()
    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
