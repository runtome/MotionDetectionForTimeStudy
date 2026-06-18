# Pipeline: RF-DETR person detection → crop → X3D-S clip classification
# X3D-S is trained on Kinetics-400 at 6 fps; we subsample webcam frames
# to match, so 13 frames ≈ 2 seconds of real action context.
#
# Setup (first time only):
#   pip install pytorchvideo
#   X3D-S weights (~200 MB) auto-download via torch.hub on first run.
#   Kinetics-400 class names JSON auto-downloads to poc/ on first run.

import argparse
import collections
import json
import os
import time
import urllib.request
import cv2
import torch
from PIL import Image
from rfdetr import RFDETRBase
import supervision as sv

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

# X3D-S clip parameters — match the training config exactly
CLIP_FRAMES = 13        # temporal depth
FRAME_SIZE  = 182       # spatial size (H and W)
SAMPLE_EVERY = 5        # keep 1 frame every 5 to approximate 6 fps from 30 fps
CROP_PAD    = 0.15      # fractional padding around person bbox
MIN_CROP_PX = 32        # skip crops smaller than this (blurry/partial detections)
PERSON_CLASS_ID = 1
KINETICS_MEAN = [0.45, 0.45, 0.45]
KINETICS_STD  = [0.225, 0.225, 0.225]

_POC_DIR = os.path.dirname(os.path.abspath(__file__))
KINETICS_LABELS_PATH = os.path.join(_POC_DIR, "kinetics400_classnames.json")
KINETICS_LABELS_URL  = (
    "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
)

# ── GPU ───────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Torch device: {device}")

# ── Kinetics-400 class names ──────────────────────────────────────────────────
kinetics_classes: dict[int, str] = {}
if not os.path.exists(KINETICS_LABELS_PATH):
    print("Downloading Kinetics-400 class names...")
    try:
        urllib.request.urlretrieve(KINETICS_LABELS_URL, KINETICS_LABELS_PATH)
        print(f"  -> {KINETICS_LABELS_PATH}")
    except Exception as e:
        print(f"  Warning: could not download class names ({e}) — will show indices")

if os.path.exists(KINETICS_LABELS_PATH):
    with open(KINETICS_LABELS_PATH) as f:
        raw = json.load(f)
    # JSON is {"class_name": index} — invert to {index: "class_name"}
    kinetics_classes = {int(v): k for k, v in raw.items()}

# ── Models ────────────────────────────────────────────────────────────────────
print("Loading RF-DETR...")
object_model = RFDETRBase(device=device)
object_model.optimize_for_inference()
box_annotator   = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

print("Loading X3D-S (~200 MB download on first run)...")
action_model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_s", pretrained=True)
action_model.eval()
action_model = action_model.to(device)
print("Models ready.")

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--file",      type=str,   default=None,  help="Input video file (omit for webcam)")
parser.add_argument("--output",    type=str,   default=None,  help="Save annotated output video")
parser.add_argument("--threshold", type=float, default=0.5,   help="RF-DETR detection threshold")
parser.add_argument("--top-k",     type=int,   default=3,     help="Number of top actions to display")
args = parser.parse_args()

# ── Helpers ───────────────────────────────────────────────────────────────────
def crop_person(frame, xyxy):
    """Return padded person crop, or None if too small."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    pw = int((x2 - x1) * CROP_PAD)
    ph = int((y2 - y1) * CROP_PAD)
    x1 = max(0, x1 - pw);  y1 = max(0, y1 - ph)
    x2 = min(w, x2 + pw);  y2 = min(h, y2 + ph)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.shape[0] >= MIN_CROP_PX and crop.shape[1] >= MIN_CROP_PX else None

def preprocess_crop(crop_bgr) -> torch.Tensor:
    """BGR crop → normalized (3, H, W) tensor for X3D-S."""
    rgb     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE))
    t = torch.from_numpy(resized).float() / 255.0    # (H, W, C)
    t = t.permute(2, 0, 1)                            # (C, H, W)
    for c in range(3):
        t[c] = (t[c] - KINETICS_MEAN[c]) / KINETICS_STD[c]
    return t

def run_action_model(buffer: collections.deque, top_k: int) -> list[tuple[str, float]]:
    clip = torch.stack(list(buffer))               # (T, C, H, W)
    clip = clip.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
    with torch.no_grad():
        logits = action_model(clip.to(device))
    probs = torch.softmax(logits[0], dim=0)
    topk  = torch.topk(probs, top_k)
    return [
        (kinetics_classes.get(idx.item(), f"action_{idx.item()}"), prob.item())
        for idx, prob in zip(topk.indices, topk.values)
    ]

def draw_action_panel(frame, results: list, buffer_len: int):
    """Draw top-k action predictions with a mini confidence bar."""
    y = 100
    status = f"Buffer: {buffer_len}/{CLIP_FRAMES}" if buffer_len < CLIP_FRAMES else "Action:"
    cv2.putText(frame, status, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA)
    y += 22
    for label, conf in results:
        bar_w = int(conf * 150)
        cv2.rectangle(frame, (10, y - 14), (10 + bar_w, y + 2), (0, 180, 255), -1)
        cv2.putText(frame, f"{label}  {conf:.2f}", (14, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22

# ── Source & writer ───────────────────────────────────────────────────────────
source = args.file if args.file else 0
cap = cv2.VideoCapture(source)

writer = None
if args.output:
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    w_src   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer  = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w_src, h_src))
    print(f"Saving output to: {args.output}")

# ── Main loop ─────────────────────────────────────────────────────────────────
frame_buffer: collections.deque = collections.deque(maxlen=CLIP_FRAMES)
action_labels: list = []
frame_count = 0
prev_time = time.time()

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── RF-DETR: detect all objects ───────────────────────────────────────
        detections = object_model.predict(Image.fromarray(rgb), threshold=args.threshold)

        primary_xyxy = None
        primary_conf = -1.0

        if len(detections) > 0:
            coco_labels = [
                f"{COCO_CLASSES.get(int(cid), f'cls{cid}')} {conf:.2f}"
                for cid, conf in zip(detections.class_id, detections.confidence)
            ]
            frame = box_annotator.annotate(frame, detections)
            frame = label_annotator.annotate(frame, detections, labels=coco_labels)

            for cid, conf, xyxy in zip(detections.class_id, detections.confidence, detections.xyxy):
                if int(cid) == PERSON_CLASS_ID and float(conf) > primary_conf:
                    primary_conf = float(conf)
                    primary_xyxy = xyxy

        # Cyan highlight on the person whose crops feed the action model
        if primary_xyxy is not None:
            x1, y1, x2, y2 = (int(v) for v in primary_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

        # ── Build clip buffer (subsample to ~6 fps) ───────────────────────────
        new_sample = False
        if frame_count % SAMPLE_EVERY == 0 and primary_xyxy is not None:
            crop = crop_person(frame, primary_xyxy)
            if crop is not None:
                frame_buffer.append(preprocess_crop(crop))
                new_sample = True

        # ── Run X3D-S when buffer is full and a new frame was just added ──────
        if new_sample and len(frame_buffer) == CLIP_FRAMES:
            action_labels = run_action_model(frame_buffer, args.top_k)

        # ── HUD ───────────────────────────────────────────────────────────────
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Device: {device.upper()}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        draw_action_panel(frame, action_labels, len(frame_buffer))

        if writer:
            writer.write(frame)

        cv2.imshow("Action Recognition — RF-DETR + X3D-S", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
