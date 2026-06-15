import argparse
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

# RF-DETR uses 1-indexed COCO IDs (1-90) with gaps — use a dict for safe lookup
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

# --- Hand Landmarker setup ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

def _make_hand_options(gpu: bool) -> vision.HandLandmarkerOptions:
    delegate = python.BaseOptions.Delegate.GPU if gpu else python.BaseOptions.Delegate.CPU
    return vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH, delegate=delegate),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

def _create_landmarker() -> vision.HandLandmarker:
    if device == "cuda":
        try:
            lm = vision.HandLandmarker.create_from_options(_make_hand_options(gpu=True))
            print("MediaPipe: GPU delegate")
            return lm
        except Exception:
            print("MediaPipe: GPU delegate unavailable, falling back to CPU")
    print("MediaPipe: CPU delegate")
    return vision.HandLandmarker.create_from_options(_make_hand_options(gpu=False))

# --- RF-DETR setup ---
print("Loading RF-DETR model...")
object_model = RFDETRBase(device=device)
object_model.optimize_for_inference()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# --- Argument parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default=None, help="Path to a video file (omit to use webcam)")
args = parser.parse_args()

source = args.file if args.file else 0
cap = cv2.VideoCapture(source)
landmarker = _create_landmarker()
prev_time = time.time()

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert once for both models
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Object detection (RF-DETR) ---
        detections = object_model.predict(Image.fromarray(rgb), threshold=0.5)
        if len(detections) > 0:
            labels = [
                f"{COCO_CLASSES.get(cid, f'cls{cid}')} {conf:.2f}"
                for cid, conf in zip(detections.class_id, detections.confidence)
            ]
            frame = box_annotator.annotate(frame, detections)
            frame = label_annotator.annotate(frame, detections, labels=labels)

        # --- Hand landmark detection (MediaPipe) ---
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect(mp_image)

        if results.hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in results.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)

                for idx, (x, y) in enumerate(pts):
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Device: {device.upper()}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Object + Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
