import cv2
import os
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time
import warnings
import sys

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
    "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

MODEL_CONFIGS = {
    "yolov5": {
        "onnx_path": "yolo5/yolov5n.onnx",
        "input_size": 416,
        "postprocess_func": "postprocess_yolov5_onnx",
        "conf_thres": 0.35,
        "iou_thres": 0.45,
    },
    "yolov11": {
        "onnx_path": "yolo11/yolo11n-416.onnx",
        "input_size": 416,
        "postprocess_func": "postprocess_yolov11_onnx",
        "conf_thres": 0.35,
        "iou_thres": 0.45,
    }
}

MODEL_KEY = sys.argv[1]

if MODEL_KEY not in MODEL_CONFIGS:
    print(f"Error: Model key '{MODEL_KEY}' not found in configurations.")
    exit()

CFG = MODEL_CONFIGS[MODEL_KEY]
ONNX_PATH = CFG["onnx_path"]
INPUT_SIZE = CFG["input_size"]
CONF_THRESHOLD = CFG["conf_thres"]
IOU_THRESHOLD = CFG["iou_thres"]

session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print(f"ONNX model '{MODEL_KEY}' loaded. Input: {input_name}, Shape: {session.get_inputs()[0].shape}")

def preprocess(frame, target_size=INPUT_SIZE):
    """Resizes and normalizes the image for model input (Letterbox)."""
    h, w, _ = frame.shape

    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    dw, dh = target_size - new_w, target_size - new_h
    dw /= 2
    dh /= 2
    img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img, scale, dw, dh, frame.shape

def postprocess_yolov5_onnx(raw_output, scale, dw, dh, original_shape, conf_thres, iou_thres):
    """Post-processes YOLOv5 ONNX output (NMS and inverse letterbox)."""
    prediction = np.squeeze(raw_output)

    boxes_raw = prediction[..., :4]
    objectness = prediction[..., 4:5]
    class_probs = prediction[..., 5:]
    scores_full = objectness * class_probs
    confidences = np.max(scores_full, axis=-1)
    class_ids = np.argmax(scores_full, axis=-1)
    mask = confidences > conf_thres

    boxes_raw, confidences, class_ids = boxes_raw[mask], confidences[mask], class_ids[mask]

    if len(boxes_raw) == 0:
        return [], [], []
    boxes_xyxy = np.zeros_like(boxes_raw)
    boxes_xyxy[:, 0] = boxes_raw[:, 0] - boxes_raw[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_raw[:, 1] - boxes_raw[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_raw[:, 0] + boxes_raw[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_raw[:, 1] + boxes_raw[:, 3] / 2
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), confidences.tolist(), conf_thres, iou_thres
    ).flatten()

    if len(indices) == 0:
        return [], [], []

    final_boxes = boxes_xyxy[indices]
    final_scores = confidences[indices]
    final_class_ids = class_ids[indices]
    h_orig, w_orig = original_shape[:2]
    final_boxes[:, [0, 2]] = (final_boxes[:, [0, 2]] - dw) / scale
    final_boxes[:, [1, 3]] = (final_boxes[:, [1, 3]] - dh) / scale
    final_boxes[:, 0] = np.clip(final_boxes[:, 0], 0, w_orig)
    final_boxes[:, 1] = np.clip(final_boxes[:, 1], 0, h_orig)
    final_boxes[:, 2] = np.clip(final_boxes[:, 2], 0, w_orig)
    final_boxes[:, 3] = np.clip(final_boxes[:, 3], 0, h_orig)

    return final_boxes.astype(int), final_scores, final_class_ids

def postprocess_yolov11_onnx(raw_output, scale, dw, dh, original_shape, conf_thres, iou_thres):
    """Post-processes YOLOv11 ONNX output (NMS and inverse letterbox)."""
    output = np.squeeze(raw_output).T

    boxes_raw = output[:, :4]
    scores_raw = output[:, 4:]
    confidences = np.max(scores_raw, axis=1)
    class_ids = np.argmax(scores_raw, axis=1)

    mask = confidences > conf_thres
    boxes_raw, confidences, class_ids = boxes_raw[mask], confidences[mask], class_ids[mask]

    if len(boxes_raw) == 0:
        return [], [], []
    boxes_xyxy = np.zeros_like(boxes_raw)
    boxes_xyxy[:, 0] = boxes_raw[:, 0] - boxes_raw[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_raw[:, 1] - boxes_raw[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_raw[:, 0] + boxes_raw[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_raw[:, 1] + boxes_raw[:, 3] / 2
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), confidences.tolist(), conf_thres, iou_thres
    ).flatten()

    if len(indices) == 0:
        return [], [], []

    final_boxes = boxes_xyxy[indices]
    final_scores = confidences[indices]
    final_class_ids = class_ids[indices]

    dw_norm = dw / scale
    dh_norm = dh / scale
    final_boxes[:, [0, 2]] = (final_boxes[:, [0, 2]] - dw_norm) / scale
    final_boxes[:, [1, 3]] = (final_boxes[:, [1, 3]] - dh_norm) / scale
    h_orig, w_orig = original_shape[:2]
    final_boxes[:, 0] = np.clip(final_boxes[:, 0], 0, w_orig)
    final_boxes[:, 1] = np.clip(final_boxes[:, 1], 0, h_orig)
    final_boxes[:, 2] = np.clip(final_boxes[:, 2], 0, w_orig)
    final_boxes[:, 3] = np.clip(final_boxes[:, 3], 0, h_orig)

    return final_boxes.astype(int), final_scores, final_class_ids

def draw_boxes(frame, boxes, scores, classes):
    """Draws bounding boxes and labels on the original frame."""
    for (box, score, cls_id) in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        label = f"{CLASSES[cls_id]} {score:.2f}"

        color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        text_y = y1 - 10 if y1 > h + 15 else y1 + 25
        text_bg_y = text_y - h - 5

        cv2.rectangle(frame, (x1, text_bg_y), (x1 + w, text_y), color, -1)

        cv2.putText(frame, label, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

picam2 = Picamera2()

config = picam2.create_preview_configuration(main={"size": (640, 480), "format": 'BGR888'})
picam2.configure(config)

try:
    picam2.start()
except Exception as e:
    print(f"Error starting Picamera2: {e}")
    print("Ensure the camera module is connected and enabled.")
    exit()

time.sleep(1)

postprocess_func = globals()[CFG["postprocess_func"]]

print(f"\n--- Unified YOLO Real-time Detection Started ({MODEL_KEY}) ---")
print(f"Model: {ONNX_PATH} | Input Size: {INPUT_SIZE}x{INPUT_SIZE}")
print("Press 'q' in the display window to quit.")

prev_time = time.time()
fps = 0

start_time = time.time()
export_csv = open(f"data-fps/fps-onnx-{MODEL_KEY}-{start_time}.csv", "w")
# update the symbolic link
if os.path.exists(f"fps-onnx-{MODEL_KEY}-latest.csv"):
    os.remove(f"fps-onnx-{MODEL_KEY}-latest.csv")
os.symlink(f"data-fps/fps-onnx-{MODEL_KEY}-{start_time}.csv", f"fps-onnx-{MODEL_KEY}-latest.csv")
export_csv.write("model,framework,timestamp,fps\n")

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img_input, scale, dw, dh, original_shape = preprocess(frame, INPUT_SIZE)

    raw_output = session.run(None, {input_name: img_input})[0]

    boxes, scores, class_ids = postprocess_func(
        raw_output, scale, dw, dh, original_shape, CONF_THRESHOLD, IOU_THRESHOLD
    )

    if len(boxes) > 0:
        draw_boxes(frame, boxes, scores, class_ids)

    current_time = time.time()
    if current_time - start_time > 30:
        break

    instant_fps = 1 / (current_time - prev_time)
    fps = 0.8 * fps + 0.2 * (instant_fps)
    line = f"{MODEL_KEY},onnx,{current_time - start_time},{fps}\n"
    export_csv.write(line)
    prev_time = current_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Unified YOLO ONNX Real-time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

export_csv.close()
picam2.stop()
picam2.close()
cv2.destroyAllWindows()

print("\n--- Detection stopped and resources released. ---")
