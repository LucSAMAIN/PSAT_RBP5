import cv2
import os
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

INPUT_SIZE = 320
MODEL_KEY = "ssdlite"
ONNX_PATH = f"ssd/ssdlite_mobilenet_v3_large-{INPUT_SIZE}.onnx"
CONF_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45
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

try:
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    print("\n--- Model Output Details ---")
    for output in session.get_outputs():
        print(f"Name: {output.name}, Shape: {output.shape}")
    print("--------------------------\n")
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    print(f"ONNX model loaded. Input: {input_name}, Shape: {session.get_inputs()[0].shape}")
    print(f"Output Tensors: {output_names}")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    print(f"Please ensure your '{ONNX_PATH}' file is valid and in the correct path.")
    exit()

def preprocess(frame, target_size=INPUT_SIZE):
    """
    Resizes the image and normalizes it to the expected SSD format (e.g., [-1, 1]).
    """
    img = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0

    if session.get_inputs()[0].shape[1] == 3:
        img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img, frame.shape

def postprocess_ssdlite_onnx(boxes_raw, scores_raw, class_ids_raw, original_shape):
    """
    Decodes the raw SSDLite MobileNet ONNX output tensors based on the provided
    [N, 4], [N], [N] structure. Assumes pre-filtered and pre-NMS boxes.
    """
    if len(scores_raw) == 0:
        return [], [], []

    h_orig, w_orig = original_shape[:2]

    confidences = scores_raw
    class_ids = class_ids_raw.astype(np.int32)

    boxes_norm = boxes_raw
    boxes_xyxy = np.zeros_like(boxes_norm)
    boxes_xyxy[:, 0] = boxes_norm[:, 1] * w_orig
    boxes_xyxy[:, 1] = boxes_norm[:, 0] * h_orig
    boxes_xyxy[:, 2] = boxes_norm[:, 3] * w_orig
    boxes_xyxy[:, 3] = boxes_norm[:, 2] * h_orig
    final_boxes = boxes_xyxy
    final_scores = confidences
    final_class_ids = class_ids
    if final_class_ids.min() >= 1:
        final_class_ids = final_class_ids - 1
    final_boxes[:, 0] = np.clip(final_boxes[:, 0], 0, w_orig)
    final_boxes[:, 1] = np.clip(final_boxes[:, 1], 0, h_orig)
    final_boxes[:, 2] = np.clip(final_boxes[:, 2], 0, w_orig)
    final_boxes[:, 3] = np.clip(final_boxes[:, 3], 0, h_orig)

    return final_boxes.astype(int), final_scores, final_class_ids



def draw_boxes(frame, boxes, scores, classes):
    """Draws bounding boxes and labels on the original frame."""
    for (box, score, cls_id) in zip(boxes, scores, classes):
        if cls_id >= len(CLASSES):
            label = f"Unknown {score:.2f}"
        else:
            label = f"{CLASSES[cls_id]} {score:.2f}"

        x1, y1, x2, y2 = box
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1+h+10), color, -1)

        cv2.putText(frame, label, (x1, y1 + 15),
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

print(f"\n--- SSDLite MobileNet V3 ONNX Real-time Detection Started ---")
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
    original_shape = frame.shape

    img_input, original_shape_info = preprocess(frame, INPUT_SIZE)

    raw_outputs = session.run(None, {input_name: img_input})

    boxes, scores, class_ids = postprocess_ssdlite_onnx(
        raw_outputs[0], raw_outputs[2], raw_outputs[1], original_shape
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
    cv2.imshow("SSDLite MobileNet V3 ONNX Real-time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

export_csv.close()
picam2.stop()
picam2.close()
cv2.destroyAllWindows()

print("\n--- Detection stopped and resources released. ---")
