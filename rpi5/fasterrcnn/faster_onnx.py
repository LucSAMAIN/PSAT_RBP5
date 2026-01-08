import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
INPUT_SIZE = 320
ONNX_PATH = f"FasterRCNN_MobileNet_V3_Large_{INPUT_SIZE}_FPN.onnx" # 🔸 Change this to your model path
CONF_THRESHOLD = 0.50

# COCO labels (91 classes including background, index 0)
CLASSES = [
    "__background__", "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
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

# -------------------------------
# 1. ONNX Runtime Setup
# -------------------------------
try:
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"ONNX model loaded. Input: {input_name}, Shape: {session.get_inputs()[0].shape}")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    exit()

# -------------------------------
# 2. Preprocessing (Simple Resize and [0, 1] Normalization)
# -------------------------------
def preprocess(frame, target_size=INPUT_SIZE):
    """Resizes, normalizes (to [0, 1]), and converts the image for ONNX input."""
    img = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img /= 255.0
    
    # Check if the input shape is NCHW (1, 3, H, W)
    if session.get_inputs()[0].shape[1] == 3: 
        img = np.transpose(img, (2, 0, 1)) # HWC -> CHW 
        
    img = np.expand_dims(img, 0) # Add batch dimension
    return img, frame.shape

# -------------------------------
# 3. Post-processing (Faster R-CNN ONNX Output Decoding)
# -------------------------------
def postprocess_fasterrcnn_onnx(raw_outputs, original_shape, conf_thres=CONF_THRESHOLD):
    """
    Decodes the raw Faster R-CNN ONNX output tensors.
    
    Assumes a pre-NMS, pre-filtered output structure (boxes, scores, classes).
    The exact order [0], [1], [2] MUST be verified against your ONNX model's output names.
    """
    h_orig, w_orig = original_shape[:2]
    
    # 🌟 CRITICAL ASSUMPTION: The order is Boxes, Scores, Classes
    try:
        boxes_norm = raw_outputs[0].squeeze() # [N, 4]
        scores_raw = raw_outputs[1].squeeze() # [N]
        class_ids_raw = raw_outputs[2].squeeze().astype(np.int32) # [N]
    except IndexError:
        print("Warning: Faster R-CNN ONNX output structure is not the expected [boxes, scores, classes].")
        return [], [], []

    # 1. Filter by Confidence
    mask = scores_raw > conf_thres
    final_boxes_norm = boxes_norm[mask]
    final_scores = scores_raw[mask]
    final_class_ids = class_ids_raw[mask]

    if len(final_boxes_norm) == 0:
        return [], [], []

    # 2. Rescale normalized [x1, y1, x2, y2] to pixel coordinates
    # Torchvision models typically output boxes scaled to the input size (320x320)
    scale_w = w_orig / INPUT_SIZE
    scale_h = h_orig / INPUT_SIZE
    
    final_boxes = np.zeros_like(final_boxes_norm)
    
    # Assume [x1, y1, x2, y2] relative to INPUT_SIZE (320)
    final_boxes[:, [0, 2]] = final_boxes_norm[:, [0, 2]] * scale_w # x coordinates
    final_boxes[:, [1, 3]] = final_boxes_norm[:, [1, 3]] * scale_h # y coordinates

    # 3. Clip coordinates to original image bounds
    final_boxes[:, 0] = np.clip(final_boxes[:, 0], 0, w_orig)
    final_boxes[:, 1] = np.clip(final_boxes[:, 1], 0, h_orig)
    final_boxes[:, 2] = np.clip(final_boxes[:, 2], 0, w_orig)
    final_boxes[:, 3] = np.clip(final_boxes[:, 3], 0, h_orig)

    return final_boxes.astype(int), final_scores, final_class_ids

# -------------------------------
# 4. Drawing Utility & Main Loop (Adapted for ONNX)
# -------------------------------
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
        cv2.putText(frame, label, (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": 'BGR888'})
picam2.configure(config)

try:
    picam2.start()
except Exception as e:
    print(f"Error starting Picamera2: {e}")
    exit()
time.sleep(1)

prev_time = time.time()
fps = 0
print(f"\n--- Faster R-CNN MobileNet V3 ONNX Real-time Detection Started ---")

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_shape = frame.shape

    img_input, _ = preprocess(frame, INPUT_SIZE)

    # 2. Inference
    raw_outputs = session.run(None, {input_name: img_input})

    # 3. Post-process
    boxes, scores, class_ids = postprocess_fasterrcnn_onnx(
        raw_outputs, original_shape, CONF_THRESHOLD
    )

    if len(boxes) > 0:
        draw_boxes(frame, boxes, scores, class_ids)
    
    current_time = time.time()
    fps = 0.9 * fps + 0.1 * (1 / (current_time - prev_time))
    prev_time = current_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Faster R-CNN MobileNet V3 Real-time (ONNX)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
picam2.close()
cv2.destroyAllWindows()
print("\n--- Detection stopped and resources released. ---")
