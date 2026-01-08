import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time
import warnings

# Suppress warnings that might clutter the console
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
INPUT_SIZE = 416             # Model input size (width and height) 640 or 416
ONNX_PATH = f"yolo11n-{INPUT_SIZE}.onnx"   # 🔸 Change this to your model path
CONF_THRESHOLD = 0.30        # Confidence threshold for filtering detections
IOU_THRESHOLD = 0.45         # IoU threshold for Non-Maximum Suppression (NMS)

# COCO labels (YOLOv11 trained on COCO has 80 classes)
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

# -------------------------------
# 1. ONNX Runtime Setup
# -------------------------------
try:
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    # Assuming fixed input size (1, 3, 640, 640)
    print(f"ONNX model loaded. Input: {input_name}, Shape: {session.get_inputs()[0].shape}")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    print("Please ensure your 'yolo11n.onnx' file is valid and in the correct path.")
    exit()

# -------------------------------
# 2. Preprocessing (Letterbox)
# -------------------------------
def preprocess(frame, target_size=INPUT_SIZE):
    """Resizes and normalizes the image for model input."""
    # 1. Resize/Letterbox
    h, w, _ = frame.shape
    scale = min(target_size / h, target_size / w)
    
    # Calculate new dimensions and padding
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    dw, dh = target_size - new_w, target_size - new_h
    dw /= 2
    dh /= 2

    # Resize image
    img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad image (Letterbox)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # 2. Normalize and format
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0         # Normalize to [0, 1]
    img = np.transpose(img, (2, 0, 1))          # HWC -> CHW (3, 640, 640)
    img = np.expand_dims(img, 0)                # Add batch dimension (1, 3, 640, 640)
    
    # Return processed input and scaling parameters for later inverse transform
    return img, scale, dw, dh, frame.shape

# -------------------------------
# 3. Post-processing (YOLOv11 ONNX Output Decoding)
# -------------------------------
def postprocess_yolov11_onnx(raw_output, scale, dw, dh, original_shape, conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD):
    """
    Decodes the raw YOLOv11 ONNX output tensor (1, 84, 8400) and applies NMS.
    """
    # 1. Prepare raw output
    # Shape: (1, 84, 8400) -> Transpose to (8400, 84)
    # The output structure is typically: [box_xywh, class_probabilities...]
    output = np.squeeze(raw_output).T
    
    # The first 4 elements are box coordinates (cx, cy, w, h)
    boxes_raw = output[:, :4] 
    
    # The remaining elements are class scores (80 classes)
    scores_raw = output[:, 4:] 

    # 2. Filter by Confidence
    # Get the max score and class ID for each candidate box
    confidences = np.max(scores_raw, axis=1)
    class_ids = np.argmax(scores_raw, axis=1)
    
    # Apply confidence filter
    mask = confidences > conf_thres
    boxes_raw, confidences, class_ids = boxes_raw[mask], confidences[mask], class_ids[mask]
    
    if len(boxes_raw) == 0:
        return [], [], []

    # 3. Convert [cx, cy, w, h] to [x1, y1, x2, y2]
    boxes_xyxy = np.zeros_like(boxes_raw)
    boxes_xyxy[:, 0] = boxes_raw[:, 0] - boxes_raw[:, 2] / 2 # x1 = cx - w/2
    boxes_xyxy[:, 1] = boxes_raw[:, 1] - boxes_raw[:, 3] / 2 # y1 = cy - h/2
    boxes_xyxy[:, 2] = boxes_raw[:, 0] + boxes_raw[:, 2] / 2 # x2 = cx + w/2
    boxes_xyxy[:, 3] = boxes_raw[:, 1] + boxes_raw[:, 3] / 2 # y2 = cy + h/2

    # 4. Apply NMS (Non-Maximum Suppression)
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), confidences.tolist(), conf_thres, iou_thres
    ).flatten()

    if len(indices) == 0:
        return [], [], []

    final_boxes = boxes_xyxy[indices]
    final_scores = confidences[indices]
    final_class_ids = class_ids[indices]

    # 5. Rescale boxes back to original image size
    # Inverse Letterbox Transformation
    
    # Calculate padding offset
    dw = dw / scale
    dh = dh / scale
    
    # Rescale and offset
    final_boxes[:, [0, 2]] = (final_boxes[:, [0, 2]] - dw) / scale
    final_boxes[:, [1, 3]] = (final_boxes[:, [1, 3]] - dh) / scale
    
    # Clip coordinates to original image bounds
    h_orig, w_orig = original_shape[:2]
    final_boxes[:, 0] = np.clip(final_boxes[:, 0], 0, w_orig)
    final_boxes[:, 1] = np.clip(final_boxes[:, 1], 0, h_orig)
    final_boxes[:, 2] = np.clip(final_boxes[:, 2], 0, w_orig)
    final_boxes[:, 3] = np.clip(final_boxes[:, 3], 0, h_orig)
    
    return final_boxes.astype(int), final_scores, final_class_ids

# -------------------------------
# 4. Drawing Utility
# -------------------------------
def draw_boxes(frame, boxes, scores, classes):
    """Draws bounding boxes and labels on the original frame."""
    for (box, score, cls_id) in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        label = f"{CLASSES[cls_id]} {score:.2f}"
        
        color = (0, 255, 0) # Green for detection

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw background for text
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1+h+10), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# -------------------------------
# 5. Camera Setup and Main Loop
# -------------------------------
picam2 = Picamera2()
# Request BGR format for OpenCV compatibility (BGR is OpenCV's native color order)
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": 'BGR888'})
picam2.configure(config)

try:
    picam2.start()
except Exception as e:
    print(f"Error starting Picamera2: {e}")
    print("Ensure the camera module is connected and enabled.")
    exit()

time.sleep(1)

print("\n--- YOLOv11 ONNX Real-time Detection Started ---")
print(f"Model: {ONNX_PATH} | Input Size: {INPUT_SIZE}x{INPUT_SIZE}")
print("Press 'q' in the display window to quit.")

prev_time = time.time()
fps = 0

while True:
    # Capture frame as a NumPy array (BGR format)
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_shape = frame.shape
    
    start_time = time.time()  # Start timing for detection
    
    # 1. Preprocess
    img_input, scale, dw, dh, original_shape = preprocess(frame, INPUT_SIZE)
    
    # 2. Inference
    raw_output = session.run(None, {input_name: img_input})[0]
    
    # 3. Post-process
    boxes, scores, class_ids = postprocess_yolov11_onnx(
        raw_output, scale, dw, dh, original_shape, CONF_THRESHOLD, IOU_THRESHOLD
    )

    # 4. Draw results
    if len(boxes) > 0:
        draw_boxes(frame, boxes, scores, class_ids)

    # 5. Compute and display FPS
    current_time = time.time()
    # Apply a smooth FPS calculation (exponential moving average)
    fps = 0.9 * fps + 0.1 * (1 / (current_time - prev_time))
    prev_time = current_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    # display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 6. Display frame and check for exit key
    cv2.imshow("YOLOv11 ONNX Real-time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
picam2.stop()
picam2.close()
cv2.destroyAllWindows()

print("\n--- Detection stopped and resources released. ---")
