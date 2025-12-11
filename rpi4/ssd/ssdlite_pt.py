import cv2
import numpy as np
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
from picamera2 import Picamera2
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

INPUT_SIZE = 320

# PT_PATH = f"ssdlite_mobilenet_v3_large-{INPUT_SIZE}.pt"
CONF_THRESHOLD = 0.40
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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    ).to(DEVICE)

    model.eval()

    print(f"PyTorch model (Torchvision SSDLite) loaded.")

except Exception as e:
    print(f"\nError loading PyTorch model: {e}")
    print("Ensure you have installed the latest torchvision version.")
    exit()

def preprocess(frame, target_size=INPUT_SIZE):
    """Resizes, normalizes, and converts the image to a PyTorch Tensor."""
    img = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = torch.from_numpy(img).to(DEVICE)

    img = img.permute(2, 0, 1)
    img /= 255.0
    img = img.unsqueeze(0)

    return img, frame.shape

def postprocess_pt_output(raw_output, original_shape, conf_thres=CONF_THRESHOLD):
    """
    Handles the output of a standard PyTorch detection model.
    Output is usually a list of dicts, one for each image in the batch.
    """
    h_orig, w_orig = original_shape[:2]
    if not isinstance(raw_output, list) or len(raw_output) == 0:
        return [], [], []

    results = raw_output[0]

    if not results or not 'boxes' in results or len(results['boxes']) == 0:
        return [], [], []
    boxes = results['boxes'].cpu().numpy()
    scores = results['scores'].cpu().numpy()
    class_ids = results['labels'].cpu().numpy().astype(np.int32)
    mask = scores > conf_thres
    final_boxes = boxes[mask]
    final_scores = scores[mask]
    final_class_ids = class_ids[mask]

    if len(final_boxes) == 0:
        return [], [], []
    scale_w = w_orig / INPUT_SIZE
    scale_h = h_orig / INPUT_SIZE
    final_boxes[:, [0, 2]] *= scale_w

    final_boxes[:, [1, 3]] *= scale_h
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
    exit()

time.sleep(1)

print(f"\n--- PyTorch SSDLite MobileNet V3 Real-time Detection Started ---")
print(f"Model: | Input Size: {INPUT_SIZE}x{INPUT_SIZE} | Device: {DEVICE}")
print("Press 'q' in the display window to quit.")

prev_time = time.time()
fps = 0

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_shape = frame.shape
    img_input_tensor, _ = preprocess(frame, INPUT_SIZE)
    with torch.no_grad():
        raw_output = model(img_input_tensor)

    boxes, scores, class_ids = postprocess_pt_output(
        raw_output, original_shape, CONF_THRESHOLD
    )
    if len(boxes) > 0:
        draw_boxes(frame, boxes, scores, class_ids)
    current_time = time.time()
    fps = 0.9 * fps + 0.1 * (1 / (current_time - prev_time))
    prev_time = current_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("PyTorch SSDLite MobileNet V3 Real-time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
picam2.close()
cv2.destroyAllWindows()

print("\n--- Detection stopped and resources released. ---")
