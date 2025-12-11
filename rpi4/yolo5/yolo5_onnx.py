import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time

# -------------------------------
# YOLOv5 ONNX setup
# -------------------------------
onnx_path = "yolov5n.onnx"   # 🔸 Change this to your model path
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# COCO labels
CLASSES = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis",
    "snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
    "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
    "pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable",
    "toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase",
    "scissors","teddy bear","hair drier","toothbrush"
]

# -------------------------------
# Letterbox (resize + pad)
# -------------------------------
def letterbox(image, new_shape=416, color=(114, 114, 114)):
    h, w = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    # Resize
    image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    # Pad
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=color)
    return image, r, dw, dh

# -------------------------------
# Preprocess
# -------------------------------
def preprocess(frame, img_size=416):
    img, r, dw, dh = letterbox(frame, img_size)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, 0)
    return img, r, dw, dh

# -------------------------------
# Postprocess (NMS)
# -------------------------------
def non_max_suppression(prediction, conf_thres=0.35, iou_thres=0.45):
    boxes = prediction[..., :4]
    objectness = prediction[..., 4:5]
    class_probs = prediction[..., 5:]
    scores = objectness * class_probs

    class_ids = np.argmax(scores, axis=-1)
    confidences = np.max(scores, axis=-1)

    mask = confidences > conf_thres
    boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

    if len(boxes) == 0:
        return []

    # Convert [cx, cy, w, h] → [x1, y1, x2, y2]
    xyxy_boxes = np.zeros_like(boxes)
    xyxy_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    indices = cv2.dnn.NMSBoxes(
        xyxy_boxes.tolist(), confidences.tolist(), conf_thres, iou_thres
    )
    if len(indices) == 0:
        return []

    indices = indices.flatten()
    return xyxy_boxes[indices], confidences[indices], class_ids[indices]

# -------------------------------
# Draw boxes
# -------------------------------
def draw_boxes(frame, boxes, scores, classes, r, dw, dh):
    h, w, _ = frame.shape
    for (box, score, cls_id) in zip(boxes, scores, classes):
        # Reverse letterbox transformation
        x1 = int((box[0] - dw) / r)
        y1 = int((box[1] - dh) / r)
        x2 = int((box[2] - dw) / r)
        y2 = int((box[3] - dh) / r)

        label = f"{CLASSES[int(cls_id)]} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# -------------------------------
# Camera setup
# -------------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration({'format': 'RGB888'})
picam2.configure(config)
picam2.start()
time.sleep(1)

print("YOLOv5 ONNX Real-time Detection started (press 'q' to quit)")

# -------------------------------
# Main loop
# -------------------------------
prev_time = 0
fps = 0

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    start_time = time.time()  # Start timing

    img_input, r, dw, dh = preprocess(frame, 416)
    pred = session.run([output_name], {input_name: img_input})[0]
    pred = np.squeeze(pred)

    detections = non_max_suppression(pred, conf_thres=0.35, iou_thres=0.45)
    if len(detections) > 0:
        boxes, scores, class_ids = detections
        draw_boxes(frame, boxes, scores, class_ids, r, dw, dh)

    # Compute FPS
    current_time = time.time()
    # fps = 1 / (current_time - prev_time) if prev_time else 0
    fps = 0.9 * fps + 0.1 * (1 / (current_time - prev_time))
    prev_time = current_time

    # Draw FPS text
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("YOLOv5 ONNX Real-time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
picam2.close()
cv2.destroyAllWindows()
