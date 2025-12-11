import cv2
import warnings
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
import time

pt_path = "yolov5n.pt"
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    # 🌟 NEW SETUP: Load the model using the YOLO class
    model = YOLO(pt_path)
    
    # 🌟 NEW SETUP: Set model parameters (arguments are passed directly to the predict method)
    # The model object itself doesn't need .eval() or explicit size/conf setting here,
    # those parameters are passed during the call to model.predict().
    print(f"YOLO model loaded successfully from: {pt_path}")

except Exception as e:
    print(f"Error loading YOLO model from {pt_path}: {e}")
    print("Please ensure 'ultralytics' and 'torch' are installed,")
    print("and that the path to your .pt file is correct.")
    exit()

# COCO labels (These are standard for COCO-trained models and can be used as-is)
CLASSES = model.names # 🌟 BETTER: Use the names dictionary provided by the YOLO model object


# -------------------------------
# Draw boxes (Adapted for ultralytics results format)
# -------------------------------
def draw_boxes(frame, results):
    # ultralytics predict returns a list of Results objects (one per image).
    # We take the first result object.
    if not results or not results[0].boxes:
        return

    # Extract all boxes data at once
    boxes_data = results[0].boxes
    
    # Extract components from the Boxes object
    # xyxy is normalized coordinates (x1, y1, x2, y2)
    # The .data tensor contains [x1, y1, x2, y2, confidence, class_id]
    detections = boxes_data.data.cpu().numpy()

    if len(detections) > 0:
        # Extract components
        boxes = detections[:, :4]
        scores = detections[:, 4]
        class_ids = detections[:, 5].astype(int)

        for (box, score, cls_id) in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)

            # Get label from the CLASSES dictionary using the class ID
            label_name = CLASSES.get(cls_id, f"Class {cls_id}")
            label = f"{label_name} {score:.2f}"

            # Draw the bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# -------------------------------
# Camera setup (Remains the same)
# -------------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration({'format': 'RGB888'})
picam2.configure(config)
picam2.start()
time.sleep(1)

print("YOLO Real-time Detection (ultralytics) started (press 'q' to quit)")

# -------------------------------
# Main loop
# -------------------------------
prev_time = 0
fps = 0

while True:
    frame = picam2.capture_array()
    # The frame is already in BGR format from Picamera2 capture_array, 
    # but the config is 'RGB888'. It is safer to convert to RGB 
    # as ultralytics models generally expect RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    start_time = time.time()  # Start timing

    # 🌟 NEW INFERENCE: Use the predict method with arguments
    # source=rgb_frame: the image data (numpy array)
    # imgsz=416: the input image size
    # conf=0.35: the confidence threshold
    # iou=0.45: the IoU threshold for NMS
    # verbose=False: suppress logging output for each frame
    results = model.predict(
        source=rgb_frame, 
        imgsz=416, 
        conf=0.35,
        iou=0.45,
        verbose=False
    )
    
    # Draw boxes
    draw_boxes(frame, results)

    # Compute FPS (Remains the same)
    current_time = time.time()
    # E-M-A for a smoother FPS display
    fps = 0.9 * fps + 0.1 * (1 / (current_time - prev_time))
    prev_time = current_time

    # Draw FPS text
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("YOLO Real-time (ultralytics)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
picam2.stop()
picam2.close()
cv2.destroyAllWindows()
