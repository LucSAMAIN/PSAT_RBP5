import cv2
import os
import warnings
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
import time
import sys

MODEL_KEY = sys.argv[1]
if MODEL_KEY == "yolov5":
    pt_path = "yolo5/yolov5nu.pt"
elif MODEL_KEY == "yolov11":
    pt_path = "yolo11/yolo11n.pt"
warnings.filterwarnings("ignore", category=FutureWarning)

model = YOLO(pt_path)
print(f"YOLO model loaded successfully from: {pt_path}")

CLASSES = model.names

def draw_boxes(frame, results):
    if not results or not results[0].boxes:
        return

    boxes_data = results[0].boxes
    detections = boxes_data.data.cpu().numpy()

    if len(detections) > 0:
        boxes = detections[:, :4]
        scores = detections[:, 4]
        class_ids = detections[:, 5].astype(int)

        for (box, score, cls_id) in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)

            label_name = CLASSES.get(cls_id, f"Class {cls_id}")
            label = f"{label_name} {score:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

picam2 = Picamera2()
config = picam2.create_preview_configuration({'format': 'RGB888'})
picam2.configure(config)
picam2.start()
time.sleep(1)

print("YOLO Real-time Detection (ultralytics) started (press 'q' to quit)")

prev_time = 0
fps = 0

start_time = time.time()
export_csv = open(f"data-fps/fps-pt-{MODEL_KEY}-{start_time}.csv", "w")
# update the symbolic link
if os.path.exists(f"fps-pt-{MODEL_KEY}-latest.csv"):
    os.remove(f"fps-pt-{MODEL_KEY}-latest.csv")
os.symlink(f"data-fps/fps-pt-{MODEL_KEY}-{start_time}.csv", f"fps-pt-{MODEL_KEY}-latest.csv")
export_csv.write("model,framework,timestamp,fps\n")

while True:
    frame = picam2.capture_array()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.predict(
        source=rgb_frame, 
        imgsz=416, 
        conf=0.35,
        iou=0.45,
        verbose=False
    )
    draw_boxes(frame, results)

    current_time = time.time()
    if current_time - start_time > 30:
        break

    instant_fps = 1 / (current_time - prev_time)
    fps = 0.8 * fps + 0.2 * (instant_fps)
    line = f"{MODEL_KEY},pt,{current_time - start_time},{fps}\n"
    export_csv.write(line)
    prev_time = current_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("YOLO Real-time (ultralytics)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

export_csv.close()
picam2.stop()
picam2.close()
cv2.destroyAllWindows()
