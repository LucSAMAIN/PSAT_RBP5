import cv2
import random
from ultralytics import YOLO
import datetime

model = "./models/yolo11x.pt"
yolo = YOLO(model) # Use the nano model for better performance
# yolo.export(format='tflite', int8=True)
def getColours(cls_num):
    """Generate unique colors for each class ID"""
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))




video_path = "/dev/video0"
videoCap = cv2.VideoCapture(video_path)

# Use a default FPS if it cannot be determined from the video source
fps = 10

# Define the codec and create VideoWriter object
output_path = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')


frame_count = 0
frame_skip = 1  # Process every 2nd frame

while True:
    ret, frame = videoCap.read()
    if not ret:
        break


    if frame_count % frame_skip == 0:
        results = yolo.track(frame, stream=True, verbose=True)
        
        for result in results:
            class_names = result.names
            for box in result.boxes:
                if box.conf[0] > 0.2:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cls = int(box.cls[0])
                    class_name = class_names[cls]   

                    conf = float(box.conf[0])

                    colour = getColours(cls)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                    cv2.putText(frame, f"{class_name} {conf:.2f}",
                                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, colour, 2)


    cv2.imshow(f"{model} Tracking", frame)
    frame_count += 1

    # Wait for 1ms and break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything when job is finished
videoCap.release()
cv2.destroyAllWindows()