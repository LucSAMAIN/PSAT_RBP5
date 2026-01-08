import cv2


video_path = "/dev/video0"
videoCap = cv2.VideoCapture(video_path)

while True:
    ret, frame = videoCap.read()
    if not ret:
        break


    cv2.imshow(f"Tracking", frame)

    # Wait for 1ms and break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything when job is finished
videoCap.release()
cv2.destroyAllWindows()
