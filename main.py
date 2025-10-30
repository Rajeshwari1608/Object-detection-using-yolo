import cv2
from ultralytics import YOLO

# Load the pretrained YOLOv8 model (COCO dataset)
model = YOLO('yolov8n.pt')  # 'n' = nano version (fast); you can try 'yolov8s.pt' for more accuracy

# Choose input: 0 for webcam, or replace with "video.mp4"
video_source = 0  # or 'video.mp4'
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: Cannot open webcam or video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended or cannot read frame.")
        break

    # Run YOLO detection
    results = model(frame)

    # Visualize detection results on the frame
    annotated_frame = results[0].plot()

    # Display the output
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()