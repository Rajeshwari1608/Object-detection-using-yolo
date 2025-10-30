# Object-detection-using-yolo
Real-Time Object Detection using YOLOv8
1. Introduction
Object detection is a key area of computer vision that focuses on identifying and locating multiple objects within an image or video. It has numerous real-world applications such as security monitoring, autonomous vehicles, and traffic management.
This project implements real-time object detection using YOLOv8 (You Only Look Once, Version 8) — a state-of-the-art deep learning model developed by Ultralytics. YOLOv8 is known for its high speed, accuracy, and ease of use, making it ideal for real-time applications.
The system uses a webcam or video input to capture frames, applies the YOLOv8 model to detect objects, and displays bounding boxes with class labels and confidence scores in real time.
________________________________________
2. Objectives
The objectives of this project are:
•	To understand and implement object detection using a pretrained YOLOv8 model.
•	To perform real-time detection using webcam or video input.
•	To visualize the detected objects through bounding boxes and confidence levels.
•	To demonstrate the use of deep learning models for visual recognition tasks.
________________________________________
3. Tools and Technologies Used
Component	Description
Programming Language	Python
Deep Learning Framework	Ultralytics YOLOv8
Image/Video Processing Library	OpenCV
Model Used	yolov8n.pt (Nano version of YOLOv8)
Dataset	COCO (Common Objects in Context) Dataset
Hardware	Laptop/PC with Webcam, optional GPU for acceleration
Python Libraries:
pip install ultralytics opencv-python
________________________________________
4. System Requirements
•	Software Requirements:
o	Python 3.8 or above
o	Ultralytics library
o	OpenCV
o	NumPy (optional)
•	Hardware Requirements:
o	A computer with minimum 4GB RAM
o	Integrated or external webcam
o	GPU (optional for faster performance)
________________________________________
5. Methodology
The project follows the following steps:
Step 1: Import Dependencies
Import necessary Python libraries:
import cv2
from ultralytics import YOLO
Step 2: Load Pretrained Model
Load the YOLOv8 model:
model = YOLO('yolov8n.pt')
The model is pretrained on the COCO dataset, which can detect 80 classes such as person, car, dog, bicycle, traffic light, etc.
Step 3: Capture Video Input
The system captures video input from a webcam or video file:
cap = cv2.VideoCapture(0)  # 0 for webcam
Step 4: Frame-by-Frame Detection
For each frame:
1.	Read the frame.
2.	Run object detection:
3.	results = model(frame)
4.	Annotate the frame with detection results:
5.	annotated_frame = results[0].plot()
6.	Display the annotated frame in a window using OpenCV.
Step 5: Exit Mechanism
The user can stop the detection process by pressing the ‘q’ key.
All resources (camera feed and windows) are then released properly.
________________________________________
6. Workflow Diagram
Start
│
├── Load YOLOv8 Model
│
├── Open Video Stream (Webcam/Video)
│
├── While Video Stream is Active:
│     ├── Capture Current Frame
│     ├── Perform Object Detection
│     ├── Annotate Frame with Bounding Boxes
│     ├── Display Annotated Frame
│     └── Wait for 'q' to Exit
│
└── Release Resources and Close Windows
________________________________________
7. Results and Observations
•	The model accurately detects multiple objects simultaneously, even under different lighting conditions.
•	Each detected object is enclosed within a bounding box labeled with its class name and confidence score.
•	The Nano version (yolov8n.pt) provides real-time detection on standard CPUs with minimal delay.
•	With a GPU, the system achieves even higher frame rates.
Example detections:
•	Person, bicycle, car, dog, chair, bottle, etc.
•	Objects are tracked across frames smoothly in real time.
________________________________________
8. Applications
This object detection system has numerous potential applications, such as:
•	Surveillance and Security: Detecting intruders, vehicles, or unattended objects.
•	Autonomous Vehicles: Detecting pedestrians, traffic lights, and road signs.
•	Retail Analytics: Counting customers or tracking product movements.
•	Robotics: Enabling robots to recognize and interact with their environment.
•	Healthcare: Monitoring patient activity or detecting medical instruments.
________________________________________
9. Advantages
•	Fast and Efficient: Real-time performance even on modest hardware.
•	Pretrained Model: No need for custom dataset training.
•	Scalable: Can easily switch to larger YOLO models (s, m, l, x) for higher accuracy.
•	Versatile Input: Works with both webcams and video files.
________________________________________
10. Limitations
•	Accuracy depends on the lighting and camera quality.
•	The Nano model may not detect very small or distant objects accurately.
•	Performance may drop on low-end CPUs without GPU support.
________________________________________
11. Future Enhancements
•	Use higher accuracy models like yolov8s.pt or yolov8m.pt.
•	Add object tracking (e.g., SORT or DeepSORT) to maintain consistent IDs across frames.
•	Integrate voice alerts or email notifications for specific detections.
•	Build a web-based interface for remote video monitoring.
•	Optimize using TensorRT or ONNX Runtime for deployment on edge devices.
________________________________________
12. Conclusion
This project successfully demonstrates how to perform real-time object detection using YOLOv8 and OpenCV. The system can detect and classify multiple objects simultaneously, providing fast and accurate results suitable for real-world computer vision tasks.
It highlights how modern deep learning models can be integrated into simple Python scripts to build intelligent and interactive applications.

