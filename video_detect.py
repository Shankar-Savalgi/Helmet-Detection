import cv2
from ultralytics import YOLO
import os

# Load trained model
model = YOLO("models/best.pt")

# Input and output paths
video_path = "videos/traffic.mp4"
output_path = "output/helmet_detection_result.mp4"

# Open traffic video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open traffic video!")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("AI Helmet Detection Started!")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO detection
    results = model(frame, conf=0.40, verbose=False)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Save processed frame
    out.write(annotated_frame)

    # Display video
    cv2.imshow("AI Helmet Detection - Traffic Video", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Detection completed!")
print("Saved result:", output_path)