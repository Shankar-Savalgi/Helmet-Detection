import cv2
from ultralytics import YOLO

# Load trained helmet detection model
model = YOLO("models/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not detected!")
    exit()

print("Helmet Detection Started!")
print("Press Q to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform helmet detection
    results = model(frame, conf=0.40)

    # Draw detection boxes
    annotated_frame = results[0].plot()

    # Display output
    cv2.imshow("AI Helmet Detection System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()