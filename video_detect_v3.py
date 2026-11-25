import cv2
import csv
import os
from ultralytics import YOLO
import supervision as sv
from trackers import ByteTrackTracker

# =====================================
# 1. CONFIGURATION
# =====================================

MODEL_PATH = "models/best.pt"
VIDEO_PATH = "videos/traffic.mp4"

OUTPUT_PATH = "output/helmet_detection_v3.mp4"
CSV_PATH = "output/violation_report.csv"

CONFIDENCE = 0.60
IMAGE_SIZE = 640

os.makedirs("output", exist_ok=True)

# =====================================
# 2. LOAD MODEL
# =====================================

print("Loading YOLOv8 model...")

model = YOLO(MODEL_PATH)

print("Model loaded:", model.names)

# =====================================
# 3. OPEN VIDEO
# =====================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit("ERROR: Cannot open traffic video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25

print("Video resolution:", width, "x", height)
print("FPS:", fps)

# =====================================
# 4. VIDEO WRITER
# =====================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    cap.release()
    raise SystemExit("ERROR: Cannot create output video.")

# =====================================
# 5. BYTETRACK INITIALIZATION
# =====================================

tracker = ByteTrackTracker(
    track_activation_threshold=0.25,
    minimum_consecutive_frames=1
)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# =====================================
# 6. CSV REPORT
# =====================================

csv_file = open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(csv_file)

writer.writerow([
    "Frame",
    "Track_ID",
    "Class",
    "Confidence"
])

# =====================================
# 7. PROCESS VIDEO
# =====================================

frame_number = 0

print("\nHELMET DETECTION V3 STARTED")

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # YOLO prediction
        results = model.predict(
            frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False
        )[0]

        # Convert YOLO results
        detections = sv.Detections.from_ultralytics(
            results
        )

        # ByteTrack tracking
        detections = tracker.update(detections)

        helmet_count = 0
        no_helmet_count = 0

        labels = []

        # Process detections
        for i in range(len(detections)):

            class_id = int(detections.class_id[i])

            confidence = float(
                detections.confidence[i]
            )

            track_id = int(
                detections.tracker_id[i]
            )

            class_name = model.names[class_id]

            if class_id == 0:

                helmet_count += 1

            elif class_id == 1:

                no_helmet_count += 1

            labels.append(
                f"ID {track_id} | "
                f"{class_name} | "
                f"{confidence:.2f}"
            )

            writer.writerow([
                frame_number,
                track_id,
                class_name,
                round(confidence, 3)
            ])

        # Draw bounding boxes
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections
        )

        # Draw labels
        frame = label_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        # Dashboard
        cv2.putText(
            frame,
            f"With Helmet: {helmet_count}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Without Helmet: {no_helmet_count}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Save output
        out.write(frame)

        # Display resized preview
        display = cv2.resize(
            frame,
            (1280, 720)
        )

        cv2.imshow(
            "AI Helmet Detection V3 - ByteTrack",
            display
        )

        if frame_number % 30 == 0:

            print(
                f"Frame: {frame_number} | "
                f"With Helmet: {helmet_count} | "
                f"Without Helmet: {no_helmet_count}"
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Processing stopped by user.")
            break

finally:

    cap.release()
    out.release()
    csv_file.close()

    cv2.destroyAllWindows()

print("\n================================")
print("HELMET DETECTION V3 FINISHED")
print("================================")

print("Output video:", OUTPUT_PATH)
print("CSV report:", CSV_PATH)