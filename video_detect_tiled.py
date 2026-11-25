import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("models/best.pt")

# Video paths
video_path = "videos/traffic.mp4"
output_path = "output/helmet_detection_tiled.mp4"

# Settings
CONFIDENCE = 0.15
IMAGE_SIZE = 640
OVERLAP = 0.15

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video!")
    exit()

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25

# Output writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Class names
names = model.names

print("TILE-BASED HELMET DETECTION STARTED")
print("Video resolution:", width, "x", height)

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Divide frame into 4 overlapping tiles
    mid_x = width // 2
    mid_y = height // 2

    overlap_x = int(mid_x * OVERLAP)
    overlap_y = int(mid_y * OVERLAP)

    tiles = [
        (0, 0, mid_x + overlap_x, mid_y + overlap_y),
        (mid_x - overlap_x, 0, width, mid_y + overlap_y),
        (0, mid_y - overlap_y, mid_x + overlap_x, height),
        (mid_x - overlap_x, mid_y - overlap_y, width, height)
    ]

    all_boxes = []

    # Run YOLO on each tile
    for x1, y1, x2, y2 in tiles:

        tile = frame[y1:y2, x1:x2]

        results = model.predict(
            tile,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False
        )

        for box in results[0].boxes:

            bx1, by1, bx2, by2 = box.xyxy[0].cpu().tolist()

            # Convert tile coordinates to full-frame coordinates
            bx1 += x1
            bx2 += x1
            by1 += y1
            by2 += y1

            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            all_boxes.append([
                bx1, by1, bx2, by2,
                confidence, class_id
            ])

    # Remove duplicate boxes class by class
    final_boxes = []

    for class_id in names:

        class_boxes = [
            b for b in all_boxes if b[5] == class_id
        ]

        if not class_boxes:
            continue

        boxes_xywh = []
        scores = []

        for b in class_boxes:
            x1, y1, x2, y2 = b[:4]

            boxes_xywh.append([
                int(x1), int(y1),
                int(x2 - x1), int(y2 - y1)
            ])

            scores.append(b[4])

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores,
            CONFIDENCE,
            0.45
        )

        if len(indices) > 0:
            for index in indices.flatten():
                final_boxes.append(class_boxes[index])

    # Detection counts for current frame
    helmet_count = 0
    no_helmet_count = 0

    # Draw final detections
    for b in final_boxes:

        x1, y1, x2, y2 = map(int, b[:4])
        confidence = b[4]
        class_id = b[5]

        if class_id == 0:
            label = "With Helmet"
            color = (0, 255, 0)
            helmet_count += 1

        else:
            label = "Without Helmet"
            color = (0, 0, 255)
            no_helmet_count += 1

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        text = f"{label} {confidence:.2f}"

        cv2.putText(
            frame,
            text,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # Display detection counts
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

    # Save processed frame
    out.write(frame)

    # Display result
    display_frame = cv2.resize(frame, (1280, 720))

    cv2.imshow(
        "AI Helmet Detection - Tiled YOLOv8",
        display_frame
    )

    if frame_number % 30 == 0:
        print(
            f"Frame {frame_number} | "
            f"With Helmet: {helmet_count} | "
            f"Without Helmet: {no_helmet_count}"
        )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("\nDETECTION FINISHED!")
print("Output saved:", output_path)