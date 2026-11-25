\# 🪖 AI Helmet Detection \& Tracking System



\### Real-Time Traffic Helmet Detection Using YOLOv8, ByteTrack and OpenCV



An AI-powered computer vision system designed to detect motorcycle riders with and without helmets in traffic videos. The system combines YOLOv8 object detection with ByteTrack tracking to analyze traffic footage, visualize detections, and generate structured CSV reports.



\## 🚀 Key Features



\- \*\*Helmet Detection:\*\* Identifies riders with helmets and without helmets.

\- \*\*Object Tracking:\*\* Uses ByteTrack to assign tracking IDs to detections.

\- \*\*Real-Time Visualization:\*\* Displays bounding boxes, class labels and confidence scores.

\- \*\*Detection Dashboard:\*\* Shows frame-wise helmet and no-helmet counts.

\- \*\*CSV Reporting:\*\* Records frame number, tracking ID, detected class and confidence.

\- \*\*Video Processing:\*\* Generates an annotated output video.



\## 🧠 Technologies Used



| Technology | Purpose |

|---|---|

| Python | Core programming |

| YOLOv8 | Object detection |

| ByteTrack | Object tracking |

| OpenCV | Video processing |

| Supervision | Detection annotation |

| PyTorch | Deep learning framework |



\## 📂 Project Structure



```text

HelmetDetection/

│

├── models/

│   └── best.pt

│

├── videos/

│   └── traffic.mp4

│

├── output/

│   ├── helmet\_detection\_v3.mp4

│   └── violation\_report.csv

│

├── detect.py

├── video\_detect.py

├── video\_detect\_tiled.py

├── video\_detect\_v3.py

│

├── requirements.txt

├── .gitignore

└── README.md

```



\## ⚙️ Installation



\### 1. Clone the repository



```bash

git clone https://github.com/YOUR\_USERNAME/HelmetDetection.git

cd HelmetDetection

```



\### 2. Create a virtual environment



```bash

python -m venv venv

```



\### 3. Activate the environment



Windows:



```bash

venv\\Scripts\\activate

```



\### 4. Install dependencies



```bash

pip install -r requirements.txt

```



\### 5. Add project files



Place your trained YOLO model at:



```text

models/best.pt

```



Place your traffic video at:



```text

videos/traffic.mp4

```



\### 6. Run the detection system



```bash

python video\_detect\_v3.py

```



\## 📊 Output



The system generates:



\*\*Annotated Video\*\*

```text

output/helmet\_detection\_v3.mp4

```



\*\*Detection Report\*\*

```text

output/violation\_report.csv

```



The CSV contains:



\- Frame

\- Track\_ID

\- Class

\- Confidence



\## 🔬 Detection Configuration



| Parameter | Value |

|---|---|

| Model | YOLOv8 |

| Confidence Threshold | 0.60 |

| Image Size | 640 |

| Tracker | ByteTrack |

| Input Format | MP4 |

| Output Format | MP4 |

| Classes | With Helmet / Without Helmet |



\## 📈 Project Results



Final recorded processing results:



| Metric | Result |

|---|---:|

| Frames Processed | 816 |

| Total Detection Records | 1,848 |

| With Helmet Records | 179 |

| Without Helmet Records | 1,669 |

| Average Detection Confidence | 80.05% |



\*These are frame-level detection records, not unique riders. Average confidence is not model accuracy.\*



\## 🔮 Future Improvements



\- Improved detection in crowded traffic scenes.

\- Rider-level tracking and violation counting.

\- License plate recognition.

\- Automatic violation alerts.

\- Live CCTV camera integration.

\- Web-based monitoring dashboard.



\## 👨‍💻 Author



\*\*Shankar Savalgi\*\*



Artificial Intelligence \& Machine Learning Student  

AI/ML Developer | Computer Vision Enthusiast



GitHub: https://github.com/Shankar-Savalgi



\---



\*Building intelligent solutions through Artificial Intelligence and Computer Vision.\*

