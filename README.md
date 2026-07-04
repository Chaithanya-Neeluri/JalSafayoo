# JALSAFAYOO AI

**Intelligent Water Surface Floater Detection System**

A production-quality AI dashboard for real-time detection of floating waste on water surfaces using YOLOv8. Built as a final-year project demonstration with a premium UI inspired by enterprise AI platforms.

![Python](https://img.shields.io/badge/Python-3.11+-4DA3FF?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-00D4B8?style=flat-square&logo=flask&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-22C55E?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-FACC15?style=flat-square)

---

## Overview

JALSAFAYOO AI demonstrates a single-class object detection model trained with YOLOv8 to identify **Floaters** — floating waste on water surfaces. The application provides:

- Live detection from **IP webcams**, **demo videos**, or **uploaded files**
- Real-time dashboard with statistics, detection table, and scrollable log
- MJPEG video streaming with bounding boxes, confidence scores, and center points
- Automatic processed video recording and snapshot capture
- CSV export of detection history
- Socket.IO powered live updates

> **Note:** This is a software demonstration only. No hardware integration is required.

---

## Screenshots

| Dashboard | Detection Feed |
|-----------|----------------|
| ![Dashboard Placeholder](static/images/screenshot-dashboard.png) | ![Detection Placeholder](static/images/screenshot-detection.png) |

| Statistics Panel | Detection Log |
|--------------------|---------------|
| ![Stats Placeholder](static/images/screenshot-stats.png) | ![Log Placeholder](static/images/screenshot-log.png) |

*Replace placeholder images in `static/images/` with actual screenshots before submission.*

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.11 or higher |
| pip | Latest recommended |
| Web browser | Chrome, Firefox, or Edge (latest) |
| GPU (optional) | CUDA-capable GPU for faster inference |

### Python Dependencies

All dependencies are listed in `requirements.txt`:

- **Flask** — Web framework
- **Flask-SocketIO** — Real-time communication
- **Ultralytics** — YOLOv8 inference
- **OpenCV** — Video capture and processing
- **NumPy / Pandas** — Data handling
- **Eventlet** — Async server for Socket.IO

---

## Installation

### 1. Clone or download the project

```bash
cd jalsafayoo-ai
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the YOLO model

Copy your trained model file to the project root:

```
jalsafayoo-ai/
└── best.pt          ← Required
```

### 5. Add demo videos (optional)

Place sample videos in the `videos/` folder:

```
jalsafayoo-ai/
└── videos/
    ├── sample1.mp4
    └── sample2.mp4
```

Supported formats: **MP4**, **AVI**, **MOV**, **MKV**

---

## Running the Application

### Start the server

```bash
python app.py
```

### Open the dashboard

Navigate to:

```
http://localhost:5000
```

### Environment variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `5000` | Server port |
| `FLASK_DEBUG` | `false` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `SECRET_KEY` | (built-in) | Flask session secret |

Example:

```bash
set PORT=8080
set FLASK_DEBUG=true
python app.py
```

---

## Usage Guide

### Input Sources

#### 1. Live IP Webcam

1. Open the **IP Webcam** tab under Input Source
2. Enter the stream URL (e.g. `http://192.168.29.166:8080/video`)
3. Click **Connect**

#### 2. Demo Videos

1. Open the **Demo Videos** tab
2. Select a video from the dropdown
3. Click **Play**

#### 3. Upload Video

1. Open the **Upload** tab
2. Drag and drop a video file or click to browse
3. Detection starts automatically after upload

### Video Controls

| Control | Action | Shortcut |
|---------|--------|----------|
| Play | Start detection | `P` |
| Pause | Pause processing | `Space` |
| Resume | Resume processing | `R` |
| Restart | Restart from beginning | — |
| Stop | Stop and save output | `S` |
| Fullscreen | Toggle fullscreen | `F` |
| Screenshot | Capture current frame | `C` |
| Download | Download processed video | `D` |

### Settings

- **Confidence Threshold** — Minimum detection confidence (default: 0.40)
- **IOU Threshold** — Non-max suppression overlap (default: 0.45)
- **Theme** — Toggle dark/light mode
- **Reset** — Restore default settings

### Export

Click **Export CSV** in the Detection Log panel (or press `E`) to download the full detection history.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard page |
| `GET` | `/video_feed` | MJPEG stream |
| `GET` | `/api/videos` | List demo videos |
| `POST` | `/api/connect/webcam` | Connect IP webcam |
| `POST` | `/api/connect/demo` | Play demo video |
| `POST` | `/api/upload` | Upload video file |
| `POST` | `/api/control/{action}` | Playback control |
| `POST` | `/api/snapshot` | Capture snapshot |
| `GET` | `/api/download/latest` | Download latest output |
| `GET` | `/api/export/csv` | Export detection log |
| `GET/POST` | `/api/settings` | Get/update settings |
| `GET` | `/api/status` | Application status |
| `GET` | `/api/health` | Health check |

### Socket.IO Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `stats_update` | Server → Client | FPS, counts, status |
| `detections_update` | Server → Client | Table and log data |
| `status_update` | Server → Client | Pipeline state |
| `output_ready` | Server → Client | Processed video saved |

---

## Folder Structure

```
jalsafayoo-ai/
├── app.py                  # Main entry point (Flask + SocketIO)
├── config.py               # Application configuration
├── detector.py             # YOLOv8 inference engine
├── routes.py               # REST API routes
├── utils.py                # Helpers, logging, CSV export
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── best.pt                 # YOLOv8 trained model (you provide)
│
├── videos/                 # Built-in demo videos
├── uploads/                # Browser-uploaded videos
├── outputs/                # Auto-saved processed videos
├── snapshots/              # Captured frame screenshots
├── logs/                   # Application and CSV logs
│
├── templates/
│   └── index.html          # Dashboard HTML template
│
└── static/
    ├── css/
    │   └── style.css       # Premium dashboard styles
    ├── js/
    │   └── app.js          # Client-side application
    └── images/
        └── (screenshots)   # Project screenshots
```

Runtime folders (`videos/`, `uploads/`, `outputs/`, `snapshots/`, `logs/`) are created automatically on first launch.

---

## Detection Model

| Property | Value |
|----------|-------|
| Framework | Ultralytics YOLOv8 |
| Model file | `best.pt` |
| Classes | 1 — **Floater** |
| Confidence | 0.4 (configurable) |
| Image size | 640 |
| Inference | `model.predict(frame, conf=0.4, imgsz=640)` |

### Per-Detection Output

Each detected floater includes:

- Bounding box with label
- Confidence score
- Center point (X, Y)
- Width, height, and area
- Unique detection ID
- Timestamp

---

## Architecture

```
┌─────────────┐     Socket.IO      ┌──────────────┐
│   Browser   │◄──────────────────►│    app.py    │
│  Dashboard  │     REST API       │ Flask+Socket │
└──────┬──────┘                    └──────┬───────┘
       │ MJPEG                            │
       ▼                                    ▼
┌─────────────┐                    ┌──────────────┐
│ video_feed  │                    │ VideoPipeline│
└─────────────┘                    │  (thread)    │
                                   └──────┬───────┘
                                          │
                                   ┌──────▼───────┐
                                   │FloaterDetector│
                                   │   YOLOv8     │
                                   └──────────────┘
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Model not found` | Place `best.pt` in the project root |
| Webcam won't connect | Verify URL is accessible in a browser first |
| Low FPS | Use GPU with CUDA, or reduce input resolution |
| Upload fails | Check file format (MP4/AVI/MOV/MKV) and size (< 500 MB) |
| Port in use | Set `PORT=8080` environment variable |

---

## Future Work

- [ ] Multi-class waste detection (bottles, plastics, organic matter)
- [ ] GPS geotagging for detected floaters on a map dashboard
- [ ] Edge deployment on NVIDIA Jetson for on-device inference
- [ ] Drone and boat-mounted camera integration
- [ ] Automated alert system (email/SMS) for high-density detections
- [ ] Historical analytics with date-range filtering
- [ ] User authentication and role-based access
- [ ] Docker containerization for one-command deployment
- [ ] Model versioning and A/B testing interface
- [ ] Integration with municipal water management systems

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, Flask, Flask-SocketIO |
| AI / CV | Ultralytics YOLOv8, OpenCV |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| UI | Bootstrap 5, Bootstrap Icons |
| Charts | Chart.js |
| Real-time | Socket.IO |

---

## License

This project is developed for academic purposes as a final-year engineering project demonstration.

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [Flask](https://flask.palletsprojects.com/) and [Flask-SocketIO](https://flask-socketio.readthedocs.io/) for the web stack
- [Chart.js](https://www.chartjs.org/) for live analytics visualization

---

**JALSAFAYOO AI** — Intelligent Water Surface Floater Detection System
