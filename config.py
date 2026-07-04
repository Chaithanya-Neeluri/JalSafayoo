"""
JALSAFAYOO AI - Application Configuration
Intelligent Water Surface Floater Detection System
"""

import os
import sys
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Base Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best.pt"
VIDEOS_DIR = BASE_DIR / "videos"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
SNAPSHOTS_DIR = BASE_DIR / "snapshots"
LOGS_DIR = BASE_DIR / "logs"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


def ensure_directories() -> None:
    """Create all required runtime directories if they do not exist."""
    for directory in (
        VIDEOS_DIR,
        UPLOADS_DIR,
        OUTPUTS_DIR,
        SNAPSHOTS_DIR,
        LOGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Flask Configuration
# ---------------------------------------------------------------------------

class Config:
    """Flask application configuration."""

    SECRET_KEY = os.environ.get("SECRET_KEY", "jalsafayoo-ai-secret-key-2026")
    DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    TEMPLATES_AUTO_RELOAD = DEBUG

    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB upload limit

    ALLOWED_VIDEO_EXTENSIONS = frozenset({"mp4", "avi", "mov", "mkv"})

    # SocketIO — threading is more stable on Windows with OpenCV + PyTorch
    SOCKETIO_ASYNC_MODE = os.environ.get(
        "SOCKETIO_ASYNC_MODE",
        "threading" if sys.platform == "win32" else "eventlet",
    )
    SOCKETIO_CORS_ALLOWED_ORIGINS = "*"
    SOCKETIO_PING_TIMEOUT = 60
    SOCKETIO_PING_INTERVAL = 25


# ---------------------------------------------------------------------------
# YOLO / Detection Configuration
# ---------------------------------------------------------------------------

class DetectionConfig:
    """YOLOv8 inference and detection parameters."""

    MODEL_FILENAME = "best.pt"
    CLASS_NAME = "Floater"
    NUM_CLASSES = 1

    DEFAULT_CONFIDENCE = 0.4
    DEFAULT_IOU = 0.45
    IMAGE_SIZE = 640

    MIN_CONFIDENCE = 0.1
    MAX_CONFIDENCE = 0.95
    MIN_IOU = 0.1
    MAX_IOU = 0.9

    # Bounding box colors (BGR for OpenCV)
    BOX_COLOR = (77, 163, 255)       # Primary #4DA3FF
    BOX_THICKNESS = 2
    CENTER_RADIUS = 5
    CENTER_COLOR = (0, 212, 184)     # Secondary #00D4B8
    LABEL_BG_COLOR = (17, 28, 45)    # Card #111C2D
    LABEL_TEXT_COLOR = (248, 250, 252)


# ---------------------------------------------------------------------------
# Video Pipeline Configuration
# ---------------------------------------------------------------------------

class VideoConfig:
    """Video capture, streaming, and output settings."""

    MJPEG_QUALITY = 85
    STREAM_FPS_TARGET = 30
    CAPTURE_BUFFER_SIZE = 1

    # IP webcam reconnect
    RECONNECT_DELAY_SEC = 3.0
    MAX_RECONNECT_ATTEMPTS = 10
    WEBCAM_CONNECT_TIMEOUT = 10

    # Common IP Webcam (Android) stream paths — tried when URL has no path
    WEBCAM_PATH_SUFFIXES = (
        "/video",
        "/videofeed",
        "/mjpegfeed",
        "/stream",
        "/cam/realmonitor",
    )

    # Output video codec
    OUTPUT_CODEC = "mp4v"
    OUTPUT_EXTENSION = ".mp4"

    # Frame resize for display (None = native resolution)
    DISPLAY_MAX_WIDTH = 1280
    DISPLAY_MAX_HEIGHT = 720


# ---------------------------------------------------------------------------
# Dashboard / UI Configuration
# ---------------------------------------------------------------------------

class UIConfig:
    """Frontend-facing constants and labels."""

    PROJECT_NAME = "JALSAFAYOO AI"
    PROJECT_SUBTITLE = "Intelligent Water Surface Floater Detection System"
    PROJECT_VERSION = "1.0.0"

    # Color palette (CSS / metadata)
    COLORS = {
        "background": "#08111F",
        "card": "#111C2D",
        "primary": "#4DA3FF",
        "secondary": "#00D4B8",
        "success": "#22C55E",
        "warning": "#FACC15",
        "danger": "#EF4444",
        "text": "#F8FAFC",
        "muted": "#94A3B8",
        "border": "rgba(255,255,255,0.08)",
    }

    # Detection log limits
    MAX_LOG_ENTRIES = 500
    MAX_TABLE_ROWS = 50

    # Chart history points
    CHART_HISTORY_LENGTH = 60


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

class LogConfig:
    """Application logging settings."""

    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    @staticmethod
    def get_log_file() -> Path:
        """Return a timestamped log file path inside /logs."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return LOGS_DIR / f"jalsafayoo_{timestamp}.log"


# ---------------------------------------------------------------------------
# Input Source Types
# ---------------------------------------------------------------------------

class SourceType:
    """Enumeration of supported video input sources."""

    NONE = "none"
    IP_WEBCAM = "ip_webcam"
    DEMO_VIDEO = "demo_video"
    UPLOAD = "upload"

    LABELS = {
        NONE: "No Source",
        IP_WEBCAM: "Live IP Webcam",
        DEMO_VIDEO: "Demo Video",
        UPLOAD: "Uploaded Video",
    }


# ---------------------------------------------------------------------------
# Application State Defaults
# ---------------------------------------------------------------------------

class AppState:
    """Default runtime state values."""

    model_status = "Idle"
    camera_status = "Disconnected"
    input_source = SourceType.NONE
    is_running = False
    is_paused = False
    fps = 0.0
    inference_time_ms = 0.0
    total_floaters = 0
    current_detections = 0
    confidence_threshold = DetectionConfig.DEFAULT_CONFIDENCE
    iou_threshold = DetectionConfig.DEFAULT_IOU
