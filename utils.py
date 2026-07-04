"""
JALSAFAYOO AI - Utility Functions
Logging, file handling, CSV export, and shared helpers.
"""

import csv
import io
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from werkzeug.utils import secure_filename

from config import (
    Config,
    DetectionConfig,
    LogConfig,
    LOGS_DIR,
    UIConfig,
    VIDEOS_DIR,
    UPLOADS_DIR,
    OUTPUTS_DIR,
    SNAPSHOTS_DIR,
    VideoConfig,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """
    Configure application-wide logging to console and file.

    Args:
        log_file: Optional path for log file. Uses LogConfig default if None.

    Returns:
        Configured root logger for the application.
    """
    logger = logging.getLogger("jalsafayoo")
    logger.setLevel(getattr(logging, LogConfig.LOG_LEVEL.upper(), logging.INFO))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        LogConfig.LOG_FORMAT,
        datefmt=LogConfig.LOG_DATE_FORMAT,
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is None:
        log_file = LogConfig.get_log_file()

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logging initialized → %s", log_file)
    return logger


# ---------------------------------------------------------------------------
# Detection Data Model
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single floater detection result."""

    detection_id: int
    center_x: int
    center_y: int
    width: int
    height: int
    area: int
    confidence: float
    timestamp: str
    class_name: str = DetectionConfig.CLASS_NAME

    @classmethod
    def from_bbox(
        cls,
        detection_id: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        confidence: float,
        timestamp: Optional[str] = None,
    ) -> "Detection":
        """Build a Detection from bounding box coordinates."""
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        center_x = x1 + width // 2
        center_y = y1 + height // 2
        area = width * height

        if timestamp is None:
            timestamp = get_timestamp()

        return cls(
            detection_id=detection_id,
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
            area=area,
            confidence=round(confidence, 4),
            timestamp=timestamp,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize detection for JSON / SocketIO emission."""
        return asdict(self)


@dataclass
class DetectionLog:
    """Rolling detection history with CSV export support."""

    entries: List[Detection] = field(default_factory=list)
    max_entries: int = UIConfig.MAX_LOG_ENTRIES

    def add(self, detection: Detection) -> None:
        """Prepend a detection (newest first)."""
        self.entries.insert(0, detection)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[: self.max_entries]

    def add_batch(self, detections: List[Detection]) -> None:
        """Prepend multiple detections from a single frame."""
        for detection in reversed(detections):
            self.add(detection)

    def clear(self) -> None:
        """Remove all log entries."""
        self.entries.clear()

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Return all entries as dictionaries."""
        return [entry.to_dict() for entry in self.entries]

    def export_csv(self) -> str:
        """
        Export log entries as CSV string.

        Returns:
            CSV content as a UTF-8 string.
        """
        output = io.StringIO()
        fieldnames = [
            "detection_id",
            "center_x",
            "center_y",
            "width",
            "height",
            "area",
            "confidence",
            "timestamp",
            "class_name",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for entry in self.entries:
            writer.writerow(entry.to_dict())
        return output.getvalue()

    def save_csv(self, filepath: Optional[Path] = None) -> Path:
        """
        Write log entries to a CSV file.

        Args:
            filepath: Target path. Auto-generated if None.

        Returns:
            Path to the saved CSV file.
        """
        if filepath is None:
            filepath = LOGS_DIR / f"detections_{generate_filename_suffix()}.csv"

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as file:
            file.write(self.export_csv())

        return filepath


# ---------------------------------------------------------------------------
# Time & Formatting
# ---------------------------------------------------------------------------

def get_timestamp(fmt: str = "%Y-%m-%d %H:%M:%S.%f") -> str:
    """Return current local timestamp as formatted string."""
    return datetime.now().strftime(fmt)[:-3]  # Trim to milliseconds


def get_date_string() -> str:
    """Return current date for navbar display."""
    return datetime.now().strftime("%A, %B %d, %Y")


def get_time_string() -> str:
    """Return current time for navbar display."""
    return datetime.now().strftime("%H:%M:%S")


def generate_filename_suffix() -> str:
    """Generate a unique timestamp-based suffix for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:6]}"


def get_webcam_url_candidates(url: str) -> List[str]:
    """
    Build a list of URL candidates for IP webcam connection.

    Android IP Webcam and similar apps serve MJPEG at paths like /video,
    not at the root URL. This expands bare host:port URLs automatically.

    Args:
        url: User-provided webcam URL.

    Returns:
        Ordered unique list of URLs to attempt.
    """
    from urllib.parse import urlparse, urlunparse

    url = url.strip().rstrip("/")
    candidates: List[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add(url)

    parsed = urlparse(url)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        base = urlunparse((parsed.scheme, parsed.netloc, "", "", "", "")).rstrip("/")
        path = parsed.path or ""

        if not path or path == "/":
            for suffix in VideoConfig.WEBCAM_PATH_SUFFIXES:
                add(f"{base}{suffix}")
        elif path not in VideoConfig.WEBCAM_PATH_SUFFIXES:
            add(f"{base}/video")

    return candidates


# ---------------------------------------------------------------------------
# File Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    """Check whether filename has an allowed video extension."""
    if not filename or "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in Config.ALLOWED_VIDEO_EXTENSIONS


def save_uploaded_file(file_storage, destination_dir: Path = UPLOADS_DIR) -> Path:
    """
    Securely save an uploaded video file.

    Args:
        file_storage: Werkzeug FileStorage object.
        destination_dir: Directory to save the file.

    Returns:
        Path to the saved file.

    Raises:
        ValueError: If file extension is not allowed.
    """
    original_name = secure_filename(file_storage.filename or "upload.mp4")
    if not allowed_file(original_name):
        raise ValueError(
            f"Unsupported file type. Allowed: {', '.join(Config.ALLOWED_VIDEO_EXTENSIONS)}"
        )

    destination_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(original_name).stem
    extension = Path(original_name).suffix
    unique_name = f"{stem}_{generate_filename_suffix()}{extension}"
    filepath = destination_dir / unique_name
    file_storage.save(str(filepath))
    return filepath


def list_demo_videos() -> List[Dict[str, str]]:
    """
    Scan /videos folder and return available demo videos.

    Returns:
        List of dicts with 'name' and 'path' keys.
    """
    videos = []
    if not VIDEOS_DIR.exists():
        return videos

    for filepath in sorted(VIDEOS_DIR.iterdir()):
        if filepath.is_file() and allowed_file(filepath.name):
            videos.append({
                "name": filepath.name,
                "path": str(filepath),
                "size_mb": round(filepath.stat().st_size / (1024 * 1024), 2),
            })
    return videos


def get_output_path(prefix: str = "processed") -> Path:
    """Generate a unique output video path in /outputs."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{generate_filename_suffix()}.mp4"
    return OUTPUTS_DIR / filename


def get_snapshot_path(prefix: str = "snapshot") -> Path:
    """Generate a unique snapshot path in /snapshots."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{generate_filename_suffix()}.jpg"
    return SNAPSHOTS_DIR / filename


def list_output_files() -> List[Dict[str, str]]:
    """Return processed videos available for download."""
    outputs = []
    if not OUTPUTS_DIR.exists():
        return outputs

    for filepath in sorted(OUTPUTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if filepath.is_file() and filepath.suffix.lower() in (".mp4", ".avi", ".mkv"):
            outputs.append({
                "name": filepath.name,
                "path": str(filepath),
                "size_mb": round(filepath.stat().st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(filepath.stat().st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            })
    return outputs


# ---------------------------------------------------------------------------
# Frame Processing
# ---------------------------------------------------------------------------

def resize_frame(
    frame: np.ndarray,
    max_width: int = 1280,
    max_height: int = 720,
) -> np.ndarray:
    """
    Resize frame maintaining aspect ratio if it exceeds max dimensions.

    Args:
        frame: Input BGR frame.
        max_width: Maximum display width.
        max_height: Maximum display height.

    Returns:
        Resized frame (or original if within limits).
    """
    height, width = frame.shape[:2]
    if width <= max_width and height <= max_height:
        return frame

    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def encode_frame_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """
    Encode a BGR frame as JPEG bytes for MJPEG streaming.

    Args:
        frame: BGR numpy array.
        quality: JPEG compression quality (0-100).

    Returns:
        JPEG-encoded bytes.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise RuntimeError("Failed to encode frame as JPEG")
    return buffer.tobytes()


def draw_detection_overlay(
    frame: np.ndarray,
    detections: List[Detection],
    show_labels: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes, center points, and labels on a frame.

    Args:
        frame: BGR frame to annotate (modified in place).
        detections: List of Detection objects.
        show_labels: Whether to draw confidence labels.

    Returns:
        Annotated frame.
    """
    annotated = frame.copy()
    cfg = DetectionConfig

    for det in detections:
        half_w = det.width // 2
        half_h = det.height // 2
        x1 = det.center_x - half_w
        y1 = det.center_y - half_h
        x2 = det.center_x + half_w
        y2 = det.center_y + half_h

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            cfg.BOX_COLOR,
            cfg.BOX_THICKNESS,
        )

        cv2.circle(
            annotated,
            (det.center_x, det.center_y),
            cfg.CENTER_RADIUS,
            cfg.CENTER_COLOR,
            -1,
        )

        if show_labels:
            label = f"#{det.detection_id} {det.class_name} {det.confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_y1 = max(y1 - text_h - 8, 0)
            cv2.rectangle(
                annotated,
                (x1, label_y1),
                (x1 + text_w + 8, label_y1 + text_h + 8),
                cfg.LABEL_BG_COLOR,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, label_y1 + text_h + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                cfg.LABEL_TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

    return annotated


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------

class FPSCounter:
    """Calculate rolling frames-per-second."""

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: List[float] = []
        self._window_size = window_size

    def tick(self) -> float:
        """
        Record a frame timestamp and return current FPS.

        Returns:
            Current FPS based on recent frame times.
        """
        now = time.perf_counter()
        self._timestamps.append(now)

        if len(self._timestamps) > self._window_size:
            self._timestamps.pop(0)

        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0

        return round((len(self._timestamps) - 1) / elapsed, 1)

    def reset(self) -> None:
        """Clear timestamp history."""
        self._timestamps.clear()


class InferenceTimer:
    """Measure inference duration in milliseconds."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.last_ms: float = 0.0

    def start(self) -> None:
        """Begin timing."""
        self._start = time.perf_counter()

    def stop(self) -> float:
        """
        End timing and return elapsed milliseconds.

        Returns:
            Inference time in milliseconds.
        """
        self.last_ms = round((time.perf_counter() - self._start) * 1000, 2)
        return self.last_ms


class DetectionCounter:
    """Track cumulative and per-frame detection counts."""

    def __init__(self) -> None:
        self.total: int = 0
        self.current: int = 0
        self._next_id: int = 1

    def update(self, count: int) -> None:
        """Update counts after a frame's detections."""
        self.current = count
        self.total += count

    def next_id(self) -> int:
        """Return and increment the next detection ID."""
        current_id = self._next_id
        self._next_id += 1
        return current_id

    def reset(self) -> None:
        """Reset all counters."""
        self.total = 0
        self.current = 0
        self._next_id = 1


# ---------------------------------------------------------------------------
# Settings Validation
# ---------------------------------------------------------------------------

def clamp_confidence(value: float) -> float:
    """Clamp confidence threshold to valid range."""
    return round(
        max(DetectionConfig.MIN_CONFIDENCE, min(DetectionConfig.MAX_CONFIDENCE, value)),
        2,
    )


def clamp_iou(value: float) -> float:
    """Clamp IOU threshold to valid range."""
    return round(
        max(DetectionConfig.MIN_IOU, min(DetectionConfig.MAX_IOU, value)),
        2,
    )


def build_stats_payload(
    fps: float,
    total_floaters: int,
    current_detections: int,
    model_status: str,
    camera_status: str,
    input_source: str,
    confidence: float,
    iou: float,
    inference_time_ms: float,
) -> Dict[str, Any]:
    """
    Build a statistics dictionary for SocketIO broadcast.

    Returns:
        Dashboard stats payload.
    """
    return {
        "fps": fps,
        "total_floaters": total_floaters,
        "current_detections": current_detections,
        "model_status": model_status,
        "camera_status": camera_status,
        "input_source": input_source,
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "inference_time_ms": inference_time_ms,
        "current_time": get_time_string(),
        "current_date": get_date_string(),
    }
