"""
JALSAFAYOO AI - Flask Routes & REST API
HTTP endpoints, MJPEG streaming, file downloads, and uploads.
"""

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from flask import (
    Blueprint,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
)

from config import (
    OUTPUTS_DIR,
    SNAPSHOTS_DIR,
    SourceType,
    UIConfig,
    VIDEOS_DIR,
    VideoConfig,
)
from utils import (
    clamp_confidence,
    clamp_iou,
    encode_frame_jpeg,
    get_date_string,
    get_time_string,
    list_demo_videos,
    list_output_files,
    save_uploaded_file,
)

if TYPE_CHECKING:
    from detector import FloaterDetector, VideoPipeline


logger = logging.getLogger("jalsafayoo.routes")

main_bp = Blueprint("main", __name__)

# Module-level references set during app initialization
_pipeline: "VideoPipeline | None" = None
_detector: "FloaterDetector | None" = None


def init_routes(pipeline: "VideoPipeline", detector: "FloaterDetector") -> Blueprint:
    """
    Bind pipeline and detector instances to route handlers.

    Args:
        pipeline: Active VideoPipeline instance.
        detector: Loaded FloaterDetector instance.

    Returns:
        Configured Blueprint.
    """
    global _pipeline, _detector
    _pipeline = pipeline
    _detector = detector
    return main_bp


def _error(message: str, status: int = 400):
    """Return a JSON error response."""
    return jsonify({"success": False, "error": message}), status


def _success(data: dict = None, status: int = 200):
    """Return a JSON success response."""
    payload = {"success": True}
    if data:
        payload.update(data)
    return jsonify(payload), status


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@main_bp.route("/")
def index():
    """Render the main dashboard."""
    return render_template(
        "index.html",
        project_name=UIConfig.PROJECT_NAME,
        project_subtitle=UIConfig.PROJECT_SUBTITLE,
        project_version=UIConfig.PROJECT_VERSION,
        colors=UIConfig.COLORS,
        current_date=get_date_string(),
        current_time=get_time_string(),
    )


# ---------------------------------------------------------------------------
# MJPEG Video Stream
# ---------------------------------------------------------------------------

@main_bp.route("/video_feed")
def video_feed():
    """MJPEG stream of the annotated detection feed."""

    def generate():
        while True:
            if _pipeline is None:
                time.sleep(0.1)
                continue

            frame = _pipeline.get_current_frame()
            if frame is not None:
                jpeg = encode_frame_jpeg(frame, VideoConfig.MJPEG_QUALITY)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            else:
                time.sleep(0.05)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Video Source Endpoints
# ---------------------------------------------------------------------------

@main_bp.route("/api/videos", methods=["GET"])
def api_list_videos():
    """List available demo videos in /videos."""
    videos = list_demo_videos()
    return _success({"videos": videos, "count": len(videos)})


@main_bp.route("/api/connect/webcam", methods=["POST"])
def api_connect_webcam():
    """Connect to a live IP webcam URL and start detection."""
    if _pipeline is None:
        return _error("Pipeline not initialized", 500)

    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()

    if not url:
        return _error("Webcam URL is required")

    if not url.startswith(("http://", "https://", "rtsp://")):
        return _error("Invalid URL format. Must start with http://, https://, or rtsp://")

    connected = _pipeline.connect_ip_webcam(url)
    if not connected:
        return _error(
            "Could not connect to webcam. Use the full stream URL "
            "(e.g. http://192.168.29.166:8080/video). "
            "Ensure your phone and PC are on the same Wi‑Fi and IP Webcam is running.",
            503,
        )

    _pipeline.start()
    logger.info("IP webcam connected and pipeline started: %s", url)
    return _success({
        "message": "Webcam connected",
        "url": url,
        "source": SourceType.IP_WEBCAM,
    })


@main_bp.route("/api/connect/demo", methods=["POST"])
def api_connect_demo():
    """Connect to a demo video and start detection."""
    if _pipeline is None:
        return _error("Pipeline not initialized", 500)

    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "").strip()

    if not filename:
        return _error("Video filename is required")

    filepath = VIDEOS_DIR / Path(filename).name
    if not filepath.exists():
        return _error(f"Video not found: {filename}", 404)

    connected = _pipeline.connect_demo_video(filepath)
    if not connected:
        return _error(f"Failed to open video: {filename}", 503)

    _pipeline.start()
    logger.info("Demo video started: %s", filename)
    return _success({
        "message": "Demo video playing",
        "filename": filename,
        "source": SourceType.DEMO_VIDEO,
    })


@main_bp.route("/api/upload", methods=["POST"])
def api_upload_video():
    """Upload a video file and start detection immediately."""
    if _pipeline is None:
        return _error("Pipeline not initialized", 500)

    if "video" not in request.files:
        return _error("No video file provided")

    file = request.files["video"]
    if not file or not file.filename:
        return _error("Empty file upload")

    try:
        filepath = save_uploaded_file(file)
    except ValueError as exc:
        return _error(str(exc))

    connected = _pipeline.connect_upload(filepath)
    if not connected:
        return _error("Failed to open uploaded video", 503)

    _pipeline.start()
    logger.info("Uploaded video started: %s", filepath.name)
    return _success({
        "message": "Upload successful, detection started",
        "filename": filepath.name,
        "source": SourceType.UPLOAD,
    })


# ---------------------------------------------------------------------------
# Playback Controls
# ---------------------------------------------------------------------------

@main_bp.route("/api/control/<action>", methods=["POST"])
def api_control(action: str):
    """Video playback controls: play, pause, resume, restart, stop."""
    if _pipeline is None:
        return _error("Pipeline not initialized", 500)

    action = action.lower()

    if action == "play":
        if not _pipeline.is_running:
            _pipeline.start()
        return _success({"action": "play", "is_running": _pipeline.is_running})

    if action == "pause":
        _pipeline.pause()
        return _success({"action": "pause", "is_paused": _pipeline.is_paused})

    if action == "resume":
        _pipeline.resume()
        return _success({"action": "resume", "is_paused": _pipeline.is_paused})

    if action == "restart":
        _pipeline.restart()
        if not _pipeline.is_running:
            _pipeline.start()
        return _success({"action": "restart"})

    if action == "stop":
        _pipeline.stop()
        return _success({"action": "stop"})

    return _error(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Snapshot & Output
# ---------------------------------------------------------------------------

@main_bp.route("/api/snapshot", methods=["POST"])
def api_snapshot():
    """Capture and save the current frame as a snapshot."""
    if _pipeline is None:
        return _error("Pipeline not initialized", 500)

    snapshot_path = _pipeline.capture_snapshot()
    if snapshot_path is None:
        return _error("No frame available to capture", 404)

    return _success({
        "message": "Snapshot saved",
        "filename": snapshot_path.name,
        "download_url": f"/api/download/snapshot/{snapshot_path.name}",
    })


@main_bp.route("/api/outputs", methods=["GET"])
def api_list_outputs():
    """List processed videos available for download."""
    outputs = list_output_files()
    return _success({"outputs": outputs, "count": len(outputs)})


@main_bp.route("/api/download/output/<filename>", methods=["GET"])
def api_download_output(filename: str):
    """Download a processed video from /outputs."""
    safe_name = Path(filename).name
    filepath = OUTPUTS_DIR / safe_name

    if not filepath.exists():
        return _error("Output file not found", 404)

    return send_file(
        filepath,
        as_attachment=True,
        download_name=safe_name,
        mimetype="video/mp4",
    )


@main_bp.route("/api/download/snapshot/<filename>", methods=["GET"])
def api_download_snapshot(filename: str):
    """Download a snapshot image from /snapshots."""
    safe_name = Path(filename).name
    filepath = SNAPSHOTS_DIR / safe_name

    if not filepath.exists():
        return _error("Snapshot not found", 404)

    return send_file(
        filepath,
        as_attachment=True,
        download_name=safe_name,
        mimetype="image/jpeg",
    )


@main_bp.route("/api/download/latest", methods=["GET"])
def api_download_latest_output():
    """Download the most recently processed video."""
    if _pipeline and _pipeline.last_output_path:
        filepath = _pipeline.last_output_path
        if filepath.exists():
            return send_file(
                filepath,
                as_attachment=True,
                download_name=filepath.name,
                mimetype="video/mp4",
            )

    outputs = list_output_files()
    if not outputs:
        return _error("No processed videos available", 404)

    filepath = OUTPUTS_DIR / outputs[0]["name"]
    return send_file(
        filepath,
        as_attachment=True,
        download_name=outputs[0]["name"],
        mimetype="video/mp4",
    )


# ---------------------------------------------------------------------------
# Detection Log & Export
# ---------------------------------------------------------------------------

@main_bp.route("/api/log", methods=["GET"])
def api_detection_log():
    """Return the current detection log entries."""
    if _pipeline is None:
        return _error("Pipeline not initialized", 500)

    return _success({
        "log": _pipeline.detection_log.to_dict_list(),
        "total_entries": len(_pipeline.detection_log.entries),
    })


@main_bp.route("/api/export/csv", methods=["GET"])
def api_export_csv():
    """Export detection log as a downloadable CSV file."""
    if _pipeline is None:
        return _error("Pipeline not initialized", 500)

    csv_path = _pipeline.detection_log.save_csv()
    return send_file(
        csv_path,
        as_attachment=True,
        download_name=csv_path.name,
        mimetype="text/csv",
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@main_bp.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    """Get or update detection settings (confidence, IOU)."""
    if _detector is None or _pipeline is None:
        return _error("Application not initialized", 500)

    if request.method == "GET":
        return _success({
            "confidence": _detector._confidence,
            "iou": _detector._iou,
        })

    data = request.get_json(silent=True) or {}

    if "confidence" in data:
        confidence = clamp_confidence(float(data["confidence"]))
        _detector._confidence = confidence

    if "iou" in data:
        iou = clamp_iou(float(data["iou"]))
        _detector._iou = iou

    _detector.set_thresholds(_detector._confidence, _detector._iou)

    logger.info(
        "Settings updated → conf=%.2f, iou=%.2f",
        _detector._confidence,
        _detector._iou,
    )

    return _success({
        "confidence": _detector._confidence,
        "iou": _detector._iou,
        "message": "Settings updated",
    })


@main_bp.route("/api/settings/reset", methods=["POST"])
def api_reset_settings():
    """Reset detection settings to defaults."""
    if _detector is None:
        return _error("Detector not initialized", 500)

    from config import DetectionConfig

    _detector.set_thresholds(
        DetectionConfig.DEFAULT_CONFIDENCE,
        DetectionConfig.DEFAULT_IOU,
    )

    return _success({
        "confidence": _detector._confidence,
        "iou": _detector._iou,
        "message": "Settings reset to defaults",
    })


# ---------------------------------------------------------------------------
# Status & Health
# ---------------------------------------------------------------------------

@main_bp.route("/api/status", methods=["GET"])
def api_status():
    """Return current application and pipeline status."""
    if _pipeline is None or _detector is None:
        return _error("Application not initialized", 500)

    from utils import build_stats_payload

    stats = build_stats_payload(
        fps=_pipeline.fps_counter.tick() if _pipeline.is_running else 0.0,
        total_floaters=_pipeline.detection_counter.total,
        current_detections=_pipeline.detection_counter.current,
        model_status=_detector.status,
        camera_status=_pipeline.camera_status,
        input_source=_pipeline.source_type,
        confidence=_detector._confidence,
        iou=_detector._iou,
        inference_time_ms=0.0,
    )

    return _success({
        **stats,
        "is_running": _pipeline.is_running,
        "is_paused": _pipeline.is_paused,
        "project_name": UIConfig.PROJECT_NAME,
        "project_version": UIConfig.PROJECT_VERSION,
    })


@main_bp.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint."""
    model_loaded = _detector is not None and _detector.is_loaded
    return _success({
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_status": _detector.status if _detector else "Not Loaded",
    })
