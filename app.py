"""
JALSAFAYOO AI - Main Application Entry Point
Flask + SocketIO server with real-time detection dashboard.
"""

import logging
import os

from config import Config, UIConfig, ensure_directories

if Config.SOCKETIO_ASYNC_MODE == "eventlet":
    import eventlet
    eventlet.monkey_patch()

from flask import Flask
from flask_socketio import SocketIO, emit

from detector import FloaterDetector, VideoPipeline
from routes import init_routes, main_bp
from utils import setup_logging


logger = logging.getLogger("jalsafayoo")

socketio = SocketIO(
    cors_allowed_origins=Config.SOCKETIO_CORS_ALLOWED_ORIGINS,
    async_mode=Config.SOCKETIO_ASYNC_MODE,
    ping_timeout=Config.SOCKETIO_PING_TIMEOUT,
    ping_interval=Config.SOCKETIO_PING_INTERVAL,
)

detector = FloaterDetector()
pipeline: VideoPipeline | None = None


def create_app() -> Flask:
    """
    Application factory — configure Flask, load model, register routes.

    Returns:
        Configured Flask application instance.
    """
    global pipeline

    ensure_directories()
    setup_logging()

    app = Flask(__name__)
    app.config.from_object(Config)

    # ------------------------------------------------------------------
    # SocketIO callbacks — broadcast pipeline events to all clients
    # ------------------------------------------------------------------

    def on_stats(payload: dict) -> None:
        socketio.emit("stats_update", payload)

    def on_detections(current: list, log: list) -> None:
        socketio.emit("detections_update", {
            "current": current,
            "log": log,
        })

    def on_status(payload: dict) -> None:
        socketio.emit("status_update", payload)

    def on_output_ready(filepath: str) -> None:
        filename = os.path.basename(filepath)
        socketio.emit("output_ready", {
            "filename": filename,
            "download_url": f"/api/download/output/{filename}",
            "message": "Processed video saved",
        })

    # ------------------------------------------------------------------
    # Initialize detection pipeline
    # ------------------------------------------------------------------

    pipeline = VideoPipeline(
        detector=detector,
        on_stats=on_stats,
        on_detections=on_detections,
        on_status=on_status,
        on_output_ready=on_output_ready,
    )

    init_routes(pipeline, detector)
    app.register_blueprint(main_bp)

    socketio.init_app(app)

    logger.info("%s v%s initialized", UIConfig.PROJECT_NAME, UIConfig.PROJECT_VERSION)
    return app


# ---------------------------------------------------------------------------
# SocketIO Event Handlers
# ---------------------------------------------------------------------------

@socketio.on("connect")
def handle_connect():
    """Client connected — send current state snapshot."""
    logger.info("Client connected")

    emit("connected", {
        "message": f"Welcome to {UIConfig.PROJECT_NAME}",
        "version": UIConfig.PROJECT_VERSION,
    })

    if pipeline is not None:
        emit("status_update", {
            "is_running": pipeline.is_running,
            "is_paused": pipeline.is_paused,
            "camera_status": pipeline.camera_status,
            "input_source": pipeline.source_type,
            "model_status": detector.status,
        })

        emit("detections_update", {
            "current": [],
            "log": pipeline.detection_log.to_dict_list(),
        })


@socketio.on("disconnect")
def handle_disconnect():
    """Client disconnected."""
    logger.info("Client disconnected")


@socketio.on("request_status")
def handle_request_status():
    """Client requests a fresh status update."""
    if pipeline is None:
        return

    from utils import build_stats_payload

    emit("stats_update", build_stats_payload(
        fps=0.0,
        total_floaters=pipeline.detection_counter.total,
        current_detections=pipeline.detection_counter.current,
        model_status=detector.status,
        camera_status=pipeline.camera_status,
        input_source=pipeline.source_type,
        confidence=detector._confidence,
        iou=detector._iou,
        inference_time_ms=0.0,
    ))


@socketio.on("request_log")
def handle_request_log():
    """Client requests the full detection log."""
    if pipeline is None:
        return

    emit("detections_update", {
        "current": [],
        "log": pipeline.detection_log.to_dict_list(),
    })


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load model and start the development server."""
    app = create_app()

    try:
        detector.load()
        logger.info("Model loaded — server ready")
    except FileNotFoundError:
        logger.warning(
            "Model file (best.pt) not found. Place it in the project root before running inference."
        )
    except Exception:
        logger.exception("Failed to load model — server starting in degraded mode")

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = Config.DEBUG

    print()
    print("=" * 60)
    print(f"  {UIConfig.PROJECT_NAME}")
    print(f"  {UIConfig.PROJECT_SUBTITLE}")
    print("=" * 60)
    print(f"  Dashboard  →  http://127.0.0.1:{port}")
    print(f"  (Use http:// — NOT https://)")
    print(f"  Model      →  {detector.status}")
    print(f"  Debug      →  {debug}")
    print(f"  Async mode →  {Config.SOCKETIO_ASYNC_MODE}")
    print("=" * 60)
    print()
    print("  Keep this terminal open while using the dashboard.")
    print()

    try:
        socketio.run(
            app,
            host=host,
            port=port,
            debug=debug,
            allow_unsafe_werkzeug=True,
        )
    except OSError as exc:
        if getattr(exc, "winerror", None) == 10048 or exc.errno in (98, 10048):
            print(f"ERROR: Port {port} is already in use.")
            print()
            print("Another JALSAFAYOO AI instance (or another app) is running on this port.")
            print("Fix options:")
            print(f"  1. Open the dashboard: http://localhost:{port}")
            print("  2. Stop the other process, then run python app.py again")
            print(f"  3. Use a different port: set PORT=5001 && python app.py")
            print()
            logger.error("Port %s already in use", port)
        else:
            raise


if __name__ == "__main__":
    main()
