"""
JALSAFAYOO AI - YOLOv8 Detection Engine
Model inference, video capture, and processed output pipeline.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import requests
from ultralytics import YOLO

from config import (
    DetectionConfig,
    MODEL_PATH,
    SourceType,
    UIConfig,
    VideoConfig,
)
from utils import (
    Detection,
    DetectionCounter,
    DetectionLog,
    FPSCounter,
    InferenceTimer,
    build_stats_payload,
    draw_detection_overlay,
    get_output_path,
    get_snapshot_path,
    get_timestamp,
    get_webcam_url_candidates,
    resize_frame,
)


logger = logging.getLogger("jalsafayoo.detector")


# ---------------------------------------------------------------------------
# YOLO Floater Detector
# ---------------------------------------------------------------------------

class FloaterDetector:
    """
    YOLOv8 single-class floater detection wrapper.

    Loads best.pt and runs inference with configurable confidence and IOU.
    """

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self._model_path = model_path
        self._model: Optional[YOLO] = None
        self._confidence = DetectionConfig.DEFAULT_CONFIDENCE
        self._iou = DetectionConfig.DEFAULT_IOU
        self._status = "Idle"

    @property
    def status(self) -> str:
        """Current model status string."""
        return self._status

    @property
    def is_loaded(self) -> bool:
        """Whether the YOLO model is loaded."""
        return self._model is not None

    def load(self) -> None:
        """Load the YOLOv8 model from disk."""
        if not self._model_path.exists():
            self._status = "Error: Model Not Found"
            logger.error("Model file not found: %s", self._model_path)
            raise FileNotFoundError(f"Model not found: {self._model_path}")

        try:
            self._status = "Loading"
            logger.info("Loading YOLO model from %s", self._model_path)
            self._model = YOLO(str(self._model_path))
            self._status = "Ready"
            logger.info("YOLO model loaded successfully")
        except Exception as exc:
            self._status = f"Error: {exc}"
            logger.exception("Failed to load YOLO model")
            raise

    def set_thresholds(self, confidence: float, iou: float) -> None:
        """Update inference thresholds."""
        self._confidence = confidence
        self._iou = iou

    def predict(
        self,
        frame: np.ndarray,
        counter: DetectionCounter,
    ) -> Tuple[List[Detection], float]:
        """
        Run YOLO inference on a single frame.

        Args:
            frame: BGR numpy array.
            counter: DetectionCounter for assigning unique IDs.

        Returns:
            Tuple of (detection list, inference time in ms).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        timer = InferenceTimer()
        timer.start()

        results = self._model.predict(
            frame,
            conf=self._confidence,
            iou=self._iou,
            imgsz=DetectionConfig.IMAGE_SIZE,
            verbose=False,
        )

        inference_ms = timer.stop()
        detections: List[Detection] = []
        timestamp = get_timestamp()

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())

                    detection = Detection.from_bbox(
                        detection_id=counter.next_id(),
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        confidence=confidence,
                        timestamp=timestamp,
                    )
                    detections.append(detection)

        return detections, inference_ms


# ---------------------------------------------------------------------------
# Video Source Abstraction
# ---------------------------------------------------------------------------

class BaseVideoSource(ABC):
    """Abstract video input source."""

    def __init__(self) -> None:
        self._capture: Optional[cv2.VideoCapture] = None
        self._status = "Disconnected"

    @property
    def status(self) -> str:
        return self._status

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @abstractmethod
    def connect(self) -> bool:
        """Open the video source."""

    def disconnect(self) -> None:
        """Release the video capture."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._status = "Disconnected"

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame.

        Returns:
            Tuple of (success, frame or None).
        """
        if not self.is_open:
            return False, None
        success, frame = self._capture.read()
        return success, frame if success else None

    def get_properties(self) -> dict:
        """Return video stream properties."""
        if not self.is_open:
            return {"width": 0, "height": 0, "fps": 0}

        return {
            "width": int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": round(self._capture.get(cv2.CAP_PROP_FPS), 2),
        }


class IPWebcamSource(BaseVideoSource):
    """
    Live IP webcam stream source.

    Tries multiple URL paths (/video, /videofeed, etc.) and falls back to
    direct HTTP MJPEG reading when OpenCV cannot open the stream.
    """

    def __init__(self, url: str) -> None:
        super().__init__()
        self._url = url
        self._active_url: Optional[str] = None
        self._mjpeg_stream: Optional[requests.Response] = None
        self._mjpeg_session: Optional[requests.Session] = None
        self._mjpeg_iter = None
        self._mjpeg_buffer = b""
        self._frame_width = 0
        self._frame_height = 0
        self._use_mjpeg = False

    @property
    def active_url(self) -> Optional[str]:
        """URL that successfully connected."""
        return self._active_url

    @property
    def is_open(self) -> bool:
        if self._use_mjpeg:
            return self._mjpeg_stream is not None
        return self._capture is not None and self._capture.isOpened()

    def connect(self) -> bool:
        self.disconnect()
        self._status = "Connecting"
        logger.info("Connecting to IP webcam: %s", self._url)

        candidates = get_webcam_url_candidates(self._url)
        logger.info("Webcam URL candidates: %s", candidates)

        for candidate in candidates:
            if self._try_opencv(candidate):
                self._active_url = candidate
                self._status = "Connected"
                logger.info("IP webcam connected via OpenCV → %s", candidate)
                return True

            if self._try_mjpeg(candidate):
                self._active_url = candidate
                self._use_mjpeg = True
                self._status = "Connected"
                logger.info("IP webcam connected via MJPEG stream → %s", candidate)
                return True

        self._status = "Connection Failed"
        logger.error(
            "Failed to connect to IP webcam. Tried %d URL(s). "
            "Ensure the phone/app is on the same network and the stream is active. "
            "Example: http://192.168.29.166:8080/video",
            len(candidates),
        )
        return False

    def disconnect(self) -> None:
        if self._mjpeg_stream is not None:
            try:
                self._mjpeg_stream.close()
            except Exception:
                pass
            self._mjpeg_stream = None

        if self._mjpeg_session is not None:
            try:
                self._mjpeg_session.close()
            except Exception:
                pass
            self._mjpeg_session = None

        self._mjpeg_buffer = b""
        self._mjpeg_iter = None
        self._use_mjpeg = False
        self._active_url = None
        self._frame_width = 0
        self._frame_height = 0
        super().disconnect()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_open:
            return False, None

        if self._use_mjpeg:
            return self._read_mjpeg_frame()

        success, frame = self._capture.read()
        return success, frame if success else None

    def get_properties(self) -> dict:
        if not self.is_open:
            return {"width": 0, "height": 0, "fps": 0}

        if self._use_mjpeg:
            return {
                "width": self._frame_width,
                "height": self._frame_height,
                "fps": VideoConfig.STREAM_FPS_TARGET,
            }

        return {
            "width": int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": round(self._capture.get(cv2.CAP_PROP_FPS), 2),
        }

    def _try_opencv(self, url: str) -> bool:
        """Attempt OpenCV FFMPEG capture and verify with a test frame."""
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, VideoConfig.CAPTURE_BUFFER_SIZE)

        if not cap.isOpened():
            cap.release()
            return False

        success, frame = cap.read()
        if success and frame is not None:
            self._capture = cap
            self._frame_height, self._frame_width = frame.shape[:2]
            return True

        cap.release()
        return False

    def _try_mjpeg(self, url: str) -> bool:
        """Attempt direct HTTP MJPEG frame capture via requests."""
        session = requests.Session()
        try:
            response = session.get(
                url,
                stream=True,
                timeout=VideoConfig.WEBCAM_CONNECT_TIMEOUT,
                headers={"User-Agent": "JALSAFAYOO-AI/1.0"},
            )
            if response.status_code != 200:
                response.close()
                session.close()
                return False

            self._mjpeg_buffer = b""
            self._mjpeg_iter = response.iter_content(chunk_size=4096)
            frame = self._extract_jpeg_frame(response, max_chunks=80)
            if frame is None:
                response.close()
                session.close()
                self._mjpeg_iter = None
                return False

            self._frame_height, self._frame_width = frame.shape[:2]
            self._mjpeg_stream = response
            self._mjpeg_session = session
            return True
        except Exception as exc:
            logger.debug("MJPEG connect failed for %s: %s", url, exc)
            session.close()
            return False

    def _read_mjpeg_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._mjpeg_stream is None:
            return False, None

        frame = self._extract_jpeg_frame(self._mjpeg_stream, max_chunks=40)
        if frame is not None:
            return True, frame
        return False, None

    def _extract_jpeg_frame(
        self,
        response: requests.Response,
        max_chunks: int = 40,
    ) -> Optional[np.ndarray]:
        """Parse the next JPEG frame from an MJPEG HTTP response stream."""
        if self._mjpeg_iter is None:
            self._mjpeg_iter = response.iter_content(chunk_size=4096)

        chunks_read = 0

        while chunks_read < max_chunks:
            try:
                chunk = next(self._mjpeg_iter)
            except StopIteration:
                break

            if not chunk:
                continue

            chunks_read += 1
            self._mjpeg_buffer += chunk

            start = self._mjpeg_buffer.find(b"\xff\xd8")
            end = self._mjpeg_buffer.find(b"\xff\xd9")

            if start != -1 and end != -1 and end > start:
                jpg = self._mjpeg_buffer[start : end + 2]
                self._mjpeg_buffer = self._mjpeg_buffer[end + 2 :]
                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame

        return None


class FileVideoSource(BaseVideoSource):
    """Local video file source (demo or uploaded)."""

    def __init__(self, filepath: Path, source_type: str = SourceType.DEMO_VIDEO) -> None:
        super().__init__()
        self._filepath = filepath
        self._source_type = source_type

    def connect(self) -> bool:
        self.disconnect()

        if not self._filepath.exists():
            self._status = "File Not Found"
            logger.error("Video file not found: %s", self._filepath)
            return False

        self._status = "Connecting"
        logger.info("Opening video file: %s", self._filepath)

        self._capture = cv2.VideoCapture(str(self._filepath))

        if self._capture.isOpened():
            self._status = "Playing"
            logger.info("Video file opened successfully")
            return True

        self._status = "Open Failed"
        logger.error("Failed to open video file: %s", self._filepath)
        return False

    def restart(self) -> bool:
        """Seek to beginning of video file."""
        if not self.is_open:
            return False
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._status = "Playing"
        return True


# ---------------------------------------------------------------------------
# Processed Video Writer
# ---------------------------------------------------------------------------

class ProcessedVideoWriter:
    """Write annotated frames to an output video file."""

    def __init__(self) -> None:
        self._writer: Optional[cv2.VideoWriter] = None
        self._output_path: Optional[Path] = None
        self._frame_size: Optional[Tuple[int, int]] = None
        self._fps: float = VideoConfig.STREAM_FPS_TARGET

    @property
    def output_path(self) -> Optional[Path]:
        return self._output_path

    @property
    def is_active(self) -> bool:
        return self._writer is not None

    def start(self, frame: np.ndarray, fps: float = 30.0) -> Path:
        """
        Initialize video writer from first frame dimensions.

        Args:
            frame: First annotated frame.
            fps: Output video FPS.

        Returns:
            Path to the output file.
        """
        self.stop()

        height, width = frame.shape[:2]
        self._frame_size = (width, height)
        self._fps = fps if fps > 0 else VideoConfig.STREAM_FPS_TARGET
        self._output_path = get_output_path("processed")

        fourcc = cv2.VideoWriter_fourcc(*VideoConfig.OUTPUT_CODEC)
        self._writer = cv2.VideoWriter(
            str(self._output_path),
            fourcc,
            self._fps,
            self._frame_size,
        )

        logger.info("Started recording → %s", self._output_path)
        return self._output_path

    def write(self, frame: np.ndarray) -> None:
        """Write a single annotated frame."""
        if self._writer is not None:
            self._writer.write(frame)

    def stop(self) -> Optional[Path]:
        """
        Finalize and close the video writer.

        Returns:
            Path to saved file, or None if nothing was written.
        """
        saved_path = self._output_path

        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Recording saved → %s", saved_path)

        self._output_path = None
        self._frame_size = None
        return saved_path


# ---------------------------------------------------------------------------
# Video Processing Pipeline
# ---------------------------------------------------------------------------

class VideoPipeline:
    """
    Orchestrates video capture, YOLO inference, annotation, and output.

    Runs in a background thread; emits updates via callback functions.
    """

    def __init__(
        self,
        detector: FloaterDetector,
        on_frame: Optional[Callable] = None,
        on_stats: Optional[Callable] = None,
        on_detections: Optional[Callable] = None,
        on_status: Optional[Callable] = None,
        on_output_ready: Optional[Callable] = None,
    ) -> None:
        self.detector = detector
        self._on_frame = on_frame
        self._on_stats = on_stats
        self._on_detections = on_detections
        self._on_status = on_status
        self._on_output_ready = on_output_ready

        self._source: Optional[BaseVideoSource] = None
        self._source_type: str = SourceType.NONE
        self._writer = ProcessedVideoWriter()

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._paused = False
        self._stop_event = threading.Event()

        self._current_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        self.fps_counter = FPSCounter()
        self.detection_counter = DetectionCounter()
        self.detection_log = DetectionLog()

        self._inference_time_ms: float = 0.0
        self._last_output_path: Optional[Path] = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def camera_status(self) -> str:
        if self._source:
            return self._source.status
        return "Disconnected"

    @property
    def last_output_path(self) -> Optional[Path]:
        return self._last_output_path

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Thread-safe access to the latest annotated frame."""
        with self._frame_lock:
            if self._current_frame is None:
                return None
            return self._current_frame.copy()

    def connect_ip_webcam(self, url: str) -> bool:
        """Connect to a live IP webcam stream."""
        self.stop()
        self._source = IPWebcamSource(url)
        self._source_type = SourceType.IP_WEBCAM
        connected = self._source.connect()
        self._emit_status()
        return connected

    def connect_demo_video(self, filepath: Path) -> bool:
        """Connect to a demo video from /videos."""
        self.stop()
        self._source = FileVideoSource(filepath, SourceType.DEMO_VIDEO)
        self._source_type = SourceType.DEMO_VIDEO
        connected = self._source.connect()
        self._emit_status()
        return connected

    def connect_upload(self, filepath: Path) -> bool:
        """Connect to an uploaded video from /uploads."""
        self.stop()
        self._source = FileVideoSource(filepath, SourceType.UPLOAD)
        self._source_type = SourceType.UPLOAD
        connected = self._source.connect()
        self._emit_status()
        return connected

    def start(self) -> None:
        """Start the detection pipeline in a background thread."""
        if self._running:
            return

        if self._source is None or not self._source.is_open:
            logger.warning("Cannot start pipeline: no active video source")
            return

        self._stop_event.clear()
        self._running = True
        self._paused = False
        self.detection_counter.reset()
        self.fps_counter.reset()

        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info("Video pipeline started")
        self._emit_status()

    def pause(self) -> None:
        """Pause frame processing."""
        self._paused = True
        if self._source:
            self._source._status = "Paused"
        self._emit_status()

    def resume(self) -> None:
        """Resume frame processing."""
        self._paused = False
        if self._source and self._source.is_open:
            self._source._status = "Playing" if isinstance(self._source, FileVideoSource) else "Connected"
        self._emit_status()

    def restart(self) -> None:
        """Restart video from beginning (file sources only)."""
        if isinstance(self._source, FileVideoSource):
            self.detection_counter.reset()
            self.detection_log.clear()
            self._source.restart()
            self._paused = False
            self.fps_counter.reset()
            logger.info("Video restarted")
            self._emit_status()

    def stop(self) -> None:
        """Stop the pipeline and release resources."""
        self._stop_event.set()
        self._running = False
        self._paused = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        saved_path = self._writer.stop()
        if saved_path:
            self._last_output_path = saved_path
            if self._on_output_ready:
                self._on_output_ready(str(saved_path))

        if self._source:
            self._source.disconnect()

        self._source = None
        self._source_type = SourceType.NONE

        with self._frame_lock:
            self._current_frame = None

        logger.info("Video pipeline stopped")
        self._emit_status()

    def capture_snapshot(self) -> Optional[Path]:
        """
        Save the current annotated frame as a snapshot.

        Returns:
            Path to saved snapshot, or None if no frame available.
        """
        frame = self.get_current_frame()
        if frame is None:
            return None

        snapshot_path = get_snapshot_path()
        cv2.imwrite(str(snapshot_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info("Snapshot saved → %s", snapshot_path)
        return snapshot_path

    def _process_loop(self) -> None:
        """Main processing loop executed in background thread."""
        reconnect_attempts = 0
        source_fps = VideoConfig.STREAM_FPS_TARGET

        if self._source and self._source.is_open:
            props = self._source.get_properties()
            if props["fps"] > 0:
                source_fps = props["fps"]

        writer_started = False

        while not self._stop_event.is_set():
            if self._paused:
                time.sleep(0.05)
                continue

            if self._source is None or not self._source.is_open:
                break

            success, frame = self._source.read()

            if not success or frame is None:
                if isinstance(self._source, FileVideoSource):
                    logger.info("Video file ended")
                    self._finalize_output()
                    break

                if isinstance(self._source, IPWebcamSource):
                    reconnect_attempts += 1
                    if reconnect_attempts > VideoConfig.MAX_RECONNECT_ATTEMPTS:
                        logger.error("Max reconnect attempts reached")
                        self._source._status = "Connection Lost"
                        break

                    self._source._status = "Reconnecting"
                    self._emit_status()
                    time.sleep(VideoConfig.RECONNECT_DELAY_SEC)
                    self._source.connect()
                    continue

                break

            reconnect_attempts = 0

            try:
                detections, inference_ms = self.detector.predict(
                    frame, self.detection_counter
                )
            except Exception:
                logger.exception("Inference failed on frame")
                detections = []
                inference_ms = 0.0

            self._inference_time_ms = inference_ms
            self.detection_counter.update(len(detections))

            if detections:
                self.detection_log.add_batch(detections)

            annotated = draw_detection_overlay(frame, detections)
            display_frame = resize_frame(
                annotated,
                VideoConfig.DISPLAY_MAX_WIDTH,
                VideoConfig.DISPLAY_MAX_HEIGHT,
            )

            with self._frame_lock:
                self._current_frame = display_frame

            if not writer_started:
                self._writer.start(display_frame, source_fps)
                writer_started = True

            self._writer.write(display_frame)

            fps = self.fps_counter.tick()
            self._emit_frame_updates(detections, fps)

            if source_fps > 0:
                time.sleep(max(0, 1.0 / source_fps - 0.001))

        self._running = False
        self._finalize_output()
        self._emit_status()

    def _finalize_output(self) -> None:
        """Save processed video when pipeline ends."""
        saved_path = self._writer.stop()
        if saved_path:
            self._last_output_path = saved_path
            if self._on_output_ready:
                self._on_output_ready(str(saved_path))

    def _emit_frame_updates(self, detections: List[Detection], fps: float) -> None:
        """Broadcast frame statistics and detections to callbacks."""
        if self._on_detections:
            self._on_detections(
                [d.to_dict() for d in detections],
                self.detection_log.to_dict_list()[:UIConfig.MAX_TABLE_ROWS],
            )

        if self._on_stats:
            payload = build_stats_payload(
                fps=fps,
                total_floaters=self.detection_counter.total,
                current_detections=self.detection_counter.current,
                model_status=self.detector.status,
                camera_status=self.camera_status,
                input_source=self._source_type,
                confidence=self.detector._confidence,
                iou=self.detector._iou,
                inference_time_ms=self._inference_time_ms,
            )
            self._on_stats(payload)

    def _emit_status(self) -> None:
        """Broadcast pipeline status change."""
        if self._on_status:
            self._on_status({
                "is_running": self._running,
                "is_paused": self._paused,
                "camera_status": self.camera_status,
                "input_source": self._source_type,
                "model_status": self.detector.status,
            })
