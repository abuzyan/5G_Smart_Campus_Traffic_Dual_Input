from collections import defaultdict
from threading import Lock, Thread, Event
import time
from pathlib import Path

import cv2
from streamlit_webrtc import VideoProcessorBase
from ultralytics import YOLO

# COCO class IDs:
# person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]
CLASS_KEYS = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
DISPLAY_NAMES = {
    "person": "Pedestrian",
    "bicycle": "Bicycle",
    "car": "Car",
    "motorcycle": "Motorcycle",
    "bus": "Bus",
    "truck": "Truck",
}


class SharedTrafficState:
    def __init__(self):
        self.lock = Lock()
        self.counts = defaultdict(int)
        self.current_objects = 0
        self.frames_processed = 0
        self.model_status = "Waiting for camera"
        self.last_error = ""

    def snapshot(self):
        with self.lock:
            return {
                "counts": dict(self.counts),
                "current_objects": self.current_objects,
                "frames_processed": self.frames_processed,
                "model_status": self.model_status,
                "last_error": self.last_error,
            }

    def reset(self):
        with self.lock:
            self.counts = defaultdict(int)
            self.current_objects = 0
            self.frames_processed = 0
            self.last_error = ""


_SHARED_STATE = SharedTrafficState()


def get_shared_state():
    return _SHARED_STATE


def reset_shared_state():
    _SHARED_STATE.reset()


def draw_detection_overlay(image, box, class_key, track_id, confidence):
    x1, y1, x2, y2 = [int(v) for v in box]
    color = (40, 190, 80) if track_id >= 0 else (0, 165, 255)
    label = DISPLAY_NAMES.get(class_key, class_key)
    if track_id >= 0:
        label += f" ID:{track_id}"
    label += f" {confidence:.2f}"

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    top = max(0, y1 - 25)
    cv2.rectangle(image, (x1, top), (min(image.shape[1] - 1, x1 + 190), y1), color, -1)
    cv2.putText(
        image, label, (x1 + 5, max(17, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    cv2.circle(image, center, 4, (255, 80, 30), -1)


def update_tracking_and_count(
    image, boxes, class_values, track_ids, confidences,
    previous_y, counted_ids, unique_objects, counts, line_y
):
    current_ids = set()

    for box, class_id, track_id, confidence in zip(
        boxes, class_values, track_ids, confidences
    ):
        class_key = CLASS_KEYS.get(int(class_id))
        if class_key is None:
            continue

        track_id = int(track_id)
        x1, y1, x2, y2 = [int(v) for v in box]
        center_y = (y1 + y2) // 2

        if track_id >= 0:
            current_ids.add(track_id)
            unique_objects.add(track_id)

            previous = previous_y.get(track_id)
            crossed = previous is not None and (
                (previous < line_y <= center_y) or
                (previous > line_y >= center_y)
            )
            if crossed and track_id not in counted_ids:
                counts[class_key] += 1
                counted_ids.add(track_id)

            previous_y[track_id] = center_y

        draw_detection_overlay(image, box, class_key, track_id, float(confidence))

    return current_ids


class TrafficProcessor(VideoProcessorBase):
    """WebRTC video processor for laptop webcam input."""

    def __init__(self, shared_state):
        self.state = shared_state
        self.model = None
        self.previous_y = {}
        self.counted_ids = set()
        self.unique_objects = set()
        self.counts = defaultdict(int)
        self.line_position_ratio = 0.55

    def _load_model(self):
        if self.model is None:
            with self.state.lock:
                self.state.model_status = "Loading yolov8n.pt..."
            self.model = YOLO("yolov8n.pt")
            with self.state.lock:
                self.state.model_status = "YOLOv8 ready"

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        height, width = image.shape[:2]
        line_y = int(height * self.line_position_ratio)

        try:
            self._load_model()
            results = self.model.track(
                source=image,
                persist=True,
                tracker="bytetrack.yaml",
                classes=TARGET_CLASSES,
                conf=0.35,
                iou=0.5,
                imgsz=640,
                verbose=False,
            )

            if results and results[0].boxes is not None:
                result_boxes = results[0].boxes
                xyxy = result_boxes.xyxy.cpu().numpy()
                class_values = result_boxes.cls.cpu().numpy().astype(int)
                ids = result_boxes.id
                track_ids = (
                    ids.cpu().numpy().astype(int)
                    if ids is not None else [-1] * len(xyxy)
                )
                confidences = (
                    result_boxes.conf.cpu().numpy()
                    if result_boxes.conf is not None else [0.0] * len(xyxy)
                )

                current_ids = update_tracking_and_count(
                    image, xyxy, class_values, track_ids, confidences,
                    self.previous_y, self.counted_ids, self.unique_objects,
                    self.counts, line_y
                )
            else:
                current_ids = set()

            cv2.line(image, (0, line_y), (width, line_y), (0, 0, 255), 3)
            cv2.putText(
                image, "COUNTING LINE", (12, max(25, line_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
            )

            with self.state.lock:
                self.state.counts = defaultdict(int, self.counts)
                self.state.current_objects = len(current_ids)
                self.state.frames_processed += 1
                self.state.last_error = ""

        except Exception as exc:
            with self.state.lock:
                self.state.last_error = f"{type(exc).__name__}: {exc}"
                self.state.model_status = "Error"
            cv2.putText(
                image, "Processing error - check dashboard",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

        return frame.from_ndarray(image, format="bgr24")



class ExternalCameraProcessor:
    """OpenCV processor for external USB cameras and IP CCTV RTSP streams."""

    def __init__(self, shared_state, camera_source, camera_name="External camera"):
        self.state = shared_state
        self.camera_source = camera_source
        self.camera_name = camera_name
        self.capture = None
        self.model = None
        self.thread = None
        self.stop_event = Event()
        self.frame_lock = Lock()
        self.latest_frame = None
        self.previous_y = {}
        self.counted_ids = set()
        self.unique_objects = set()
        self.counts = defaultdict(int)
        self.current_objects = 0
        self.line_position_ratio = 0.55

    @staticmethod
    def _normalise_source(source):
        if isinstance(source, str) and source.strip().isdigit():
            return int(source.strip())
        return source

    def start(self):
        if self.thread and self.thread.is_alive():
            return

        source = self._normalise_source(self.camera_source)
        with self.state.lock:
            self.state.model_status = f"Connecting to {self.camera_name}..."
            self.state.last_error = ""

        # DirectShow is preferred for USB camera devices on Windows.
        if isinstance(source, int):
            self.capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not self.capture.isOpened():
                self.capture.release()
                self.capture = cv2.VideoCapture(source)
        else:
            # OpenCV uses FFmpeg for RTSP when its build supports it.
            self.capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if not self.capture.isOpened():
                self.capture.release()
                self.capture = cv2.VideoCapture(source)

        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(
                "Camera open nahi hua. USB ke liye device index check karein; "
                "IP CCTV ke liye RTSP URL, credentials aur network check karein."
            )

        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stop_event.clear()
        self.thread = Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def _process_loop(self):
        try:
            self.model = YOLO("yolov8n.pt")
            with self.state.lock:
                self.state.model_status = f"Running: {self.camera_name}"

            while not self.stop_event.is_set():
                if self.capture is None:
                    break
                ok, image = self.capture.read()
                if not ok:
                    with self.state.lock:
                        self.state.last_error = (
                            "Camera se frame nahi mil raha. Connection/RTSP stream check karein."
                        )
                    time.sleep(0.15)
                    continue

                height, width = image.shape[:2]
                line_y = int(height * self.line_position_ratio)
                try:
                    results = self.model.track(
                        source=image,
                        persist=True,
                        tracker="bytetrack.yaml",
                        classes=TARGET_CLASSES,
                        conf=0.35,
                        iou=0.5,
                        imgsz=640,
                        verbose=False,
                    )

                    if results and results[0].boxes is not None:
                        boxes = results[0].boxes
                        xyxy = boxes.xyxy.cpu().numpy()
                        class_values = boxes.cls.cpu().numpy().astype(int)
                        ids = boxes.id
                        track_ids = (
                            ids.cpu().numpy().astype(int)
                            if ids is not None else [-1] * len(xyxy)
                        )
                        confidences = (
                            boxes.conf.cpu().numpy()
                            if boxes.conf is not None else [0.0] * len(xyxy)
                        )
                        current_ids = update_tracking_and_count(
                            image, xyxy, class_values, track_ids, confidences,
                            self.previous_y, self.counted_ids, self.unique_objects,
                            self.counts, line_y
                        )
                    else:
                        current_ids = set()

                    cv2.line(image, (0, line_y), (width, line_y), (0, 0, 255), 3)
                    cv2.putText(
                        image, "COUNTING LINE", (12, max(25, line_y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
                    )
                    with self.state.lock:
                        self.state.counts = defaultdict(int, self.counts)
                        self.state.current_objects = len(current_ids)
                        self.state.frames_processed += 1
                        self.state.model_status = f"Running: {self.camera_name}"
                        self.state.last_error = ""
                    with self.frame_lock:
                        self.latest_frame = image.copy()

                except Exception as exc:
                    with self.state.lock:
                        self.state.last_error = f"{type(exc).__name__}: {exc}"
                        self.state.model_status = "Processing error"
                    time.sleep(0.1)
        except Exception as exc:
            with self.state.lock:
                self.state.last_error = f"{type(exc).__name__}: {exc}"
                self.state.model_status = "Camera error"
        finally:
            if self.capture is not None:
                self.capture.release()
                self.capture = None

    def get_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.thread = None
        with self.state.lock:
            self.state.model_status = "Camera stopped"
            self.state.current_objects = 0

def process_uploaded_video(input_path, output_path, progress_callback=None):
    """
    Process an uploaded video frame-by-frame using YOLOv8 + ByteTrack.
    Returns line-crossing counts and unique tracked-object statistics.
    """
    input_path = str(input_path)
    output_path = str(output_path)

    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise ValueError(
            "OpenCV could not open this video. Try MP4 (H.264) or AVI format."
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 120:
        fps = 25.0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("The uploaded video has invalid dimensions.")

    # Resize very large videos to keep laptop processing manageable.
    max_width = 960
    if width > max_width:
        scale = max_width / width
        width = int(width * scale)
        height = int(height * scale)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create the annotated output video.")

    model = YOLO("yolov8n.pt")
    previous_y = {}
    counted_ids = set()
    unique_objects = set()
    counts = defaultdict(int)
    line_position_ratio = 0.55
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            line_y = int(height * line_position_ratio)

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=TARGET_CLASSES,
                conf=0.35,
                iou=0.5,
                imgsz=640,
                verbose=False,
            )

            if results and results[0].boxes is not None:
                result_boxes = results[0].boxes
                xyxy = result_boxes.xyxy.cpu().numpy()
                class_values = result_boxes.cls.cpu().numpy().astype(int)
                ids = result_boxes.id
                track_ids = (
                    ids.cpu().numpy().astype(int)
                    if ids is not None else [-1] * len(xyxy)
                )
                confidences = (
                    result_boxes.conf.cpu().numpy()
                    if result_boxes.conf is not None else [0.0] * len(xyxy)
                )

                update_tracking_and_count(
                    frame, xyxy, class_values, track_ids, confidences,
                    previous_y, counted_ids, unique_objects, counts, line_y
                )

            cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
            cv2.putText(
                frame, "COUNTING LINE", (12, max(25, line_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                frame, f"Frames: {frame_index + 1}",
                (12, height - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA
            )
            writer.write(frame)

            frame_index += 1
            if progress_callback and total_frames > 0 and (
                frame_index % max(1, total_frames // 100) == 0
            ):
                percent = min(99, int(frame_index * 100 / total_frames))
                progress_callback(percent, f"Processing frame {frame_index}/{total_frames}...")

    finally:
        capture.release()
        writer.release()

    if frame_index == 0:
        raise ValueError("No readable frames were found in the uploaded video.")

    if progress_callback:
        progress_callback(100, "Finishing video analysis...")

    return {
        "counts": dict(counts),
        "unique_objects": len(unique_objects),
        "frames": frame_index,
        "fps": float(fps),
        "duration": frame_index / float(fps),
    }
