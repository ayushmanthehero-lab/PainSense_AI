"""
modules/pose_estimator.py
─────────────────────────
MediaPipe Pose Landmarker (Tasks API v0.10+) wrapper.
Uses BlazePose Full model (.task bundle) for 33 3D body landmarks.

Processes a video file frame-by-frame and returns:
  - A list of per-frame landmark dicts
  - A representative annotated thumbnail
"""

from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import POSE_MODEL_COMPLEXITY, POSE_MIN_DETECTION_CONFIDENCE, POSE_MIN_TRACKING_CONFIDENCE

# Model file location (in painsense_ai/ root directory)
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pose_landmarker_full.task"
)

# MediaPipe landmark index map (33 landmarks from BlazePose GHUM)
LANDMARK_NAMES: Dict[int, str] = {
    0:  "nose",
    1:  "left_eye_inner",  2:  "left_eye",       3:  "left_eye_outer",
    4:  "right_eye_inner", 5:  "right_eye",       6:  "right_eye_outer",
    7:  "left_ear",        8:  "right_ear",
    9:  "mouth_left",      10: "mouth_right",
    11: "left_shoulder",   12: "right_shoulder",
    13: "left_elbow",      14: "right_elbow",
    15: "left_wrist",      16: "right_wrist",
    17: "left_pinky",      18: "right_pinky",
    19: "left_index",      20: "right_index",
    21: "left_thumb",      22: "right_thumb",
    23: "left_hip",        24: "right_hip",
    25: "left_knee",       26: "right_knee",
    27: "left_ankle",      28: "right_ankle",
    29: "left_heel",       30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}


@dataclass
class PoseFrame:
    """Holds landmark data for a single video frame."""
    frame_idx: int
    landmarks: Dict[str, Dict[str, float]]  # name → {x, y, z, visibility}
    annotated_image: Optional[np.ndarray] = field(default=None, repr=False)

    def get_lm(self, name: str) -> Optional[Dict[str, float]]:
        return self.landmarks.get(name)


class PoseEstimator:
    """
    Wraps MediaPipe PoseLandmarker (Tasks API v0.10+) for video-level extraction.

    Usage
    -----
    estimator = PoseEstimator()
    pose_frames, thumbnail = estimator.process_video("patient_clip.mp4")
    """

    def __init__(self) -> None:
        if not os.path.isfile(_MODEL_PATH):
            raise FileNotFoundError(
                f"MediaPipe pose model not found: {_MODEL_PATH}\n"
                "Ensure 'pose_landmarker_full.task' is in the painsense_ai/ directory."
            )
        base_options = mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
        self._options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=POSE_MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
            output_segmentation_masks=False,
        )

    def process_video(
        self,
        video_path: str,
        max_frames: int = 300,
        sample_every: int = 3,
    ) -> tuple[List[PoseFrame], Optional[np.ndarray]]:
        """
        Extract pose landmarks from every `sample_every`-th frame.

        Returns
        -------
        pose_frames  : list of PoseFrame objects
        thumbnail    : BGR annotated numpy array of the first detected frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        pose_frames: List[PoseFrame] = []
        thumbnail: Optional[np.ndarray] = None
        frame_idx = 0
        processed = 0

        with mp_vision.PoseLandmarker.create_from_options(self._options) as landmarker:
            while cap.isOpened() and processed < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_every != 0:
                    frame_idx += 1
                    continue

                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_img)

                if result.pose_landmarks:
                    raw_lms = result.pose_landmarks[0]
                    landmarks: Dict[str, Dict[str, float]] = {}
                    for idx, lm in enumerate(raw_lms):
                        name = LANDMARK_NAMES.get(idx, f"lm_{idx}")
                        landmarks[name] = {
                            "x":          lm.x,
                            "y":          lm.y,
                            "z":          lm.z,
                            "visibility": getattr(lm, "visibility", 1.0),
                        }

                    annotated = _draw_landmarks_on_image(frame.copy(), result)
                    pf = PoseFrame(frame_idx=frame_idx, landmarks=landmarks, annotated_image=annotated)
                    pose_frames.append(pf)
                    if thumbnail is None:
                        thumbnail = annotated.copy()

                processed += 1
                frame_idx += 1

        cap.release()
        return pose_frames, thumbnail

    def process_image(self, image_bgr: np.ndarray) -> Optional[PoseFrame]:
        """Extract pose from a single BGR numpy array."""
        rgb    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with mp_vision.PoseLandmarker.create_from_options(self._options) as landmarker:
            result = landmarker.detect(mp_img)

        if not result.pose_landmarks:
            return None

        raw_lms = result.pose_landmarks[0]
        landmarks: Dict[str, Dict[str, float]] = {}
        for idx, lm in enumerate(raw_lms):
            name = LANDMARK_NAMES.get(idx, f"lm_{idx}")
            landmarks[name] = {
                "x": lm.x, "y": lm.y, "z": lm.z,
                "visibility": getattr(lm, "visibility", 1.0),
            }

        annotated = _draw_landmarks_on_image(image_bgr.copy(), result)
        return PoseFrame(frame_idx=0, landmarks=landmarks, annotated_image=annotated)


# Drawing helper (Tasks API v0.10+)

def _draw_landmarks_on_image(bgr_image: np.ndarray, detection_result) -> np.ndarray:
    """Draw skeleton overlay; falls back to circle dots if drawing utils fail."""
    if not detection_result.pose_landmarks:
        return bgr_image

    try:
        from mediapipe.framework.formats import landmark_pb2  # type: ignore[import]
        from mediapipe.tasks.python.vision import drawing_utils as du
        from mediapipe.tasks.python.vision import drawing_styles as ds

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        for pose_lms in detection_result.pose_landmarks:
            proto = landmark_pb2.NormalizedLandmarkList()
            for lm in pose_lms:
                proto.landmark.add(x=lm.x, y=lm.y, z=lm.z)
            du.draw_landmarks(
                rgb,
                proto,
                mp_vision.PoseLandmarksConnections.POSE_LANDMARKS,
                ds.get_default_pose_landmarks_style(),
            )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    except Exception:
        h, w = bgr_image.shape[:2]
        for pose_lms in detection_result.pose_landmarks:
            for lm in pose_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(bgr_image, (cx, cy), 4, (0, 255, 0), -1)
        return bgr_image
