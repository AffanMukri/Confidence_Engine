"""
Download and cache MediaPipe Tasks model files.

MediaPipe >= 0.10.31 removed the legacy ``mp.solutions`` API.  This module
provides local paths to the ``.task`` model bundles used by the Tasks vision
landmarkers.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "mp"

MODEL_URLS: dict[str, str] = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/"
        "pose_landmarker_lite.task"
    ),
}


def ensure_model(name: str) -> Path:
    """Return the local path to a model file, downloading it if needed."""
    if name not in MODEL_URLS:
        raise ValueError(f"Unknown model: {name}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / name

    if path.exists() and path.stat().st_size > 0:
        return path

    url = MODEL_URLS[name]
    logger.info("Downloading MediaPipe model %s …", name)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    logger.info("Model ready: %s", path)
    return path
