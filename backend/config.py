"""
Configuration for the Confidence Scoring Engine.

All scoring weights, thresholds, and connection parameters are defined here.
Override any value via environment variables for deployment flexibility.
"""

import os
from pathlib import Path
from typing import Dict


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
_DEFAULT_DB_PATH = Path(__file__).resolve().parent / "confidence.db"
_DEFAULT_DATABASE_URL = f"sqlite+aiosqlite:///{_DEFAULT_DB_PATH.as_posix()}"

# SQLite makes the application work out of the box. Production deployments can
# opt into PostgreSQL by setting DATABASE_URL explicitly.
DATABASE_URL: str = os.getenv("DATABASE_URL", _DEFAULT_DATABASE_URL)

# Synchronous URL variant (used by tests / Alembic if added later)
DATABASE_URL_SYNC: str = os.getenv(
    "DATABASE_URL_SYNC",
    f"sqlite:///{_DEFAULT_DB_PATH.as_posix()}",
)

# Local SQLite fallback when PostgreSQL is unavailable (dev convenience).
SQLITE_FALLBACK_URL: str = os.getenv("SQLITE_FALLBACK_URL", "")

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
]

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", "8765"))

# ---------------------------------------------------------------------------
# Scoring weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHTS: Dict[str, float] = {
    # Face signals — total 0.50
    "eye_contact": 0.20,
    "blink_rate": 0.10,
    "brow_tension": 0.10,
    "smile": 0.10,
    # Hand signals — total 0.30
    "hand_stability": 0.12,
    "fidgeting": 0.12,
    "hand_openness": 0.06,
    # Pose signals — total 0.20
    "shoulder_alignment": 0.10,
    "posture": 0.10,
}

# ---------------------------------------------------------------------------
# EMA smoothing (Section 2: composite score smoothing)
# ---------------------------------------------------------------------------
EMA_ALPHA: float = float(os.getenv("EMA_ALPHA", "0.2"))
# Face signals use alpha 0.3 for eye contact, 0.25 for smile
FACE_GAZE_EMA_ALPHA: float = float(os.getenv("FACE_GAZE_EMA_ALPHA", "0.3"))
FACE_SMILE_EMA_ALPHA: float = float(os.getenv("FACE_SMILE_EMA_ALPHA", "0.25"))

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
SCORE_UPDATE_INTERVAL: float = float(
    os.getenv("SCORE_UPDATE_INTERVAL", "2.0")
)  # seconds between score updates sent to client
TARGET_FPS: int = int(os.getenv("TARGET_FPS", "10"))
MAX_FRAME_BYTES: int = int(os.getenv("MAX_FRAME_BYTES", str(5 * 1024 * 1024)))
MAX_SESSION_ID_LENGTH: int = int(os.getenv("MAX_SESSION_ID_LENGTH", "255"))

# ---------------------------------------------------------------------------
# Visibility Gating (Section 1)
# ---------------------------------------------------------------------------
SHOULDER_VISIBILITY_THRESHOLD: float = float(
    os.getenv("SHOULDER_VISIBILITY_THRESHOLD", "0.6")
)
WRIST_VISIBILITY_THRESHOLD: float = float(
    os.getenv("WRIST_VISIBILITY_THRESHOLD", "0.6")
)
VISIBILITY_ON_FRAMES: int = int(os.getenv("VISIBILITY_ON_FRAMES", "5"))
VISIBILITY_OFF_FRAMES: int = int(os.getenv("VISIBILITY_OFF_FRAMES", "10"))

# ---------------------------------------------------------------------------
# Blink Rate — Bell Curve Scoring (Section 3 — Blink Rate)
# ---------------------------------------------------------------------------
EAR_THRESHOLD: float = float(os.getenv("EAR_THRESHOLD", "0.21"))
BLINK_CONSEC_FRAMES: int = int(os.getenv("BLINK_CONSEC_FRAMES", "3"))
BLINK_BELL_CURVE_CENTER: float = float(
    os.getenv("BLINK_BELL_CURVE_CENTER", "17.0")
)
BLINK_BELL_CURVE_SIGMA: float = float(
    os.getenv("BLINK_BELL_CURVE_SIGMA", "10.0")
)
BLINK_RATE_WINDOW_SECONDS: float = float(
    os.getenv("BLINK_RATE_WINDOW_SECONDS", "60.0")
)

# ---------------------------------------------------------------------------
# Eye Contact (Section 3 — Eye Contact)
# ---------------------------------------------------------------------------
HEAD_POSE_YAW_LIMIT: float = float(
    os.getenv("HEAD_POSE_YAW_LIMIT", "15.0")
)
HEAD_POSE_PITCH_LIMIT: float = float(
    os.getenv("HEAD_POSE_PITCH_LIMIT", "15.0")
)

# ---------------------------------------------------------------------------
# Hand Analysis (Section 3 — Hand Stability, Fidgeting, Openness)
# ---------------------------------------------------------------------------
HAND_STABILITY_WINDOW_SECONDS: float = float(
    os.getenv("HAND_STABILITY_WINDOW_SECONDS", "2.0")
)
HAND_REFERENCE_RANGE: float = float(
    os.getenv("HAND_REFERENCE_RANGE", "0.05")
)
FIDGET_WINDOW_SECONDS: float = float(
    os.getenv("FIDGET_WINDOW_SECONDS", "10.0")
)
FIDGET_MIN_FREQ_HZ: float = float(
    os.getenv("FIDGET_MIN_FREQ_HZ", "1.5")
)
FIDGET_EXPECTED_MAX_REVERSALS: int = int(
    os.getenv("FIDGET_EXPECTED_MAX_REVERSALS", "50")
)
FIDGET_NOISE_THRESHOLD_PX: float = float(
    os.getenv("FIDGET_NOISE_THRESHOLD_PX", "5.0")
)

# ---------------------------------------------------------------------------
# Posture (Section 3 — Posture)
# ---------------------------------------------------------------------------
POSTURE_SPINE_ANGLE_LIMIT: float = float(
    os.getenv("POSTURE_SPINE_ANGLE_LIMIT", "20.0")
)

# ---------------------------------------------------------------------------
# Baseline Calibration (Section 3 — brow, smile)
# ---------------------------------------------------------------------------
BASELINE_CALIBRATION_FRAMES: int = int(
    os.getenv("BASELINE_CALIBRATION_FRAMES", "30")
)

# ---------------------------------------------------------------------------
# Face analysis thresholds
# ---------------------------------------------------------------------------
NORMAL_BLINK_RATE_MIN: int = int(os.getenv("NORMAL_BLINK_RATE_MIN", "15"))
NORMAL_BLINK_RATE_MAX: int = int(os.getenv("NORMAL_BLINK_RATE_MAX", "20"))

# ---------------------------------------------------------------------------
# Session thresholds
# ---------------------------------------------------------------------------
LOW_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("LOW_CONFIDENCE_THRESHOLD", "50")
)

# ---------------------------------------------------------------------------
# MediaPipe model confidence
# ---------------------------------------------------------------------------
MP_MIN_DETECTION_CONFIDENCE: float = 0.5
MP_MIN_TRACKING_CONFIDENCE: float = 0.5
# Face and hand use very low thresholds for reliable webcam detection
MP_FACE_DETECTION_CONFIDENCE: float = 0.15
MP_FACE_TRACKING_CONFIDENCE: float = 0.15
MP_HAND_DETECTION_CONFIDENCE: float = 0.15
MP_HAND_TRACKING_CONFIDENCE: float = 0.15
# Delegate: CPU is most reliable across environments
MP_DELEGATE: str = os.getenv("MP_DELEGATE", "CPU")
