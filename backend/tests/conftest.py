"""
Shared test fixtures and synthetic landmark helpers.
"""

from __future__ import annotations

import os
import tempfile

# Use a local file-based SQLite database for tests so that tables
# created synchronously are visible to the async engine used by
# the TestClient.  :memory: databases are per-connection and don't
# survive event-loop boundaries.
_DB_FILE = f"confidence_engine_test_{os.getpid()}.db"
_DB_ABS_PATH = os.path.join(tempfile.gettempdir(), _DB_FILE)

os.environ.setdefault(
    "DATABASE_URL",
    f"sqlite+aiosqlite:///{_DB_ABS_PATH}",
)

import sys
from dataclasses import dataclass
from typing import List

import pytest

# Ensure backend/ is on sys.path so `import config` etc. works
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---------------------------------------------------------------------------
# Synchronous table creation (avoids event-loop isolation issues)
# ---------------------------------------------------------------------------
def _create_tables_sync() -> None:
    """Create all ORM tables using a synchronous SQLAlchemy engine.

    This runs before any async code, so the database file exists and
    has the correct schema when the TestClient's event loop connects.
    """
    from sqlalchemy import create_engine
    from db import Base
    # Import the model so it registers with Base.metadata
    from models.session import ConfidenceSession  # noqa: F401

    sync_url = f"sqlite:///{_DB_ABS_PATH}"
    sync_engine = create_engine(sync_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()


_create_tables_sync()


# ---------------------------------------------------------------------------
# Cleanup: remove test database after all tests finish
# ---------------------------------------------------------------------------
def pytest_sessionfinish(session):
    """Remove the test database file after the test session completes."""
    import os
    try:
        if os.path.exists(_DB_ABS_PATH):
            os.unlink(_DB_ABS_PATH)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Synthetic landmark helper
# ---------------------------------------------------------------------------
@dataclass
class FakeLandmark:
    """Mimics ``mediapipe.framework.formats.landmark_pb2.NormalizedLandmark``.

    Includes visibility (default 1.0 = fully visible) so that pose
    visibility-gating checks pass by default in synthetic tests.
    """
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


class FakeLandmarkList:
    """Mimics a MediaPipe landmark list with ``.landmark`` attribute."""
    def __init__(self, landmarks: List[FakeLandmark]) -> None:
        self.landmark = landmarks


def make_landmarks(count: int = 478, **overrides) -> FakeLandmarkList:
    """Create a fake face landmark list with *count* landmarks.

    All landmarks default to (0.5, 0.5, 0.0) — center of frame.
    Use *overrides* to set specific indices:
        ``make_landmarks(478, **{33: (0.3, 0.4), 468: (0.5, 0.5)})``
    """
    lms = [FakeLandmark(0.5, 0.5, 0.0) for _ in range(count)]
    for idx_str, coords in overrides.items():
        idx = int(idx_str)
        if isinstance(coords, (tuple, list)):
            lms[idx] = FakeLandmark(coords[0], coords[1], coords[2] if len(coords) > 2 else 0.0)
        elif isinstance(coords, FakeLandmark):
            lms[idx] = coords
    return FakeLandmarkList(lms)


def make_hand_landmarks(count: int = 21, **overrides) -> FakeLandmarkList:
    """Create a fake hand landmark list (21 points)."""
    lms = [FakeLandmark(0.5, 0.5, 0.0) for _ in range(count)]
    for idx_str, coords in overrides.items():
        idx = int(idx_str)
        if isinstance(coords, (tuple, list)):
            lms[idx] = FakeLandmark(coords[0], coords[1], coords[2] if len(coords) > 2 else 0.0)
    return FakeLandmarkList(lms)


def make_pose_landmarks(count: int = 33, **overrides) -> FakeLandmarkList:
    """Create a fake pose landmark list (33 points).

    Each landmark defaults to fully visible (visibility=1.0).
    Override visibility by passing a FakeLandmark with specific visibility.
    """
    lms = [FakeLandmark(0.5, 0.5, 0.0) for _ in range(count)]
    for idx_str, coords in overrides.items():
        idx = int(idx_str)
        if isinstance(coords, (tuple, list)):
            lms[idx] = FakeLandmark(coords[0], coords[1], coords[2] if len(coords) > 2 else 0.0)
        elif isinstance(coords, FakeLandmark):
            lms[idx] = coords
    return FakeLandmarkList(lms)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
FRAME_SHAPE = (480, 640, 3)  # Standard VGA


@pytest.fixture
def frame_shape():
    """Standard 640×480 frame shape tuple."""
    return FRAME_SHAPE


@pytest.fixture
def centered_face():
    """Face landmarks where everything is roughly centered/neutral."""
    overrides = {}
    # Eye corners (for gaze)
    overrides[33] = (0.35, 0.45)    # Right eye outer
    overrides[133] = (0.45, 0.45)   # Right eye inner
    overrides[362] = (0.55, 0.45)   # Left eye inner
    overrides[263] = (0.65, 0.45)   # Left eye outer

    # Iris centers — centered between corners
    overrides[468] = (0.60, 0.45)   # Left iris center
    overrides[473] = (0.40, 0.45)   # Right iris center

    # EAR landmarks — open eyes (vertical spread)
    # Right eye: [33, 160, 158, 133, 153, 145]
    overrides[160] = (0.38, 0.43)   # upper inner
    overrides[158] = (0.42, 0.43)   # upper outer
    overrides[153] = (0.42, 0.47)   # lower outer
    overrides[145] = (0.38, 0.47)   # lower inner

    # Left eye: [362, 385, 386, 263, 374, 380]
    overrides[385] = (0.58, 0.43)
    overrides[386] = (0.62, 0.43)
    overrides[374] = (0.62, 0.47)
    overrides[380] = (0.58, 0.47)

    # Brow / face structure
    overrides[65] = (0.48, 0.38)    # Left inner brow
    overrides[295] = (0.52, 0.38)   # Right inner brow
    overrides[168] = (0.50, 0.50)   # Upper nose
    overrides[10] = (0.50, 0.30)    # Forehead
    overrides[152] = (0.50, 0.75)   # Chin

    # Mouth corners
    overrides[61] = (0.42, 0.62)    # Left mouth corner
    overrides[291] = (0.58, 0.62)   # Right mouth corner

    # Face width
    overrides[234] = (0.25, 0.50)   # Face left
    overrides[454] = (0.75, 0.50)   # Face right

    return make_landmarks(478, **{str(k): v for k, v in overrides.items()})
