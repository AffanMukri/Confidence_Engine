"""
Unit tests for FaceAnalyzer.

Tests EAR calculation, gaze scoring, brow tension, and smile ratio
using synthetic landmarks — no webcam or MediaPipe required.
"""

import time
import numpy as np
import pytest

from scoring.face_analyzer import FaceAnalyzer
from conftest import (
    FakeLandmark,
    FakeLandmarkList,
    make_landmarks,
    FRAME_SHAPE,
)


class TestEARCalculation:
    """Test the Eye Aspect Ratio helper."""

    def test_ear_open_eye(self):
        """Open eye should have EAR > 0.25."""
        analyzer = FaceAnalyzer()
        # 6 points: outer_corner, upper_inner, upper_outer,
        #           inner_corner, lower_outer, lower_inner
        pts = np.array([
            [100, 200],  # p1 outer
            [120, 180],  # p2 upper inner
            [140, 180],  # p3 upper outer
            [160, 200],  # p4 inner
            [140, 220],  # p5 lower outer
            [120, 220],  # p6 lower inner
        ], dtype=float)
        ear = analyzer._ear(pts)
        assert ear > 0.25, f"Open eye EAR should be > 0.25, got {ear}"

    def test_ear_closed_eye(self):
        """Closed eye should have EAR < 0.2."""
        analyzer = FaceAnalyzer()
        pts = np.array([
            [100, 200],  # p1
            [120, 199],  # p2 very close to p6
            [140, 199],  # p3 very close to p5
            [160, 200],  # p4
            [140, 201],  # p5
            [120, 201],  # p6
        ], dtype=float)
        ear = analyzer._ear(pts)
        assert ear < 0.1, f"Closed eye EAR should be < 0.1, got {ear}"

    def test_ear_zero_width(self):
        """EAR with zero-width eye should return 0."""
        analyzer = FaceAnalyzer()
        pts = np.array([
            [100, 200],
            [100, 190],
            [100, 190],
            [100, 200],  # same x as p1 → zero width
            [100, 210],
            [100, 210],
        ], dtype=float)
        ear = analyzer._ear(pts)
        assert ear == 0.0


class TestEyeContact:
    """Test gaze/iris offset scoring."""

    def test_centered_gaze(self, centered_face, frame_shape):
        """Iris centered between eye corners → score near 100."""
        analyzer = FaceAnalyzer()
        result = analyzer.analyze(centered_face, frame_shape)
        assert result["eye_contact"] is not None
        assert result["eye_contact"] >= 80, (
            f"Centered gaze should score >= 80, got {result['eye_contact']}"
        )

    def test_extreme_gaze(self, frame_shape):
        """Iris at the edge of the eye → much lower score."""
        analyzer = FaceAnalyzer()
        overrides = {
            # Eye corners
            "33": (0.35, 0.45), "133": (0.45, 0.45),
            "362": (0.55, 0.45), "263": (0.65, 0.45),
            # Iris pushed to the edge of the eye
            "468": (0.64, 0.45),  # Left iris near outer corner
            "473": (0.36, 0.45),  # Right iris near outer corner
            # EAR landmarks (keep eyes open)
            "160": (0.38, 0.43), "158": (0.42, 0.43),
            "153": (0.42, 0.47), "145": (0.38, 0.47),
            "385": (0.58, 0.43), "386": (0.62, 0.43),
            "374": (0.62, 0.47), "380": (0.58, 0.47),
            # Face structure
            "65": (0.48, 0.38), "295": (0.52, 0.38),
            "168": (0.50, 0.50), "10": (0.50, 0.30), "152": (0.50, 0.75),
            "61": (0.42, 0.62), "291": (0.58, 0.62),
            "234": (0.25, 0.50), "454": (0.75, 0.50),
        }
        face = make_landmarks(478, **overrides)
        result = analyzer.analyze(face, frame_shape)
        assert result["eye_contact"] < 60, (
            f"Extreme gaze should score < 60, got {result['eye_contact']}"
        )

    def test_no_face_returns_none(self, frame_shape):
        """No face → all signals None."""
        analyzer = FaceAnalyzer()
        result = analyzer.analyze(None, frame_shape)
        assert result["eye_contact"] is None
        assert result["blink_rate"] is None
        assert result["brow_tension"] is None
        assert result["smile"] is None


class TestBrowTension:
    """Test brow tension normalization."""

    def test_normal_brow(self, centered_face, frame_shape):
        """Normal brow position → reasonable score."""
        analyzer = FaceAnalyzer()
        result = analyzer.analyze(centered_face, frame_shape)
        assert result["brow_tension"] is not None
        assert 0 <= result["brow_tension"] <= 100


class TestSmile:
    """Test smile ratio scoring."""

    def test_neutral_face(self, centered_face, frame_shape):
        """Neutral expression → around 80 baseline."""
        analyzer = FaceAnalyzer()
        result = analyzer.analyze(centered_face, frame_shape)
        assert result["smile"] is not None
        # Should be in the 60-100 range for neutral
        assert 40 <= result["smile"] <= 100

    def test_wide_smile(self, frame_shape):
        """Wide smile (larger mouth width ratio) → higher score."""
        analyzer = FaceAnalyzer()
        # Bypass baseline calibration so geometric fallback uses instant data
        analyzer._cal_frame_count = analyzer._cal_frames + 1
        overrides = {
            "33": (0.35, 0.45), "133": (0.45, 0.45),
            "362": (0.55, 0.45), "263": (0.65, 0.45),
            "468": (0.60, 0.45), "473": (0.40, 0.45),
            "160": (0.38, 0.43), "158": (0.42, 0.43),
            "153": (0.42, 0.47), "145": (0.38, 0.47),
            "385": (0.58, 0.43), "386": (0.62, 0.43),
            "374": (0.62, 0.47), "380": (0.58, 0.47),
            "65": (0.48, 0.38), "295": (0.52, 0.38),
            "168": (0.50, 0.50), "10": (0.50, 0.30), "152": (0.50, 0.75),
            # Wide mouth = smile
            "61": (0.30, 0.62), "291": (0.70, 0.62),
            "234": (0.25, 0.50), "454": (0.75, 0.50),
        }
        face = make_landmarks(478, **overrides)
        result = analyzer.analyze(face, frame_shape)
        assert result["smile"] > 80, (
            f"Wide smile should score > 80, got {result['smile']}"
        )


class TestScoreRanges:
    """All scores should always be in [0, 100] or None."""

    def test_all_scores_in_range(self, centered_face, frame_shape):
        analyzer = FaceAnalyzer()
        result = analyzer.analyze(centered_face, frame_shape)
        for key, val in result.items():
            if val is not None:
                assert 0.0 <= val <= 100.0, (
                    f"{key} = {val} is out of [0, 100] range"
                )
