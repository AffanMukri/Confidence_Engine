"""
Unit tests for PoseAnalyzer.

Tests shoulder alignment angle and forward-lean / posture quality
using synthetic pose landmarks.
"""

import numpy as np
import pytest

from scoring.pose_analyzer import PoseAnalyzer
from conftest import FakeLandmark, make_pose_landmarks, FRAME_SHAPE


class TestShoulderAlignment:
    """Test shoulder angle scoring."""

    def test_level_shoulders(self, frame_shape):
        """Perfectly level shoulders → score near 100."""
        analyzer = PoseAnalyzer()
        pose = make_pose_landmarks(
            33,
            **{
                "7": (0.45, 0.30),   # Left ear
                "8": (0.55, 0.30),   # Right ear
                "11": (0.35, 0.50),  # Left shoulder
                "12": (0.65, 0.50),  # Right shoulder (same y = level)
                "23": (0.40, 0.80),  # Left hip
                "24": (0.60, 0.80),  # Right hip
            }
        )
        result = analyzer.analyze(pose, frame_shape, shoulders_visible=True)
        assert result["shoulder_alignment"] is not None
        assert result["shoulder_alignment"] >= 90, (
            f"Level shoulders should score >= 90, got {result['shoulder_alignment']}"
        )

    def test_tilted_shoulders(self, frame_shape):
        """Noticeably tilted shoulders → lower score."""
        analyzer = PoseAnalyzer()
        pose = make_pose_landmarks(
            33,
            **{
                "7": (0.45, 0.30),
                "8": (0.55, 0.30),
                "11": (0.35, 0.45),  # Left shoulder higher
                "12": (0.65, 0.60),  # Right shoulder lower → tilt
                "23": (0.40, 0.80),
                "24": (0.60, 0.80),
            }
        )
        result = analyzer.analyze(pose, frame_shape, shoulders_visible=True)
        assert result["shoulder_alignment"] is not None
        assert result["shoulder_alignment"] < 80, (
            f"Tilted shoulders should score < 80, got {result['shoulder_alignment']}"
        )


class TestPostureQuality:
    """Test forward-lean / upright posture scoring."""

    def test_upright_posture(self, frame_shape):
        """Shoulders directly above hips → 0° spine angle → score near 100."""
        analyzer = PoseAnalyzer()
        pose = make_pose_landmarks(
            33,
            **{
                "0": (0.50, 0.35),   # Nose
                "7": (0.45, 0.25),   # Left ear
                "8": (0.55, 0.25),   # Right ear
                "11": (0.35, 0.50),  # Left shoulder
                "12": (0.65, 0.50),  # Right shoulder
                "23": (0.40, 0.80),  # Left hip (directly below shoulder midpoint)
                "24": (0.60, 0.80),  # Right hip
            }
        )
        result = analyzer.analyze(pose, frame_shape, shoulders_visible=True)
        assert result["posture"] is not None
        assert result["posture"] >= 90, (
            f"Upright posture should score >= 90, got {result['posture']}"
        )

    def test_spine_lean(self, frame_shape):
        """Shoulders offset horizontally from hips → tilted spine → lower score."""
        analyzer = PoseAnalyzer()
        pose = make_pose_landmarks(
            33,
            **{
                "0": (0.50, 0.35),
                "7": (0.45, 0.25),
                "8": (0.55, 0.25),
                "11": (0.35, 0.50),  # Left shoulder
                "12": (0.65, 0.50),  # Right shoulder
                # Hips offset — shoulder_mid at (0.50,0.50),
                # hip_mid at (0.45,0.80) → ~10° spine lean → ~50 score
                "23": (0.35, 0.80),
                "24": (0.55, 0.80),
            }
        )
        result = analyzer.analyze(pose, frame_shape, shoulders_visible=True)
        assert result["posture"] is not None
        # ~10° lean → ~50 score
        assert result["posture"] < 70, (
            f"Spine lean should score < 70, got {result['posture']}"
        )
        assert result["posture"] > 25, (
            f"Moderate spine lean should score > 25, got {result['posture']}"
        )

    def test_hips_not_visible_fallback_good(self, frame_shape):
        """Hips not visible + head centered → fallback gives decent score."""
        analyzer = PoseAnalyzer()
        pose = make_pose_landmarks(
            33,
            **{
                "0": (0.50, 0.35),   # Nose centered
                "7": (0.45, 0.25),   # Left ear
                "8": (0.55, 0.25),   # Right ear
                "11": (0.35, 0.50),  # Left shoulder
                "12": (0.65, 0.50),  # Right shoulder
                # Hips not visible (visibility=0)
                "23": FakeLandmark(0.40, 0.80, 0.0, visibility=0.0),
                "24": FakeLandmark(0.60, 0.80, 0.0, visibility=0.0),
            }
        )
        result = analyzer.analyze(pose, frame_shape, shoulders_visible=True)
        assert result["posture"] is not None
        # Head centered → fallback should give reasonable score
        assert result["posture"] >= 60, (
            f"Fallback with centered head should score >= 60, got {result['posture']}"
        )

    def test_hips_not_visible_fallback_poor(self, frame_shape):
        """Hips not visible + head far forward → fallback gives low score."""
        analyzer = PoseAnalyzer()
        pose = make_pose_landmarks(
            33,
            **{
                "0": (0.58, 0.32),   # Nose far forward (offset from shoulder midpoint)
                "7": (0.56, 0.26),   # Left ear shifted
                "8": (0.66, 0.26),   # Right ear shifted
                "11": (0.30, 0.50),
                "12": (0.70, 0.50),
                # Hips not visible (visibility=0)
                "23": FakeLandmark(0.40, 0.80, 0.0, visibility=0.0),
                "24": FakeLandmark(0.60, 0.80, 0.0, visibility=0.0),
            }
        )
        result = analyzer.analyze(pose, frame_shape, shoulders_visible=True)
        assert result["posture"] is not None
        assert result["posture"] < 60, (
            f"Fallback with forward head should score < 60, got {result['posture']}"
        )


class TestNoPose:
    """Test behaviour when no pose is detected."""

    def test_no_pose_returns_none(self, frame_shape):
        analyzer = PoseAnalyzer()
        result = analyzer.analyze(None, frame_shape)
        assert result["shoulder_alignment"] is None
        assert result["posture"] is None


class TestScoreRanges:
    """All scores should be in [0, 100] or None."""

    def test_all_in_range(self, frame_shape):
        analyzer = PoseAnalyzer()
        pose = make_pose_landmarks(33)
        result = analyzer.analyze(pose, frame_shape, shoulders_visible=True)
        for key, val in result.items():
            if val is not None:
                assert 0 <= val <= 100, f"{key} = {val} out of range"
