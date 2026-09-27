"""
Unit tests for HandAnalyzer.

Tests stability variance, fidget/jerk detection, and hand openness ratio
using synthetic landmark trajectories.
"""

import numpy as np
import pytest

from scoring.hand_analyzer import HandAnalyzer
from conftest import FakeLandmark, FakeLandmarkList, make_hand_landmarks, FRAME_SHAPE


class TestHandStability:
    """Test hand stability scoring (velocity variance)."""

    def test_steady_hand(self, frame_shape):
        """Hands held perfectly still → high stability score."""
        analyzer = HandAnalyzer()

        # Send 10 frames of identical hand positions
        for _ in range(10):
            hand = make_hand_landmarks(
                21,
                **{
                    "0": (0.5, 0.5),   # wrist
                    "5": (0.5, 0.4),   # index MCP
                    "8": (0.5, 0.3),   # index tip
                    "9": (0.5, 0.4),   # middle MCP
                    "12": (0.5, 0.3),  # middle tip
                }
            )
            result = analyzer.analyze([hand], frame_shape, hands_visible=True)

        assert result["hand_stability"] is not None
        assert result["hand_stability"] >= 90, (
            f"Steady hand should score >= 90, got {result['hand_stability']}"
        )

    def test_jittery_hand(self, frame_shape):
        """Rapidly moving hands → lower stability score."""
        analyzer = HandAnalyzer()

        # Send 15 frames with alternating positions (jitter)
        for i in range(15):
            offset = 0.1 * (i % 2)  # alternate between 0 and 0.1
            hand = make_hand_landmarks(
                21,
                **{
                    "0": (0.5 + offset, 0.5 - offset),
                    "5": (0.5 + offset, 0.4),
                    "8": (0.5 + offset, 0.3),
                    "9": (0.5 + offset, 0.4),
                    "12": (0.5 + offset, 0.3),
                }
            )
            result = analyzer.analyze([hand], frame_shape, hands_visible=True)

        assert result["hand_stability"] is not None
        # Jittery hands should score lower than steady
        assert result["hand_stability"] < 95


class TestFidgeting:
    """Test fidget/jerk detection."""

    def test_no_fidgeting(self, frame_shape):
        """Slow, smooth motion → low jerk → high fidgeting score."""
        analyzer = HandAnalyzer()

        for i in range(10):
            x = 0.5 + i * 0.005  # very slow drift
            hand = make_hand_landmarks(
                21,
                **{
                    "0": (x, 0.5),
                    "5": (x, 0.4),
                    "8": (x, 0.3),
                    "9": (x, 0.4),
                    "12": (x, 0.3),
                }
            )
            result = analyzer.analyze([hand], frame_shape, hands_visible=True)

        assert result["fidgeting"] is not None
        assert result["fidgeting"] >= 80, (
            f"Smooth motion should give high fidgeting score, got {result['fidgeting']}"
        )


class TestHandOpenness:
    """Test open vs. closed hand ratio."""

    def test_open_hand(self, frame_shape):
        """Fingers extended → high openness score."""
        analyzer = HandAnalyzer()
        hand = make_hand_landmarks(
            21,
            **{
                "0": (0.5, 0.7),    # wrist
                "5": (0.5, 0.55),   # index MCP
                "8": (0.5, 0.2),    # index tip (far from wrist)
                "9": (0.5, 0.55),   # middle MCP
                "12": (0.5, 0.2),   # middle tip (far from wrist)
            }
        )
        result = analyzer.analyze([hand], frame_shape, hands_visible=True)
        assert result["hand_openness"] is not None
        assert result["hand_openness"] > 50, (
            f"Open hand should score > 50, got {result['hand_openness']}"
        )

    def test_closed_fist(self, frame_shape):
        """Fingers curled in → low openness score."""
        analyzer = HandAnalyzer()
        hand = make_hand_landmarks(
            21,
            **{
                "0": (0.5, 0.7),    # wrist
                "5": (0.5, 0.6),    # index MCP
                "8": (0.5, 0.65),   # index tip (close to wrist — curled)
                "9": (0.5, 0.6),    # middle MCP
                "12": (0.5, 0.65),  # middle tip (close to wrist — curled)
            }
        )
        result = analyzer.analyze([hand], frame_shape, hands_visible=True)
        assert result["hand_openness"] is not None
        # Closed fist has low openness
        assert result["hand_openness"] < 50, (
            f"Closed fist should score < 50, got {result['hand_openness']}"
        )


class TestNoHands:
    """Test behaviour when no hands are detected."""

    def test_no_hands_returns_none(self, frame_shape):
        analyzer = HandAnalyzer()
        result = analyzer.analyze(None, frame_shape)
        assert result["hand_stability"] is None
        assert result["fidgeting"] is None
        assert result["hand_openness"] is None

    def test_empty_list_returns_none(self, frame_shape):
        analyzer = HandAnalyzer()
        result = analyzer.analyze([], frame_shape)
        assert result["hand_stability"] is None


class TestScoreRanges:
    """All scores should be in [0, 100] or None."""

    def test_all_in_range(self, frame_shape):
        analyzer = HandAnalyzer()
        hand = make_hand_landmarks(21)
        for _ in range(5):
            result = analyzer.analyze([hand], frame_shape)
        for key, val in result.items():
            if val is not None:
                assert 0 <= val <= 100, f"{key} = {val} out of range"
