"""
Unit tests for CompositeScorer.

Tests weighted scoring math, EMA smoothing, graceful degradation
when signal groups are missing, and boundary values.
"""

import pytest

from scoring.composite import CompositeScorer


class TestWeightedScoring:
    """Test basic weighted combination."""

    def test_all_signals_100(self):
        """If every signal is 100, composite should be 100."""
        scorer = CompositeScorer()
        signals = {
            "eye_contact": 100.0,
            "blink_rate": 100.0,
            "brow_tension": 100.0,
            "smile": 100.0,
            "hand_stability": 100.0,
            "fidgeting": 100.0,
            "hand_openness": 100.0,
            "shoulder_alignment": 100.0,
            "posture": 100.0,
        }
        result = scorer.calculate(signals)
        assert result["status"] == "ok"
        assert abs(result["composite_score"] - 100.0) < 0.01
        # Check weights_used key exists (Section 5)
        assert "weights_used" in result
        # All 9 weights should sum to 1.0
        assert abs(sum(result["weights_used"].values()) - 1.0) < 0.01

    def test_all_signals_0(self):
        """If every signal is 0, composite should be 0."""
        scorer = CompositeScorer()
        signals = {
            "eye_contact": 0.0,
            "blink_rate": 0.0,
            "brow_tension": 0.0,
            "smile": 0.0,
            "hand_stability": 0.0,
            "fidgeting": 0.0,
            "hand_openness": 0.0,
            "shoulder_alignment": 0.0,
            "posture": 0.0,
        }
        result = scorer.calculate(signals)
        assert result["status"] == "ok"
        assert abs(result["composite_score"]) < 0.01

    def test_mixed_signals(self):
        """Verify weighted math with known values."""
        scorer = CompositeScorer()
        signals = {
            "eye_contact": 80.0,      # weight 0.20
            "blink_rate": 60.0,       # weight 0.10
            "brow_tension": 70.0,     # weight 0.10
            "smile": 90.0,           # weight 0.10
            "hand_stability": 50.0,   # weight 0.12
            "fidgeting": 40.0,       # weight 0.12
            "hand_openness": 100.0,   # weight 0.06
            "shoulder_alignment": 85.0,  # weight 0.10
            "posture": 75.0,         # weight 0.10
        }
        result = scorer.calculate(signals)
        assert result["status"] == "ok"
        # Expected weighted average:
        # 80*0.20 + 60*0.10 + 70*0.10 + 90*0.10 + 50*0.12 + 40*0.12 +
        # 100*0.06 + 85*0.10 + 75*0.10
        expected = (
            80 * 0.20 + 60 * 0.10 + 70 * 0.10 + 90 * 0.10
            + 50 * 0.12 + 40 * 0.12 + 100 * 0.06
            + 85 * 0.10 + 75 * 0.10
        )
        # First call → EMA = raw score (alpha init)
        assert abs(result["composite_score"] - expected) < 0.5
        # Verify weights_used has all 9 metrics and sums to 1.0
        assert "weights_used" in result
        assert len(result["weights_used"]) == 9
        assert abs(sum(result["weights_used"].values()) - 1.0) < 0.01


class TestEMASmoothing:
    """Test exponential moving average behaviour."""

    def test_ema_smoothing_reduces_jitter(self):
        """Alternating high/low scores should converge to a middle range."""
        scorer = CompositeScorer()
        all_100 = {k: 100.0 for k in [
            "eye_contact", "blink_rate", "brow_tension", "smile",
            "hand_stability", "fidgeting", "hand_openness",
            "shoulder_alignment", "posture",
        ]}
        all_0 = {k: 0.0 for k in all_100}

        scores = []
        for i in range(20):
            sig = all_100 if i % 2 == 0 else all_0
            r = scorer.calculate(sig)
            scores.append(r["composite_score"])

        # After 20 alternations, score should be in a middle band
        assert 30 < scores[-1] < 70, (
            f"EMA should converge to ~50, got {scores[-1]}"
        )

    def test_ema_first_value_equals_raw(self):
        """First call should set EMA = raw (no smoothing yet)."""
        scorer = CompositeScorer()
        signals = {k: 75.0 for k in [
            "eye_contact", "blink_rate", "brow_tension", "smile",
            "hand_stability", "fidgeting", "hand_openness",
            "shoulder_alignment", "posture",
        ]}
        result = scorer.calculate(signals)
        assert abs(result["composite_score"] - 75.0) < 0.01
        assert "weights_used" in result


class TestGracefulDegradation:
    """Test weight redistribution when signals are missing."""

    def test_missing_hand_signals(self):
        """Missing hand signals → face+pose weights redistributed."""
        scorer = CompositeScorer()
        signals = {
            "eye_contact": 80.0,
            "blink_rate": 80.0,
            "brow_tension": 80.0,
            "smile": 80.0,
            "hand_stability": None,
            "fidgeting": None,
            "hand_openness": None,
            "shoulder_alignment": 80.0,
            "posture": 80.0,
        }
        result = scorer.calculate(signals)
        assert result["status"] == "ok"
        # Should still produce a valid score from face+pose
        assert abs(result["composite_score"] - 80.0) < 1.0
        # Verify weights_used only has 6 metrics (no hand signals)
        assert "weights_used" in result
        assert len(result["weights_used"]) == 6
        assert abs(sum(result["weights_used"].values()) - 1.0) < 0.01

    def test_missing_pose_signals(self):
        """Missing pose signals → face+hand weights redistributed."""
        scorer = CompositeScorer()
        signals = {
            "eye_contact": 60.0,
            "blink_rate": 60.0,
            "brow_tension": 60.0,
            "smile": 60.0,
            "hand_stability": 60.0,
            "fidgeting": 60.0,
            "hand_openness": 60.0,
            "shoulder_alignment": None,
            "posture": None,
        }
        result = scorer.calculate(signals)
        assert result["status"] == "ok"
        assert abs(result["composite_score"] - 60.0) < 1.0
        # Verify weights_used only has 7 metrics (no pose signals)
        assert "weights_used" in result
        assert len(result["weights_used"]) == 7
        assert abs(sum(result["weights_used"].values()) - 1.0) < 0.01

    def test_all_signals_none(self):
        """All signals None → status no_face, score null."""
        scorer = CompositeScorer()
        signals = {
            "eye_contact": None,
            "blink_rate": None,
            "brow_tension": None,
            "smile": None,
            "hand_stability": None,
            "fidgeting": None,
            "hand_openness": None,
            "shoulder_alignment": None,
            "posture": None,
        }
        result = scorer.calculate(signals)
        assert result["status"] == "no_face"
        assert result["composite_score"] is None
        assert result["weights_used"] == {}

    def test_only_eye_contact_available(self):
        """Only one signal available → full weight on it."""
        scorer = CompositeScorer()
        signals = {
            "eye_contact": 90.0,
            "blink_rate": None,
            "brow_tension": None,
            "smile": None,
            "hand_stability": None,
            "fidgeting": None,
            "hand_openness": None,
            "shoulder_alignment": None,
            "posture": None,
        }
        result = scorer.calculate(signals)
        assert result["status"] == "ok"
        assert abs(result["composite_score"] - 90.0) < 0.5
        # Only eye_contact visible → weight should be 1.0
        assert result["weights_used"] == {"eye_contact": 1.0}


class TestBoundaryValues:
    """Test edge cases at score boundaries."""

    def test_composite_never_exceeds_100(self):
        scorer = CompositeScorer()
        signals = {k: 100.0 for k in [
            "eye_contact", "blink_rate", "brow_tension", "smile",
            "hand_stability", "fidgeting", "hand_openness",
            "shoulder_alignment", "posture",
        ]}
        for _ in range(10):
            result = scorer.calculate(signals)
        # Allow tiny floating-point epsilon
        assert result["composite_score"] <= 100.0 + 1e-6

    def test_composite_never_below_0(self):
        scorer = CompositeScorer()
        signals = {k: 0.0 for k in [
            "eye_contact", "blink_rate", "brow_tension", "smile",
            "hand_stability", "fidgeting", "hand_openness",
            "shoulder_alignment", "posture",
        ]}
        for _ in range(10):
            result = scorer.calculate(signals)
        assert result["composite_score"] >= 0.0

    def test_empty_signals_dict(self):
        """Passing an empty dict → no_face status."""
        scorer = CompositeScorer()
        result = scorer.calculate({})
        assert result["status"] == "no_face"
        assert result["composite_score"] is None
        assert result["weights_used"] == {}
