"""
Confidence Scoring Engine — public API.

Exports
-------
- ``ConfidenceEngine`` : high-level class wrapping frame processing and
  session-level statistics.  Can be used directly by another Python service
  (``from scoring import ConfidenceEngine``) without going through WebSocket.

Implements the Section 5 output contract:
- ``process_frame()`` returns ``composite_score``, ``signals``, ``visibility``,
  ``weights_used`` per frame.
- ``get_session_summary()`` returns aggregate stats.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .frame_processor import FrameProcessor
from .composite import CompositeScorer
from .face_analyzer import FaceAnalyzer
from .hand_analyzer import HandAnalyzer
from .pose_analyzer import PoseAnalyzer

__all__ = [
    "ConfidenceEngine",
    "FrameProcessor",
    "CompositeScorer",
    "FaceAnalyzer",
    "HandAnalyzer",
    "PoseAnalyzer",
]


class ConfidenceEngine:
    """Top-level entry point for real-time confidence scoring.

    Usage::

        engine = ConfidenceEngine()
        result = engine.process_frame(jpeg_bytes)
        # result["composite_score"]  →  float 0-100 or None
        # result["signals"]          →  dict of 9 sub-scores (None when gated)
        # result["visibility"]       →  {hands_visible, shoulders_visible}
        # result["weights_used"]     →  {signal: weight} post-redistribution

        summary = engine.get_session_summary()
    """

    def __init__(self) -> None:
        self._processor = FrameProcessor()
        self._score_history: List[float] = []
        self._signals_acc: Dict[str, List[float]] = {}
        self._frame_count: int = 0
        self.latest_result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Frame-level
    # ------------------------------------------------------------------
    def process_frame(self, jpeg_bytes: bytes) -> Dict[str, Any]:
        """Process a single JPEG frame and return the scoring result.

        Returns dict with keys: status, composite_score, signals,
        visibility, weights_used, raw_score, (error).
        """
        result = self._processor.process_frame(jpeg_bytes)

        self.latest_result = result
        self._frame_count += 1

        score = result.get("composite_score")
        if score is not None:
            self._score_history.append(score)

        # Accumulate per-signal values
        signals = result.get("signals", {})
        for key, val in signals.items():
            if val is not None:
                self._signals_acc.setdefault(key, []).append(val)

        return result

    # ------------------------------------------------------------------
    # Session summary
    # ------------------------------------------------------------------
    def get_session_summary(self) -> Dict[str, Any]:
        """Return aggregate statistics for the current session.

        Returns dict with avg_score, min_score, max_score, total_frames,
        low_confidence_count, and per-signal summaries (avg/min/max).
        """
        if not self._score_history:
            return {
                "average_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "total_frames": self._frame_count,
                "low_confidence_count": 0,
                "signals_summary": {},
            }

        try:
            import config
            threshold = getattr(config, "LOW_CONFIDENCE_THRESHOLD", 50)
        except ImportError:
            threshold = 50

        low_count = sum(1 for s in self._score_history if s < threshold)
        signals_summary = {}
        for key, vals in self._signals_acc.items():
            if vals:
                signals_summary[key] = {
                    "avg": round(sum(vals) / len(vals), 1),
                    "min": round(min(vals), 1),
                    "max": round(max(vals), 1),
                }

        return {
            "average_score": round(
                sum(self._score_history) / len(self._score_history), 1
            ),
            "min_score": round(min(self._score_history), 1),
            "max_score": round(max(self._score_history), 1),
            "total_frames": self._frame_count,
            "low_confidence_count": low_count,
            "signals_summary": signals_summary,
        }

    def reset(self) -> None:
        """Clear session state for reuse."""
        self._score_history.clear()
        self._signals_acc.clear()
        self._frame_count = 0
        self.latest_result = None
        self._processor.visibility.reset()
        self._processor.face_analyzer.reset_calibration()
        self._processor.hand_analyzer.reset()
        self._processor.composite.reset()

    def close(self) -> None:
        """Release the native MediaPipe resources owned by this engine."""
        self._processor.close()
