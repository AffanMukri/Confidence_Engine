"""
Composite Scorer — aggregates sub-scores using dynamic weight redistribution.

Implements Section 2 of the detection spec:
- ``computeComposite(metrics, weights)``: Only visible/non-null metrics contribute.
- Removed weights are redistributed proportionally among remaining metrics.
- Composite score smoothed with EMA (alpha ≈ 0.2).
- Exposes ``weightsUsed`` per frame for debugging/UI.
"""

from __future__ import annotations

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CompositeScorer:
    """Aggregates sub-scores using dynamic weights and EMA smoothing.

    Section 2 — Dynamic Reweighting:
    ```js
    function computeComposite(metrics, weights) {
      const visible = Object.keys(weights).filter(k => metrics[k] !== null);
      const totalWeight = visible.reduce((sum, k) => sum + weights[k], 0);
      const score = visible.reduce(
        (sum, k) => sum + (metrics[k] * (weights[k] / totalWeight)), 0
      );
      return Math.round(score);
    }
    ```
    EMA smooths the result (alpha ≈ 0.2) so it doesn't jump per-frame.
    """

    def __init__(self) -> None:
        self.smoothed_score: Optional[float] = None
        self._weights: Dict[str, float] = {}

        try:
            import config  # noqa: E402

            self._weights = dict(getattr(config, "WEIGHTS", {}))
            self._ema_alpha = getattr(config, "EMA_ALPHA", 0.2)
        except (ImportError, AttributeError):
            # Fallback default weights
            self._weights = {
                "eye_contact": 0.20,
                "blink_rate": 0.10,
                "brow_tension": 0.10,
                "smile": 0.10,
                "hand_stability": 0.12,
                "fidgeting": 0.12,
                "hand_openness": 0.06,
                "shoulder_alignment": 0.10,
                "posture": 0.10,
            }
            self._ema_alpha = 0.2

    def calculate(self, signals: Dict[str, Optional[float]]) -> Dict[str, Any]:
        """Compute composite score with dynamic reweighting.

        Parameters
        ----------
        signals : dict of signal_name → float or None
            None means the metric is excluded due to visibility gating.

        Returns
        -------
        dict with keys:
            status: "ok" | "no_face"
            composite_score: float or None
            raw_score: float or None (pre-smoothing)
            weights_used: dict of signal_name → weight (post-redistribution)
        """
        # ---- Filter visible signals (Section 2) ----
        visible = {
            key: val
            for key, val in signals.items()
            if val is not None and key in self._weights
        }

        if not visible:
            return {
                "status": "no_face",
                "composite_score": None,
                "raw_score": None,
                "weights_used": {},
            }

        # ---- Dynamic reweighting ----
        total_weight = sum(self._weights[k] for k in visible)
        if total_weight == 0:
            return {
                "status": "no_face",
                "composite_score": None,
                "raw_score": None,
                "weights_used": {},
            }

        raw_score = 0.0
        weights_used: Dict[str, float] = {}
        for key, val in visible.items():
            normalized_weight = self._weights[key] / total_weight
            raw_score += val * normalized_weight
            weights_used[key] = round(normalized_weight, 4)

        # ---- EMA smoothing (Section 2) ----
        if self.smoothed_score is None:
            self.smoothed_score = raw_score
        else:
            a = self._ema_alpha
            self.smoothed_score = a * raw_score + (1.0 - a) * self.smoothed_score

        return {
            "status": "ok",
            "composite_score": float(self.smoothed_score),
            "raw_score": float(raw_score),
            "weights_used": weights_used,
        }

    def reset(self) -> None:
        """Clear EMA state for a new session."""
        self.smoothed_score = None
