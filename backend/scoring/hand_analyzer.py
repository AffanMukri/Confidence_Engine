"""
Hand Analyzer — extracts hand-based confidence signals from MediaPipe Hand landmarks.

Implements Section 3 of the detection spec:
- Hand Stability (0-100): Wrist position variance over a rolling 2s window.
- Fidgeting (0-100): High-frequency (>1.5 Hz) oscillation / velocity zero-crossing
  count, distinguishing fidgeting from deliberate gestures.
- Hand Openness (0-100): Normalized finger-tip-to-wrist distance vs hand span.

All metrics return None when ``handsVisible`` is False (Section 1 gating).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class HandAnalyzer:
    """Analyzes hand landmarks for stability, fidgeting, and openness.

    Maintains rolling-window histories for wrist position, velocity, and
    openness scores across frames.

    Hand landmark indices (MediaPipe 21-point model):
        0: wrist
        4: thumb tip
        8: index finger tip
        12: middle finger tip
        16: ring finger tip
        20: pinky finger tip
        5: index MCP, 9: middle MCP, 13: ring MCP, 17: pinky MCP
    """

    # Landmark indices
    WRIST = 0
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17

    TIPS = [4, 8, 12, 16, 20]  # all 5 finger tips

    def __init__(self) -> None:
        try:
            import config  # noqa: E402

            self._stability_window = getattr(
                config, "HAND_STABILITY_WINDOW_SECONDS", 2.0
            )
            self._reference_range = getattr(
                config, "HAND_REFERENCE_RANGE", 0.05
            )
            self._fidget_window = getattr(
                config, "FIDGET_WINDOW_SECONDS", 10.0
            )
            self._fidget_min_hz = getattr(config, "FIDGET_MIN_FREQ_HZ", 1.5)
            self._fidget_max_reversals = getattr(
                config, "FIDGET_EXPECTED_MAX_REVERSALS", 50
            )
            self._noise_threshold = getattr(
                config, "FIDGET_NOISE_THRESHOLD_PX", 5.0
            )
        except ImportError:
            self._stability_window = 2.0
            self._reference_range = 0.05
            self._fidget_window = 10.0
            self._fidget_min_hz = 1.5
            self._fidget_max_reversals = 50
            self._noise_threshold = 5.0

        # ---- Rolling window state ----
        # Each entry: (timestamp, wrist_x, wrist_y, fingertip_distances)
        self._position_history: deque[tuple] = deque(maxlen=300)

        # ---- Fidgeting state ----
        # Track velocity sign for zero-crossing detection
        self._prev_vel_x: Optional[float] = None
        self._prev_vel_y: Optional[float] = None
        # Timestamps of direction reversals (zero-crossings above noise threshold)
        self._reversal_timestamps: deque[float] = deque(maxlen=100)

    def reset(self) -> None:
        """Clear all rolling-window state for a new session."""
        self._position_history.clear()
        self._prev_vel_x = None
        self._prev_vel_y = None
        self._reversal_timestamps.clear()

    @staticmethod
    def _to_px(lm_entry: Any, w: int, h: int) -> np.ndarray:
        """Convert a normalized hand landmark to pixel coordinates."""
        return np.array([lm_entry.x * w, lm_entry.y * h])

    def _compute_openness(
        self, hand_lm: Any, w: int, h: int
    ) -> float:
        """Compute openness for a single hand.

        Section 3 — Hand Openness:
        ``openness = avg(fingerTipToWristDistance) / handSpan``
        where handSpan = wrist-to-middle-MCP distance.

        Returns 0-100 score.
        """
        wrist = self._to_px(hand_lm[self.WRIST], w, h)
        mid_mcp = self._to_px(hand_lm[self.MIDDLE_MCP], w, h)
        hand_span = float(np.linalg.norm(mid_mcp - wrist))

        if hand_span < 1.0:
            return 50.0  # degenerate case

        # Average normalized tip-to-wrist distance for all 5 fingers
        tip_dists = []
        for tip_idx in self.TIPS:
            tip = self._to_px(hand_lm[tip_idx], w, h)
            dist = float(np.linalg.norm(tip - wrist))
            tip_dists.append(dist)

        avg_tip_dist = sum(tip_dists) / len(tip_dists)
        ratio = avg_tip_dist / hand_span

        # Ratio ~1.0 = fist, ~2.5+ = fully open hand
        openness = max(0.0, min(100.0, (ratio - 1.0) * 60.0))
        return openness

    def analyze(
        self,
        hand_landmarks_list: Optional[List[Any]],
        frame_shape: Tuple[int, ...],
        *,
        hands_visible: Optional[bool] = None,
    ) -> Dict[str, Optional[float]]:
        """Analyze hand landmarks and return confidence sub-scores.

        Parameters
        ----------
        hand_landmarks_list : list of LandmarkSet or None
        frame_shape : (height, width, channels)
        hands_visible : optional debounced visibility override

        Returns
        -------
        dict with keys: hand_stability, fidgeting, hand_openness
        (0-100 float or None when hands not visible)
        """
        # -- Visibility gate (Section 1) --
        if hands_visible is False:
            return {
                "hand_stability": None,
                "fidgeting": None,
                "hand_openness": None,
            }

        if not hand_landmarks_list:
            return {
                "hand_stability": None,
                "fidgeting": None,
                "hand_openness": None,
            }

        h, w = frame_shape[:2]

        try:
            # Aggregate across all detected hands — use the best (most visible)
            wrist_positions = []
            openness_scores = []

            for hand_landmarks in hand_landmarks_list:
                lm = (
                    hand_landmarks.landmark
                    if hasattr(hand_landmarks, "landmark")
                    else hand_landmarks
                )

                wrist = self._to_px(lm[self.WRIST], w, h)
                wrist_positions.append(wrist)

                open_score = self._compute_openness(lm, w, h)
                openness_scores.append(open_score)

            # Average across hands
            avg_wrist = np.mean(wrist_positions, axis=0)
            avg_openness = float(np.mean(openness_scores)) if openness_scores else 50.0
            now = time.time()

            # ---- Update rolling history ----
            self._position_history.append(
                (now, avg_wrist[0], avg_wrist[1], avg_openness)
            )

            # ---- Prune old entries (keep only within windows) ----
            self._prune_history(now)

            # ================================================================
            # Hand Stability (0-100) — Section 3
            # ================================================================
            stability = self._compute_stability(now)

            # ================================================================
            # Fidgeting (0-100) — Section 3
            # ================================================================
            fidgeting = self._compute_fidgeting(now)

            # ================================================================
            # Hand Openness (0-100) — Section 3
            # ================================================================
            openness = avg_openness

            return {
                "hand_stability": float(stability),
                "fidgeting": float(fidgeting),
                "hand_openness": float(openness),
            }

        except (IndexError, AttributeError, TypeError) as exc:
            logger.debug("Hand analysis failed: %s", exc)
            return {
                "hand_stability": None,
                "fidgeting": None,
                "hand_openness": None,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prune_history(self, now: float) -> None:
        """Remove entries older than the combined window needed."""
        max_window = max(self._stability_window, self._fidget_window)
        cutoff = now - max_window - 1.0  # keep a buffer
        while self._position_history and self._position_history[0][0] < cutoff:
            self._position_history.popleft()

    def _compute_stability(self, now: float) -> float:
        """Compute hand stability from wrist position variance.

        Section 3 — Hand Stability:
        ``stability = 100 * (1 - clamp(stddev(wristPositions) / referenceRange, 0, 1))``
        where referenceRange ~0.05 of frame width.
        """
        cutoff = now - self._stability_window
        recent = [p for p in self._position_history if p[0] >= cutoff]

        if len(recent) < 3:
            return 100.0  # not enough data, assume stable

        wrist_xs = [p[1] for p in recent]
        wrist_ys = [p[2] for p in recent]

        # Standard deviation of position in pixels.
        # Scale reference range to frame size since wrist coords are in pixels.
        std_x = np.std(wrist_xs) if len(set(wrist_xs)) > 1 else 0.0
        std_y = np.std(wrist_ys) if len(set(wrist_ys)) > 1 else 0.0
        std_magnitude = np.sqrt(std_x ** 2 + std_y ** 2)

        # reference_range ≈ 0.05 * frame_dimension (~48px for 960px frame)
        # — anything below that is natural landmark jitter / small gestures.
        frame_ref = self._reference_range * max(960.0, 1.0)  # approximate frame width
        stability = 100.0 * (1.0 - min(1.0, max(0.0, std_magnitude / frame_ref)))
        return max(0.0, min(100.0, float(stability)))

    def _compute_fidgeting(self, now: float) -> float:
        """Compute fidgeting from high-frequency direction reversals.

        Section 3 — Fidgeting:
        - Track velocity zero-crossings of wrist position.
        - Count reversals within a rolling window.
        - Only count reversals where the velocity magnitude exceeds a noise
          threshold (to ignore tiny jitter).
        - Distinguish from deliberate gestures by requiring the movement to
          be high-frequency (>1.5 Hz) and small-amplitude.
        - ``fidgetScore = clamp(100 * (reversalCount / expectedMaxReversals), 0, 100)``
        """
        cutoff = now - self._fidget_window
        # Need at least 3 recent entries for velocity computation
        recent = [p for p in self._position_history if p[0] >= cutoff]

        if len(recent) < 3:
            return 100.0  # not enough data, assume not fidgeting

        # Compute velocities (frame-to-frame wrist displacement)
        xs = [p[1] for p in recent]
        ys = [p[2] for p in recent]
        ts = [p[0] for p in recent]

        velocities = []
        for i in range(1, len(recent)):
            dt = ts[i] - ts[i - 1]
            if dt > 0:
                vx = (xs[i] - xs[i - 1]) / dt
                vy = (ys[i] - ys[i - 1]) / dt
                speed = np.sqrt(vx ** 2 + vy ** 2)
                velocities.append((vx, vy, speed, dt))

        if len(velocities) < 3:
            return 100.0

        # Count zero-crossings above noise threshold
        reversal_count = 0
        for i in range(1, len(velocities)):
            prev_vx, prev_vy, prev_speed, _ = velocities[i - 1]
            curr_vx, curr_vy, curr_speed, _ = velocities[i]

            # Require movement above noise threshold
            if prev_speed < self._noise_threshold:  # px/s — filter jitter
                continue
            if curr_speed < self._noise_threshold:
                continue

            # Check for direction reversal (sign change in velocity)
            # Use a small tolerance to avoid counting noise
            if prev_vx * curr_vx < -1e-6 or prev_vy * curr_vy < -1e-6:
                reversal_count += 1

        # Compute frequency: reversals / window_seconds
        window_duration = ts[-1] - ts[0]
        if window_duration <= 0:
            return 100.0

        reversal_freq = reversal_count / window_duration

        # Only count as fidgeting if frequency ≥ min_fidget_hz
        # This distinguishes fidgeting (rapid small oscillations > 1.5 Hz)
        # from deliberate gestures (slower, larger movements).
        if reversal_freq < self._fidget_min_hz:
            return 100.0

        # Map reversal count to 0-100 score
        # Higher reversal_count = more fidgeting = lower score
        fidget_raw = min(1.0, reversal_count / self._fidget_max_reversals)
        fidget_score = 100.0 * (1.0 - fidget_raw)
        return max(0.0, min(100.0, float(fidget_score)))
