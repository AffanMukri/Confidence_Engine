"""
Face Analyzer — extracts confidence signals from MediaPipe Face Mesh landmarks.

Implements Section 3 of the detection spec:
- Eye Contact (0-100): Combines iris gaze ratio with head pose (yaw/pitch).
- Blink Rate (0-100): EAR-based blink detection, bell-curve scoring centred on 17 blinks/min.
- Brow Tension (0-100): Blendshape-driven with geometric fallback + baseline calibration.
- Smile (0-100): Blendshape-driven with geometric fallback + EMA smoothing.

Uses 478 landmarks (refine_landmarks=True for iris tracking).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FaceAnalyzer:
    """Extracts facial confidence signals from MediaPipe Face Mesh landmarks.

    Maintains internal state for blink tracking, EMA smoothing, and
    baseline calibration across frames.
    """

    # Landmark index constants
    LEFT_EYE_EAR = [362, 385, 386, 263, 374, 380]
    RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 145]

    # Iris landmarks (refine_landmarks=True)
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473

    # Eye corners for gaze reference
    LEFT_EYE_INNER = 362
    LEFT_EYE_OUTER = 263
    RIGHT_EYE_INNER = 133
    RIGHT_EYE_OUTER = 33

    # Brow landmarks (geometric fallback)
    LEFT_INNER_BROW = 65
    RIGHT_INNER_BROW = 295
    UPPER_NOSE = 168
    FOREHEAD = 10
    CHIN = 152

    # Mouth / face width
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    FACE_LEFT = 234
    FACE_RIGHT = 454

    def __init__(
        self,
        ear_threshold: float = 0.21,
        blink_consec_frames: int = 3,
        calibration_frames: int = 30,
    ) -> None:
        # Try to pull thresholds from config
        try:
            import config  # noqa: E402

            self.ear_threshold = getattr(config, "EAR_THRESHOLD", ear_threshold)
            self.blink_consec_frames = getattr(
                config, "BLINK_CONSEC_FRAMES", blink_consec_frames
            )
            self._gaze_ema_alpha = getattr(config, "FACE_GAZE_EMA_ALPHA", 0.3)
            self._smile_ema_alpha = getattr(config, "FACE_SMILE_EMA_ALPHA", 0.25)
            self._bell_center = getattr(config, "BLINK_BELL_CURVE_CENTER", 17.0)
            self._bell_sigma = getattr(config, "BLINK_BELL_CURVE_SIGMA", 10.0)
            self._blink_window = getattr(
                config, "BLINK_RATE_WINDOW_SECONDS", 60.0
            )
            self._yaw_limit = getattr(config, "HEAD_POSE_YAW_LIMIT", 15.0)
            self._pitch_limit = getattr(config, "HEAD_POSE_PITCH_LIMIT", 15.0)
            self._cal_frames = getattr(
                config, "BASELINE_CALIBRATION_FRAMES", calibration_frames
            )
        except ImportError:
            self.ear_threshold = ear_threshold
            self.blink_consec_frames = blink_consec_frames
            self._gaze_ema_alpha = 0.3
            self._smile_ema_alpha = 0.25
            self._bell_center = 17.0
            self._bell_sigma = 10.0
            self._blink_window = 60.0
            self._yaw_limit = 15.0
            self._pitch_limit = 15.0
            self._cal_frames = calibration_frames

        # -- Blink state machine (Section 3 — Blink Rate) --
        self._blink_timestamps: deque[float] = deque(maxlen=200)
        self._consec_below: int = 0
        self._last_blink_was_below: bool = (
            False  # rising-edge detection: only count 1 blink per dip
        )

        # -- EMA smoothers (Section 2 style, per-signal) --
        self._gaze_smoothed: Optional[float] = None
        self._smile_smoothed: Optional[float] = None

        # -- Baseline calibration (Section 3 — brow, smile fallback) --
        # Collect neutral-brow vertical distances and mouth width ratios
        # over the first ~cal_frames frames of the session.
        self._cal_brow_dists: deque[float] = deque(maxlen=self._cal_frames)
        self._cal_mouth_ratios: deque[float] = deque(maxlen=self._cal_frames)
        self._cal_frame_count: int = 0
        self._baseline_brow_dist: Optional[float] = None  # calibrated neutral
        self._baseline_mouth_ratio: Optional[float] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _lm_to_px(lm: Any, idx: int, w: int, h: int) -> np.ndarray:
        """Convert a single normalized landmark to pixel coordinates."""
        return np.array([lm[idx].x * w, lm[idx].y * h])

    @staticmethod
    def _ear(pts: np.ndarray) -> float:
        """Eye Aspect Ratio from 6 ordered points.

        pts shape: (6, 2)  — [outer_corner, upper_inner, upper_outer,
                              inner_corner, lower_outer, lower_inner]
        """
        p2_p6 = np.linalg.norm(pts[1] - pts[5])
        p3_p5 = np.linalg.norm(pts[2] - pts[4])
        p1_p4 = np.linalg.norm(pts[0] - pts[3])
        if p1_p4 == 0:
            return 0.0
        return float((p2_p6 + p3_p5) / (2.0 * p1_p4))

    @staticmethod
    def _rotation_from_matrix(M: np.ndarray) -> Tuple[float, float, float]:
        """Extract yaw, pitch, roll (degrees) from a 4x4 transformation matrix.

        MediaPipe's ``facial_transformation_matrixes`` returns a 4x4 where the
        top-left 3x3 is the rotation matrix.  The coordinate system has:
        - X pointing right (from subject's perspective)
        - Y pointing up
        - Z pointing outward from face (toward camera)

        We decompose using the YXZ convention (common in face-pose):
        - Yaw   = rotation around Y axis (head turning left/right)
        - Pitch = rotation around X axis (head tilting up/down)
        - Roll  = rotation around Z axis (head tilting sideways)
        """
        R = M[:3, :3]
        # Yaw (Y-axis): turning left/right
        yaw = np.degrees(np.arctan2(R[0, 2], R[2, 2]))
        # Pitch (X-axis): nodding up/down
        pitch = np.degrees(np.arcsin(np.clip(-R[1, 2], -1.0, 1.0)))
        # Roll (Z-axis): head tilt sideways
        roll = np.degrees(np.arctan2(R[1, 0], R[1, 1]))
        return yaw, pitch, roll

    @staticmethod
    def _get_blendshape_score(
        categories: Any, name: str, default: float = 0.0
    ) -> float:
        """Look up a blendshape score by name from MediaPipe's category list.

        ``categories`` is a list of objects with ``.category_name`` and ``.score``.
        """
        for cat in categories:
            if cat.category_name == name:
                return cat.score
        return default

    # ------------------------------------------------------------------
    # Sub-signal calculators — Section 3
    # ------------------------------------------------------------------
    def _eye_contact(
        self,
        lm: Any,
        w: int,
        h: int,
        face_transform_matrix: Optional[np.ndarray],
    ) -> Optional[float]:
        """Score 0-100 combining iris gaze ratio and head pose.

        Section 3 — Eye Contact:
        - ``gazeScore = 1 - clamp(distance(irisCenter, eyeBoxCenter) / (eyeBoxWidth/2), 0, 1)``
        - ``headScore = 1 - clamp(max(|yaw|,|pitch|) / 30, 0, 1)``
        - ``eyeContact = round(100 * (0.6*gazeScore + 0.4*headScore))``
        - Smoothed with EMA alpha ≈ 0.3
        """
        try:
            # ---- Gaze component ----
            # Measure how far each iris center deviates from the eye center
            # (midpoint of inner and outer corners).  This is symmetric and
            # gives 0 deviation when looking straight regardless of eye shape.
            l_iris = self._lm_to_px(lm, self.LEFT_IRIS_CENTER, w, h)
            r_iris = self._lm_to_px(lm, self.RIGHT_IRIS_CENTER, w, h)
            l_inner = self._lm_to_px(lm, self.LEFT_EYE_INNER, w, h)
            l_outer = self._lm_to_px(lm, self.LEFT_EYE_OUTER, w, h)
            r_inner = self._lm_to_px(lm, self.RIGHT_EYE_INNER, w, h)
            r_outer = self._lm_to_px(lm, self.RIGHT_EYE_OUTER, w, h)

            l_width = float(np.linalg.norm(l_outer - l_inner))
            r_width = float(np.linalg.norm(r_outer - r_inner))

            # Use eye center (midpoint of inner+outer) as the reference.
            # Deviation = distance(iris, eye_center) / (eye_width / 2)
            if l_width > 0:
                l_eye_center = (l_inner + l_outer) / 2.0
                l_deviation = float(np.linalg.norm(l_iris - l_eye_center)) / (l_width / 2.0)
            else:
                l_deviation = 0.0
            if r_width > 0:
                r_eye_center = (r_inner + r_outer) / 2.0
                r_deviation = float(np.linalg.norm(r_iris - r_eye_center)) / (r_width / 2.0)
            else:
                r_deviation = 0.0

            avg_deviation = (l_deviation + r_deviation) / 2.0
            gaze_score = 1.0 - min(1.0, max(0.0, avg_deviation))

            # ---- Head pose component ----
            head_score = 1.0
            if face_transform_matrix is not None:
                yaw, pitch, _ = self._rotation_from_matrix(face_transform_matrix)
                # headScore = 1 - clamp(max(|yaw|,|pitch|) / 35, 0, 1)
                # Divisor 35° (not 30°) — more forgiving for natural laptop use
                max_angle = max(abs(yaw), abs(pitch))
                head_score = 1.0 - min(1.0, max(0.0, max_angle / 35.0))

            # ---- Combine (60% gaze, 40% head pose) ----
            raw = 100.0 * (0.6 * gaze_score + 0.4 * head_score)
            raw = max(0.0, min(100.0, raw))

            # ---- EMA smooth (alpha ≈ 0.3) ----
            if self._gaze_smoothed is None:
                self._gaze_smoothed = raw
            else:
                a = self._gaze_ema_alpha
                self._gaze_smoothed = a * raw + (1.0 - a) * self._gaze_smoothed

            return self._gaze_smoothed

        except (IndexError, AttributeError, TypeError) as exc:
            logger.debug("Eye contact computation failed: %s", exc)
            return None

    def _blink_rate(self, lm: Any, w: int, h: int) -> Optional[float]:
        """Score 0-100 based on blinks per minute using EAR-based detection.

        Section 3 — Blink Rate:
        - EAR dip below threshold (≈0.21) for 1-4 frames triggers rising-edge detection
        - ``rate = (blinkCount / elapsedSeconds) * 60``
        - ``score = 100 * exp(-((rate-17)^2)/(2*10^2))``, clamped 0-100
        - Returns None when no observation time has elapsed.
        """
        try:
            # Compute average EAR for this frame
            left_pts = np.array(
                [self._lm_to_px(lm, i, w, h) for i in self.LEFT_EYE_EAR]
            )
            right_pts = np.array(
                [self._lm_to_px(lm, i, w, h) for i in self.RIGHT_EYE_EAR]
            )
            avg_ear = (self._ear(left_pts) + self._ear(right_pts)) / 2.0

            # ---- Rising-edge blink detection ----
            currently_below = avg_ear < self.ear_threshold
            if currently_below:
                self._consec_below += 1
            else:
                # Rising edge: was below threshold for enough frames
                if (
                    self._consec_below >= self.blink_consec_frames
                    and not self._last_blink_was_below
                ):
                    self._blink_timestamps.append(time.time())
                    self._last_blink_was_below = True
                elif self._consec_below == 0:
                    # Fully open, reset the edge flag for next dip
                    self._last_blink_was_below = False
                self._consec_below = 0

            # ---- Compute blinks per minute ----
            now = time.time()
            # Use rolling window or session elapsed for rate
            if len(self._blink_timestamps) > 0:
                # Find blinks within the rolling window
                window_start = now - self._blink_window
                recent_blinks = sum(
                    1 for t in self._blink_timestamps if t >= window_start
                )

                # If we have at least 5 seconds of data, use the rolling window rate
                elapsed = now - window_start
                if recent_blinks > 0 and elapsed >= 5.0:
                    rate = (recent_blinks / elapsed) * 60.0
                else:
                    # Fall back to extrapolation from session start
                    session_elapsed = now - self._blink_timestamps[0]
                    if session_elapsed >= 5.0:
                        rate = (len(self._blink_timestamps) / session_elapsed) * 60.0
                    else:
                        # Too early — return neutral score
                        return 80.0
            else:
                # No blinks detected yet — return neutral score if camera is on
                return 80.0

            # ---- Bell-curve scoring (Section 3: centered on 17 blinks/min) ----
            # score = 100 * exp(-((rate-17)^2)/(2*10^2))
            score = 100.0 * np.exp(
                -((rate - self._bell_center) ** 2) / (2.0 * self._bell_sigma ** 2)
            )
            score = max(0.0, min(100.0, float(score)))
            return score

        except (IndexError, AttributeError, TypeError) as exc:
            logger.debug("Blink rate computation failed: %s", exc)
            return None

    def _brow_tension(
        self,
        lm: Any,
        w: int,
        h: int,
        blendshape_categories: Optional[Any],
    ) -> Optional[float]:
        """Score 0-100. Higher = more relaxed, Lower = furrowed/tense.

        Section 3 — Brow Tension (NOTE: we invert the raw signal so that
        the UI's "higher is better" convention is preserved).
        - Primary: blendshapes ``browDownLeft``, ``browDownRight`` averaged.
        - Fallback: geometric brow-to-nose distance with baseline calibration.
        - ``tension = round(100 * avg(browDownLeft, browDownRight))`` but inverted:
          raw tension (furrowed) → low score (0), relaxed brows → high score (100).
        """
        try:
            # ---- Primary: blendshapes ----
            if blendshape_categories is not None:
                bdl = self._get_blendshape_score(
                    blendshape_categories, "browDownLeft"
                )
                bdr = self._get_blendshape_score(
                    blendshape_categories, "browDownRight"
                )
                brow_down_avg = (bdl + bdr) / 2.0

                # browDownLeft/Right are 0=relaxed, 1=fully down (furrowed).
                # Invert so higher = relaxed = better for "Confidence" gauge.
                # Also incorporate browInnerUp: if brows are raised (surprise),
                # that's also tension, so we add that to the tension signal.
                biu = self._get_blendshape_score(
                    blendshape_categories, "browInnerUp"
                )
                tension = max(brow_down_avg, biu)  # worst-case tension
                score = 100.0 * (1.0 - tension)
                score = max(0.0, min(100.0, score))
                return score

            # ---- Fallback: geometric with baseline calibration ----

            brow_y = (
                lm[self.LEFT_INNER_BROW].y + lm[self.RIGHT_INNER_BROW].y
            ) / 2.0 * h
            nose_y = lm[self.UPPER_NOSE].y * h
            face_h = abs(lm[self.FOREHEAD].y - lm[self.CHIN].y) * h

            if face_h <= 0:
                return None

            brow_dist = abs(nose_y - brow_y) / face_h

            # Accumulate calibration baseline during first N frames
            if self._cal_frame_count <= self._cal_frames:
                self._cal_brow_dists.append(brow_dist)
                return 80.0  # neutral during calibration

            # Set baseline once we have enough samples
            if self._baseline_brow_dist is None and len(self._cal_brow_dists) > 0:
                self._baseline_brow_dist = sum(self._cal_brow_dists) / len(
                    self._cal_brow_dists
                )

            baseline = self._baseline_brow_dist or 0.30
            # Deviation from baseline: brows lower = more tension
            deviation = baseline - brow_dist  # positive means brows are lower (tense)
            # Map: 0 deviation → 80, >0.1 deviation → near 0
            score = 80.0 - max(0.0, deviation * 500.0)
            score = max(0.0, min(100.0, score))
            return score

        except (IndexError, AttributeError, TypeError) as exc:
            logger.debug("Brow tension computation failed: %s", exc)
            return None

    def _smile(
        self,
        lm: Any,
        w: int,
        h: int,
        blendshape_categories: Optional[Any],
    ) -> Optional[float]:
        """Score 0-100. Higher = more smiling.

        Section 3 — Smile:
        - Primary: blendshapes ``mouthSmileLeft`` / ``mouthSmileRight`` averaged.
        - Fallback: geometric mouth width ratio with baseline calibration.
        - Smoothed with EMA alpha ≈ 0.25 to avoid flicker on talking.
        """
        try:
            # ---- Primary: blendshapes ----
            if blendshape_categories is not None:
                msl = self._get_blendshape_score(
                    blendshape_categories, "mouthSmileLeft"
                )
                msr = self._get_blendshape_score(
                    blendshape_categories, "mouthSmileRight"
                )
                # Frown detection: penalise frowning below neutral baseline
                mfl = self._get_blendshape_score(
                    blendshape_categories, "mouthFrownLeft"
                )
                mfr = self._get_blendshape_score(
                    blendshape_categories, "mouthFrownRight"
                )
                smile_avg = (msl + msr) / 2.0
                frown_avg = (mfl + mfr) / 2.0

                # Neutral face = 60 (acceptable for confidence).
                # Smiling adds up to +40 (gentle smile ~80, broad ~100).
                # Frowning subtracts up to -50 (grimace → ~10-20).
                raw = 60.0 + (smile_avg * 50.0) - (frown_avg * 50.0)
                raw = max(0.0, min(100.0, raw))

                # EMA smooth (alpha ≈ 0.25)
                if self._smile_smoothed is None:
                    self._smile_smoothed = raw
                else:
                    a = self._smile_ema_alpha
                    self._smile_smoothed = a * raw + (1.0 - a) * self._smile_smoothed
                return self._smile_smoothed

            # ---- Fallback: geometric mouth width ratio ----

            mouth_l = self._lm_to_px(lm, self.MOUTH_LEFT, w, h)
            mouth_r = self._lm_to_px(lm, self.MOUTH_RIGHT, w, h)
            face_l = self._lm_to_px(lm, self.FACE_LEFT, w, h)
            face_r = self._lm_to_px(lm, self.FACE_RIGHT, w, h)

            mouth_w = float(np.linalg.norm(mouth_r - mouth_l))
            face_w = float(np.linalg.norm(face_r - face_l))

            if face_w <= 0:
                return None

            ratio = mouth_w / face_w

            # Accumulate during calibration phase
            if self._cal_frame_count <= self._cal_frames:
                self._cal_mouth_ratios.append(ratio)
                return 80.0

            if self._baseline_mouth_ratio is None and len(self._cal_mouth_ratios) > 0:
                self._baseline_mouth_ratio = sum(self._cal_mouth_ratios) / len(
                    self._cal_mouth_ratios
                )

            baseline = self._baseline_mouth_ratio or 0.40
            # Deviation above baseline → smile
            deviation = ratio - baseline
            raw = 80.0 + max(0.0, deviation * 200.0)
            raw = max(0.0, min(100.0, raw))

            # EMA smooth
            if self._smile_smoothed is None:
                self._smile_smoothed = raw
            else:
                a = self._smile_ema_alpha
                self._smile_smoothed = a * raw + (1.0 - a) * self._smile_smoothed

            return self._smile_smoothed

        except (IndexError, AttributeError, TypeError) as exc:
            logger.debug("Smile computation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(
        self,
        face_landmarks: Any,
        frame_shape: Tuple[int, ...],
        blendshape_categories: Optional[Any] = None,
        face_transform_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, Optional[float]]:
        """Extract all face-based confidence signals.

        Parameters
        ----------
        face_landmarks : mediapipe face_landmarks object or None
        frame_shape : (height, width, channels) tuple
        blendshape_categories : list of Category objects from FaceLandmarker, or None
        face_transform_matrix : 4x4 numpy array from FaceLandmarker, or None

        Returns
        -------
        dict with keys: eye_contact, blink_rate, brow_tension, smile
        Each value is 0-100 float or None if no face / computation failed.
        """
        if face_landmarks is None:
            return {
                "eye_contact": None,
                "blink_rate": None,
                "brow_tension": None,
                "smile": None,
            }

        lm = (
            face_landmarks.landmark
            if hasattr(face_landmarks, "landmark")
            else face_landmarks
        )
        h, w = frame_shape[0], frame_shape[1]

        # Increment calibration frame counter once per frame (not per sub-signal)
        if blendshape_categories is None:
            self._cal_frame_count += 1

        return {
            "eye_contact": self._eye_contact(lm, w, h, face_transform_matrix),
            "blink_rate": self._blink_rate(lm, w, h),
            "brow_tension": self._brow_tension(lm, w, h, blendshape_categories),
            "smile": self._smile(lm, w, h, blendshape_categories),
        }

    def reset_calibration(self) -> None:
        """Reset all calibration state for a new session."""
        self._cal_brow_dists.clear()
        self._cal_mouth_ratios.clear()
        self._cal_frame_count = 0
        self._baseline_brow_dist = None
        self._baseline_mouth_ratio = None
        self._gaze_smoothed = None
        self._smile_smoothed = None
        # Reset blink state machine
        self._blink_timestamps.clear()
        self._consec_below = 0
        self._last_blink_was_below = False
