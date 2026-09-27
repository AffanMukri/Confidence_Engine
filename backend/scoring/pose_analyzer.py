"""
Pose Analyzer — extracts body-pose confidence signals from MediaPipe Pose landmarks.

Implements Section 3 of the detection spec:
- Shoulder Alignment (0-100): Tilt angle of shoulder line vs horizontal.
  ``alignment = round(100 * (1 - clamp(|tiltAngle| / 15, 0, 1)))``
- Posture (0-100): Spine lean (shoulder-mid to hip-mid) vs vertical, with
  forward-head fallback when hips are not visible.
  ``posture = round(100 * (1 - clamp(spineAngleDeviation / 20, 0, 1)))``

Visibility gating (Section 1):
- ``shouldersVisible`` requires both left/right shoulder visibility > 0.6.
- When shoulders not visible, both metrics return None.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PoseAnalyzer:
    """Analyzes body pose landmarks for shoulder alignment and posture.

    Uses MediaPipe Pose 33-landmark model.
    Key landmarks:
        - 0: nose
        - 7: left ear, 8: right ear
        - 11: left shoulder, 12: right shoulder
        - 23: left hip, 24: right hip
    Each landmark has .x, .y, .z (normalized coords) and .visibility (0-1).
    """

    # Landmark indices (COCO/Pose topology)
    NOSE = 0
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def __init__(self) -> None:
        try:
            import config  # noqa: E402

            self._shoulder_vis_threshold = getattr(
                config, "SHOULDER_VISIBILITY_THRESHOLD", 0.6
            )
            self._spine_angle_limit = getattr(
                config, "POSTURE_SPINE_ANGLE_LIMIT", 20.0
            )
        except ImportError:
            self._shoulder_vis_threshold = 0.6
            self._spine_angle_limit = 20.0

    @staticmethod
    def _to_px(lm_entry: Any, w: int, h: int) -> np.ndarray:
        """Convert a single pose landmark to pixel coordinates."""
        return np.array([lm_entry.x * w, lm_entry.y * h])

    def _check_shoulder_visibility(self, landmark_list: Any) -> bool:
        """Check if both shoulders are visible.

        Section 1 — Shoulder visibility: requires both shoulders.visibility > 0.6.
        """
        try:
            ls = landmark_list[self.LEFT_SHOULDER]
            rs = landmark_list[self.RIGHT_SHOULDER]
            return (
                ls.visibility > self._shoulder_vis_threshold
                and rs.visibility > self._shoulder_vis_threshold
            )
        except (IndexError, AttributeError, TypeError):
            return False

    def analyze(
        self,
        pose_landmarks: Any,
        frame_shape: Tuple[int, ...],
        *,
        shoulders_visible: Optional[bool] = None,
    ) -> Dict[str, Optional[float]]:
        """Analyze pose landmarks and return confidence sub-scores.

        Parameters
        ----------
        pose_landmarks : LandmarkSet or None
        frame_shape : (height, width, channels)
        shoulders_visible : optional override (e.g. from debounced state).
            If omitted, computed fresh from landmark visibility.

        Returns
        -------
        dict with keys: shoulder_alignment, posture (0-100 float or None)
        """
        if not pose_landmarks:
            return {"shoulder_alignment": None, "posture": None}

        lm = (
            pose_landmarks.landmark
            if hasattr(pose_landmarks, "landmark")
            else pose_landmarks
        )
        h, w = frame_shape[:2]

        # ---- Visibility gate (Section 1) ----
        if shoulders_visible is None:
            shoulders_visible = self._check_shoulder_visibility(lm)

        if not shoulders_visible:
            return {"shoulder_alignment": None, "posture": None}

        try:
            # ---- Common landmark positions ----
            l_shoulder = self._to_px(lm[self.LEFT_SHOULDER], w, h)
            r_shoulder = self._to_px(lm[self.RIGHT_SHOULDER], w, h)
            l_ear = self._to_px(lm[self.LEFT_EAR], w, h)
            r_ear = self._to_px(lm[self.RIGHT_EAR], w, h)
            nose = self._to_px(lm[self.NOSE], w, h)
            l_hip = self._to_px(lm[self.LEFT_HIP], w, h)
            r_hip = self._to_px(lm[self.RIGHT_HIP], w, h)
            hip_mid = (l_hip + r_hip) / 2.0

            shoulder_mid = (l_shoulder + r_shoulder) / 2.0
            ear_mid = (l_ear + r_ear) / 2.0
            shoulder_width = float(np.linalg.norm(r_shoulder - l_shoulder))

            # ============================================================
            # Shoulder Alignment (0-100) — Section 3
            # ============================================================
            # tiltAngle = atan2(rightShoulder.y - leftShoulder.y,
            #                    rightShoulder.x - leftShoulder.x)
            dx = r_shoulder[0] - l_shoulder[0]
            dy = r_shoulder[1] - l_shoulder[1]
            tilt_angle = abs(np.degrees(np.arctan2(dy, dx)))
            # Normalize to 0-90°: we only care about tilt from horizontal,
            # not which direction the shoulder line points.  In MediaPipe's
            # coordinate system, left shoulder typically has higher x than
            # right shoulder, so atan2 returns ~180° for level shoulders.
            if tilt_angle > 90:
                tilt_angle = 180.0 - tilt_angle
            # alignment = round(100 * (1 - clamp(|tiltAngle| / 15, 0, 1)))
            shoulder_alignment = 100.0 * (1.0 - min(1.0, max(0.0, tilt_angle / 15.0)))
            shoulder_alignment = max(0.0, min(100.0, float(shoulder_alignment)))

            # ============================================================
            # Posture Quality (0-100) — Section 3
            # ============================================================
            hip_vis = (
                lm[self.LEFT_HIP].visibility > self._shoulder_vis_threshold
                and lm[self.RIGHT_HIP].visibility > self._shoulder_vis_threshold
            )

            if hip_vis:
                # ---- Primary: spine angle ----
                # Vector from hip-midpoint to shoulder-midpoint
                spine_vec = shoulder_mid - hip_mid
                # Vertical reference
                vertical = np.array([0.0, -1.0])
                if np.linalg.norm(spine_vec) > 0:
                    spine_vec_n = spine_vec / np.linalg.norm(spine_vec)
                    # Angle deviation from vertical (degrees)
                    cos_angle = np.dot(spine_vec_n, vertical)
                    cos_angle = max(-1.0, min(1.0, cos_angle))
                    spine_angle_dev = np.degrees(np.arccos(cos_angle))
                else:
                    spine_angle_dev = 0.0
                # posture = round(100 * (1 - clamp(spineAngleDeviation / 20, 0, 1)))
                posture = 100.0 * (
                    1.0 - min(1.0, max(0.0, spine_angle_dev / self._spine_angle_limit))
                )
            else:
                # ---- Fallback: forward-head posture proxy (Section 3) ----
                # When hips aren't visible, use the nose/ear horizontal offset from
                # shoulder midpoint as a slouch indicator.
                if shoulder_width > 0:
                    # Forward lean metric: horizontal offset of ears relative to shoulders
                    ear_offset = abs(ear_mid[0] - shoulder_mid[0]) / shoulder_width
                    nose_offset = abs(nose[0] - shoulder_mid[0]) / shoulder_width
                    head_offset = max(ear_offset, nose_offset)

                    # Vertical slouch: ear-to-shoulder distance ratio
                    vert_slouch = 0.0
                    dist_y = shoulder_mid[1] - ear_mid[1]
                    torso_h = hip_mid[1] - shoulder_mid[1]
                    if torso_h > 0:
                        vert_ratio = dist_y / torso_h
                        if vert_ratio < 0.25:
                            vert_slouch = (0.25 - vert_ratio) / 0.25  # 0-1
                    else:
                        vert_slouch = 0.0

                    # Combined: forward lean (horizontal) + slouch (vertical)
                    # This is a degraded fallback, so we're more conservative.
                    lean_penalty = max(0.0, min(1.0, head_offset * 3.0))
                    slouch_penalty = min(1.0, vert_slouch * 1.5)
                    combined_penalty = max(lean_penalty, slouch_penalty)
                    posture = 100.0 * (1.0 - combined_penalty)
                else:
                    posture = 50.0  # Can't determine, neutral

            posture = max(0.0, min(100.0, float(posture)))

        except (IndexError, AttributeError, TypeError) as exc:
            logger.debug("Pose analysis failed: %s", exc)
            return {"shoulder_alignment": None, "posture": None}

        return {
            "shoulder_alignment": float(shoulder_alignment),
            "posture": float(posture),
        }
