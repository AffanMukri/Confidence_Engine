"""
Frame Processor — orchestrates MediaPipe Tasks models and per-frame analysis.

Implements the detection spec:
- Section 1: Debounced visibility gating for hands/shoulders.
- Section 3: Passes blendshapes and face transform matrix to FaceAnalyzer.
- Section 4: Dynamic status indicator (active, hands/shoulders not detected, etc.).
- Section 5: Structured output contract with visibilityState and weightsUsed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .face_analyzer import FaceAnalyzer
from .hand_analyzer import HandAnalyzer
from .pose_analyzer import PoseAnalyzer
from .composite import CompositeScorer
from .mediapipe_models import ensure_model

logger = logging.getLogger(__name__)


@dataclass
class LandmarkSet:
    """Adapter so analyzers can keep using ``.landmark`` lists."""

    landmark: List[Any]


class _TransformedLandmark:
    """Re-project landmarks detected on a cropped sub-image back to the
    original frame's normalized coordinate space.

    MediaPipe Tasks returns normalized coordinates relative to the input
    image. When we run detection on a cropped ROI, the resulting x/y are
    relative to the crop, not the full frame. This wrapper maps them back
    so downstream analyzers (which multiply by frame_shape) compute correct
    pixel positions.
    """

    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(self, lm: Any, x1: int, y1: int, crop_w: int, crop_h: int, frame_w: int, frame_h: int) -> None:
        self.x = min(1.0, max(0.0, (lm.x * crop_w + x1) / frame_w))
        self.y = min(1.0, max(0.0, (lm.y * crop_h + y1) / frame_h))
        self.z = getattr(lm, "z", 0.0)
        self.visibility = getattr(lm, "visibility", 1.0)
        self.presence = getattr(lm, "presence", 1.0)


# ------------------------------------------------------------------
# Visibility state machine — Section 1
# ------------------------------------------------------------------
class VisibilityGate:
    """Debounced visibility detection for hands and shoulders.

    Section 1:
    - Require ≥ 10 consecutive frames visible before marking "visible".
    - Require ≥ 15 consecutive frames absent before marking "not visible".
    - Prevents flicker between ``--`` and a number.
    """

    def __init__(self) -> None:
        try:
            import config  # noqa: E402

            self._shoulder_threshold = getattr(
                config, "SHOULDER_VISIBILITY_THRESHOLD", 0.6
            )
            self._wrist_threshold = getattr(
                config, "WRIST_VISIBILITY_THRESHOLD", 0.6
            )
            self._on_frames = getattr(config, "VISIBILITY_ON_FRAMES", 10)
            self._off_frames = getattr(config, "VISIBILITY_OFF_FRAMES", 15)
        except ImportError:
            self._shoulder_threshold = 0.6
            self._wrist_threshold = 0.6
            self._on_frames = 10
            self._off_frames = 15

        self.shoulders_debounced: bool = False
        self.hands_debounced: bool = False
        self._shoulder_consec_visible: int = 0
        self._shoulder_consec_invisible: int = 0
        self._hand_consec_visible: int = 0
        self._hand_consec_invisible: int = 0

    def update(
        self,
        shoulder_vis: bool,
        hand_landmarks_present: bool,
        wrist_vis_left: bool,
        wrist_vis_right: bool,
    ) -> None:
        """Update visibility state for both shoulders and hands.

        shoulders_visible = both shoulders.visibility > 0.6
        hands_visible = (leftWrist.visibility > 0.6 || rightWrist.visibility > 0.6)
                        AND at least one full hand landmark set detected
        """
        # ---- Shoulders ----
        if shoulder_vis:
            self._shoulder_consec_visible += 1
            self._shoulder_consec_invisible = 0
        else:
            self._shoulder_consec_invisible += 1
            self._shoulder_consec_visible = 0

        if self._shoulder_consec_visible >= self._on_frames:
            self.shoulders_debounced = True
        if self._shoulder_consec_invisible >= self._off_frames:
            self.shoulders_debounced = False

        # ---- Hands ----
        hands_raw = (wrist_vis_left or wrist_vis_right) and hand_landmarks_present
        if hands_raw:
            self._hand_consec_visible += 1
            self._hand_consec_invisible = 0
        else:
            self._hand_consec_invisible += 1
            self._hand_consec_visible = 0

        if self._hand_consec_visible >= self._on_frames:
            self.hands_debounced = True
        if self._hand_consec_invisible >= self._off_frames:
            self.hands_debounced = False

    def get_state(self) -> Dict[str, bool]:
        """Return current debounced visibility state."""
        return {
            "hands_visible": self.hands_debounced,
            "shoulders_visible": self.shoulders_debounced,
        }

    def reset(self) -> None:
        """Reset all counters for a new session."""
        self.shoulders_debounced = False
        self.hands_debounced = False
        self._shoulder_consec_visible = 0
        self._shoulder_consec_invisible = 0
        self._hand_consec_visible = 0
        self._hand_consec_invisible = 0


class FrameProcessor:
    """Orchestrates MediaPipe Tasks models and analyzers for each video frame."""

    def __init__(self) -> None:
        self.face_analyzer = FaceAnalyzer()
        self.hand_analyzer = HandAnalyzer()
        self.pose_analyzer = PoseAnalyzer()
        self.composite = CompositeScorer()
        self.visibility = VisibilityGate()

        self._pose_disabled = False  # developer/perf toggle (not default)
        self._mp_initialized = False
        self._mp_init_failed = False
        self._face_landmarker = None
        self._hand_landmarker = None
        self._pose_landmarker = None
        self._mp_image_cls = None
        self._image_format = None

        # Diagnostics: how often the pose-guided ROI fallback is used
        self._face_fallback_hits: int = 0
        self._hand_fallback_hits: int = 0
        self._frame_count: int = 0

    def _init_mediapipe(self) -> Optional[str]:
        """Lazy-init MediaPipe Tasks models on first frame.

        Returns ``None`` on success, or an error string on failure.
        Once failed, subsequent calls short-circuit immediately.
        """
        if self._mp_initialized:
            return None
        if self._mp_init_failed:
            return "MediaPipe initialization previously failed"

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        # Use lower thresholds for face/hand to improve detection in
        # typical webcam conditions (lower lighting, smaller face in frame).
        # Pose keeps a slightly higher threshold since it works on full body.
        try:
            import config

            face_det = getattr(config, "MP_FACE_DETECTION_CONFIDENCE", 0.15)
            hand_det = getattr(config, "MP_HAND_DETECTION_CONFIDENCE", 0.15)
            face_trk = getattr(config, "MP_FACE_TRACKING_CONFIDENCE", 0.15)
            hand_trk = getattr(config, "MP_HAND_TRACKING_CONFIDENCE", 0.15)
            pose_det = getattr(config, "MP_MIN_DETECTION_CONFIDENCE", 0.5)
            pose_trk = getattr(config, "MP_MIN_TRACKING_CONFIDENCE", 0.5)
            delegate_name = getattr(config, "MP_DELEGATE", "CPU").upper()
        except ImportError:
            face_det = hand_det = face_trk = hand_trk = 0.15
            pose_det = pose_trk = 0.5
            delegate_name = "CPU"

        try:
            face_model = str(ensure_model("face_landmarker.task"))
            hand_model = str(ensure_model("hand_landmarker.task"))
            pose_model = str(ensure_model("pose_landmarker_lite.task"))
        except Exception as exc:
            self._mp_init_failed = True
            logger.exception("Failed to download MediaPipe models")
            return f"Model download failed: {exc}"

        try:
            base_options = python.BaseOptions
            running_mode = vision.RunningMode.IMAGE

            # CPU delegate is most reliable across Windows/Mac/Linux
            delegate = base_options.Delegate.CPU

            self._face_landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=base_options(
                        model_asset_path=face_model, delegate=delegate
                    ),
                    running_mode=running_mode,
                    num_faces=1,
                    min_face_detection_confidence=face_det,
                    min_face_presence_confidence=face_det,
                    min_tracking_confidence=face_trk,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                )
            )
            self._hand_landmarker = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=base_options(
                        model_asset_path=hand_model, delegate=delegate
                    ),
                    running_mode=running_mode,
                    num_hands=2,
                    min_hand_detection_confidence=hand_det,
                    min_hand_presence_confidence=hand_det,
                    min_tracking_confidence=hand_trk,
                )
            )
            self._pose_landmarker = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=base_options(
                        model_asset_path=pose_model, delegate=delegate
                    ),
                    running_mode=running_mode,
                    min_pose_detection_confidence=pose_det,
                    min_pose_presence_confidence=pose_det,
                    min_tracking_confidence=pose_trk,
                )
            )

            self._mp_image_cls = mp.Image
            self._image_format = mp.ImageFormat.SRGB
            self._mp_initialized = True
            logger.info(
                "MediaPipe Tasks initialized — face_det=%.2f hand_det=%.2f pose_det=%.2f (blendshapes+transform enabled)",
                face_det, hand_det, pose_det,
            )
            return None
        except Exception as exc:
            self._mp_init_failed = True
            logger.exception("MediaPipe model creation failed")
            return f"Model init failed: {exc}"

    @staticmethod
    def _to_mp_image(frame_rgb: np.ndarray, mp_image_cls: Any, image_format: Any) -> Any:
        """Build a MediaPipe Image from an RGB numpy array."""
        if not frame_rgb.flags["C_CONTIGUOUS"]:
            frame_rgb = np.ascontiguousarray(frame_rgb)
        return mp_image_cls(image_format=image_format, data=frame_rgb)

    @staticmethod
    def _pick_largest_face(
        face_sets: List[LandmarkSet], w: int, h: int
    ) -> LandmarkSet:
        """Return the face with the largest bounding box."""
        if len(face_sets) == 1:
            return face_sets[0]

        best, best_area = face_sets[0], 0.0
        for face in face_sets:
            xs = [lm.x * w for lm in face.landmark]
            ys = [lm.y * h for lm in face.landmark]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            if area > best_area:
                best, best_area = face, area
        return best

    @staticmethod
    def _crop_region(
        frame_rgb: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> Optional[Tuple[np.ndarray, int, int, int, int]]:
        """Crop a region from the RGB frame, clamped to frame bounds.

        Returns the cropped array and the *actual* pixel coordinates used
        (after clamping) so that landmarks detected on the crop can be
        mapped back to the original frame.
        """
        h, w = frame_rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 64 or y2 - y1 < 64:
            return None
        return frame_rgb[y1:y2, x1:x2], x1, y1, x2, y2

    @staticmethod
    def _head_bbox_from_pose(
        pose_lm: List[Any], frame_w: int, frame_h: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """Estimate a head bounding box from pose landmarks.

        Uses nose (0), left ear (7), right ear (8), left shoulder (11),
        right shoulder (12).  Returns (x1, y1, x2, y2) in pixel coords.
        """
        try:
            nose = pose_lm[0]
            l_ear = pose_lm[7]
            r_ear = pose_lm[8]
            l_shoulder = pose_lm[11]
            r_shoulder = pose_lm[12]

            # Center x from nose/ears, top above ears, bottom at shoulder line
            center_x = (nose.x + l_ear.x + r_ear.x) / 3.0
            top_y = min(nose.y, l_ear.y, r_ear.y)
            bottom_y = (l_shoulder.y + r_shoulder.y) / 2.0

            # Width: 3x the ear-to-ear distance, minimum 35% of frame width
            ear_span = abs(r_ear.x - l_ear.x)
            width = max(ear_span * 3.0, 0.35)

            # Height: from top to shoulder, expanded upward for forehead
            height = (bottom_y - top_y) * 1.4

            x1 = int((center_x - width / 2.0) * frame_w)
            x2 = int((center_x + width / 2.0) * frame_w)
            y1 = int((top_y - height * 0.35) * frame_h)
            y2 = int(bottom_y * frame_h)

            return (x1, y1, x2, y2)
        except (IndexError, AttributeError, TypeError):
            return None

    @staticmethod
    def _hand_bboxes_from_pose(
        pose_lm: List[Any], frame_w: int, frame_h: int
    ) -> List[Tuple[int, int, int, int]]:
        """Estimate hand bounding boxes from pose wrist landmarks.

        Wrist indices: left=15, right=16.  Returns list of (x1,y1,x2,y2).
        """
        # Map wrist index to its corresponding elbow index so we can shift
        # the crop from the wrist toward the hand itself.
        wrist_to_elbow = {15: 13, 16: 14}
        bboxes = []
        for wrist_idx in [15, 16]:
            try:
                wrist = pose_lm[wrist_idx]
                if wrist.visibility < 0.3:
                    continue

                # Larger square crop (~25% of frame max dimension)
                size = int(0.25 * max(frame_w, frame_h))

                # Shift center from wrist toward hand using elbow→wrist direction
                cx = int(wrist.x * frame_w)
                cy = int(wrist.y * frame_h)
                try:
                    elbow = pose_lm[wrist_to_elbow[wrist_idx]]
                    dx = wrist.x - elbow.x
                    dy = wrist.y - elbow.y
                    norm = (dx * dx + dy * dy) ** 0.5
                    if norm > 0.001:
                        cx = int((wrist.x + dx / norm * 0.35) * frame_w)
                        cy = int((wrist.y + dy / norm * 0.35) * frame_h)
                except (IndexError, AttributeError, TypeError):
                    pass

                x1 = cx - size // 2
                y1 = cy - size // 2
                x2 = cx + size // 2
                y2 = cy + size // 2
                bboxes.append((x1, y1, x2, y2))
            except (IndexError, AttributeError, TypeError):
                continue
        return bboxes

    @staticmethod
    def _reproject_landmarks(
        landmarks_list: List[Any],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        frame_w: int,
        frame_h: int,
    ) -> List[Any]:
        """Map landmarks detected on a cropped ROI to original-frame coords."""
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w <= 0 or crop_h <= 0:
            return landmarks_list
        return [
            _TransformedLandmark(lm, x1, y1, crop_w, crop_h, frame_w, frame_h)
            for lm in landmarks_list
        ]

    def _detect_face(
        self,
        mp_image: Any,
        frame_rgb: np.ndarray,
        pose_lm: Optional[List[Any]],
    ) -> Tuple[Optional[List[Any]], Optional[Any], Optional[Any]]:
        """Run face detection; fall back to a pose-guided head crop if needed.

        Returns (face_landmarks_list_or_none, blendshape_categories, transform_matrix)
        """
        face_result = self._face_landmarker.detect(mp_image)

        if face_result.face_landmarks:
            blendshapes = None
            if face_result.face_blendshapes and len(face_result.face_blendshapes) > 0:
                # In this MediaPipe version face_blendshapes is a list of lists.
                inner = face_result.face_blendshapes[0]
                blendshapes = inner if isinstance(inner, list) else inner.categories
            transform = None
            if (
                hasattr(face_result, "facial_transformation_matrixes")
                and face_result.facial_transformation_matrixes
                and len(face_result.facial_transformation_matrixes) > 0
            ):
                transform = face_result.facial_transformation_matrixes[0]
            return (face_result.face_landmarks, blendshapes, transform)

        # Fallback: crop head region from pose and retry
        if pose_lm is None:
            return (None, None, None)

        h, w = frame_rgb.shape[:2]
        bbox = self._head_bbox_from_pose(pose_lm, w, h)
        if bbox is None:
            return (None, None, None)

        crop_info = self._crop_region(frame_rgb, *bbox)
        if crop_info is None:
            return (None, None, None)
        cropped, x1, y1, x2, y2 = crop_info

        crop_image = self._to_mp_image(cropped, self._mp_image_cls, self._image_format)
        crop_result = self._face_landmarker.detect(crop_image)

        if crop_result.face_landmarks:
            self._face_fallback_hits += 1
            logger.debug("Face detected via pose-guided crop")
            h, w = frame_rgb.shape[:2]
            reprojected = [
                self._reproject_landmarks(face, x1, y1, x2, y2, w, h)
                for face in crop_result.face_landmarks
            ]
            blendshapes = None
            if crop_result.face_blendshapes and len(crop_result.face_blendshapes) > 0:
                inner = crop_result.face_blendshapes[0]
                blendshapes = inner if isinstance(inner, list) else inner.categories
            transform = None
            if (
                hasattr(crop_result, "facial_transformation_matrixes")
                and crop_result.facial_transformation_matrixes
                and len(crop_result.facial_transformation_matrixes) > 0
            ):
                transform = crop_result.facial_transformation_matrixes[0]
            return (reprojected, blendshapes, transform)

        return (None, None, None)

    def _detect_hands(
        self,
        mp_image: Any,
        frame_rgb: np.ndarray,
        pose_lm: Optional[List[Any]],
    ) -> Tuple[bool, Optional[List[Any]]]:
        """Run hand detection; fall back to pose-guided wrist crops if needed.

        Returns (landmarks_present, hand_landmarks_list_or_none)
        """
        hand_result = self._hand_landmarker.detect(mp_image)
        if hand_result.hand_landmarks:
            return (True, hand_result.hand_landmarks)

        if pose_lm is None:
            return (False, None)

        h, w = frame_rgb.shape[:2]
        bboxes = self._hand_bboxes_from_pose(pose_lm, w, h)
        if not bboxes:
            return (False, None)

        h, w = frame_rgb.shape[:2]
        all_hands = []
        for bbox in bboxes:
            crop_info = self._crop_region(frame_rgb, *bbox)
            if crop_info is None:
                continue
            cropped, x1, y1, x2, y2 = crop_info
            crop_image = self._to_mp_image(
                cropped, self._mp_image_cls, self._image_format
            )
            crop_result = self._hand_landmarker.detect(crop_image)
            if crop_result.hand_landmarks:
                self._hand_fallback_hits += 1
                logger.debug("Hand detected via pose-guided crop")
                for hand in crop_result.hand_landmarks:
                    all_hands.append(
                        self._reproject_landmarks(hand, x1, y1, x2, y2, w, h)
                    )

        if all_hands:
            return (True, all_hands)
        return (False, None)

    def process_frame(self, jpeg_bytes: bytes) -> Dict[str, Any]:
        """Decode a JPEG frame and run all analyses.

        Section 5 — Output data contract:

        Returns
        -------
        dict with:
            status: str ("ok", "no_face", "model_init_failed", "error")
            composite_score: float or None
            signals: dict of signal_name → float or None
            visibility: dict of {hands_visible, shoulders_visible}
            weights_used: dict of signal_name → weight (post-redistribution)
            error: str (optional, on failure)
        """
        import cv2  # noqa: E402

        # Lazy-init MediaPipe
        init_error = self._init_mediapipe()
        if init_error is not None:
            return {
                "status": "model_init_failed",
                "composite_score": None,
                "signals": {},
                "visibility": {"hands_visible": False, "shoulders_visible": False},
                "weights_used": {},
                "error": init_error,
            }

        try:
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return {
                    "status": "error",
                    "composite_score": None,
                    "signals": {},
                    "visibility": {"hands_visible": False, "shoulders_visible": False},
                    "weights_used": {},
                    "error": "Invalid JPEG data",
                }

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            shape = frame.shape
            mp_image = self._to_mp_image(
                frame_rgb, self._mp_image_cls, self._image_format
            )

            signals: Dict[str, Optional[float]] = {}
            face_detected = False

            # ================================================================
            # --- Pose detection FIRST (used to guide face/hand ROI crops) ---
            # ================================================================
            shoulder_vis_raw = False
            wrist_vis_left = False
            wrist_vis_right = False
            pose_landmarks_list = None
            pose_lm_for_roi: Optional[List[Any]] = None

            if not self._pose_disabled:
                try:
                    pose_result = self._pose_landmarker.detect(mp_image)

                    if pose_result.pose_landmarks:
                        pose_lm = list(pose_result.pose_landmarks[0])
                        pose_landmarks_list = LandmarkSet(pose_lm)
                        pose_lm_for_roi = pose_lm

                        # Check raw visibility for gating (Section 1)
                        if len(pose_lm) > 12:
                            shoulder_vis_raw = (
                                pose_lm[11].visibility > 0.6
                                and pose_lm[12].visibility > 0.6
                            )
                        if len(pose_lm) > 16:
                            wrist_vis_left = pose_lm[15].visibility > 0.6
                            wrist_vis_right = pose_lm[16].visibility > 0.6

                except Exception as exc:
                    logger.warning("Pose detection error: %s", exc)

            # ================================================================
            # --- Face (landmarks + blendshapes + transform matrix) ---
            # ================================================================
            n_face_detected = 0
            try:
                face_landmarks, blendshapes, face_transform = self._detect_face(
                    mp_image, frame_rgb, pose_lm_for_roi
                )

                if face_landmarks:
                    n_face_detected = len(face_landmarks)
                    face_sets = [
                        LandmarkSet(list(landmarks))
                        for landmarks in face_landmarks
                    ]
                    best_face = self._pick_largest_face(
                        face_sets, shape[1], shape[0]
                    )
                    signals.update(
                        self.face_analyzer.analyze(
                            best_face,
                            shape,
                            blendshape_categories=blendshapes,
                            face_transform_matrix=face_transform,
                        )
                    )
                    face_detected = True
                else:
                    signals.update(self.face_analyzer.analyze(None, shape))
            except Exception as exc:
                logger.warning("Face analyzer error: %s", exc)
                signals.update(self.face_analyzer.analyze(None, shape))

            # Log detection status periodically (every 50 frames)
            self._frame_count += 1
            if self._frame_count % 50 == 1:
                logger.info(
                    "Detection status: faces=%d pose=%s face_fb=%d hand_fb=%d "
                    "face_signals=%s hand_signals=%s",
                    n_face_detected,
                    "ok" if pose_lm_for_roi else "none",
                    self._face_fallback_hits,
                    self._hand_fallback_hits,
                    {k: ("OK" if v is not None else "None") for k, v in signals.items() if k in ('eye_contact', 'blink_rate', 'brow_tension', 'smile')},
                    {k: ("OK" if v is not None else "None") for k, v in signals.items() if k in ('hand_stability', 'fidgeting', 'hand_openness')},
                )

            # ================================================================
            # --- Hands (landmarks via HandLandmarker + pose ROI fallback) ---
            # ================================================================
            hand_sets = None
            hand_landmarks_exist = False
            try:
                hand_landmarks_exist, hand_landmarks = self._detect_hands(
                    mp_image, frame_rgb, pose_lm_for_roi
                )
                if hand_landmarks_exist and hand_landmarks:
                    hand_sets = [
                        LandmarkSet(list(landmarks))
                        for landmarks in hand_landmarks
                    ]
            except Exception as exc:
                logger.warning("Hand detection error: %s", exc)

            # ================================================================
            # --- Debounce visibility (single update per frame) ---
            # ================================================================
            self.visibility.update(
                shoulder_vis_raw,
                hand_landmarks_exist,
                wrist_vis_left,
                wrist_vis_right,
            )
            vis_state = self.visibility.get_state()

            # ================================================================
            # --- Pose analysis (uses debounced shoulder visibility) ---
            # ================================================================
            if not self._pose_disabled:
                try:
                    signals.update(
                        self.pose_analyzer.analyze(
                            pose_landmarks_list,
                            shape,
                            shoulders_visible=vis_state["shoulders_visible"],
                        )
                    )
                except Exception as exc:
                    logger.warning("Pose analyzer error: %s", exc)
                    signals.update(self.pose_analyzer.analyze(None, shape))
            else:
                # Developer-disabled — Section 4 status indicator logic
                signals.update(self.pose_analyzer.analyze(None, shape))

            # ================================================================
            # --- Hand analysis (uses debounced hand visibility) ---
            # ================================================================
            try:
                signals.update(
                    self.hand_analyzer.analyze(
                        hand_sets,
                        shape,
                        hands_visible=vis_state["hands_visible"],
                    )
                )
            except Exception as exc:
                logger.warning("Hand analyzer error: %s", exc)
                signals.update(self.hand_analyzer.analyze(None, shape))

            # ================================================================
            # --- Composite score with dynamic reweighting (Section 2) ---
            # ================================================================
            score_data = self.composite.calculate(signals)
            score_data["signals"] = signals
            score_data["visibility"] = self.visibility.get_state()

            # ---- Status indicator (Section 4) ----
            vis = self.visibility.get_state()
            if score_data.get("status") == "no_face":
                score_data["status"] = "no_face"
            elif self._pose_disabled:
                score_data["status"] = "pose_skipped"
            elif not vis["shoulders_visible"] and not vis["hands_visible"]:
                score_data["status"] = "no_pose"
            elif not vis["shoulders_visible"]:
                score_data["status"] = "shoulders_not_detected"
            elif not vis["hands_visible"] and face_detected:
                score_data["status"] = "hands_not_detected"
            else:
                score_data["status"] = "ok"

            return score_data

        except Exception as exc:
            logger.exception("Frame processing error")
            return {
                "status": "error",
                "composite_score": None,
                "signals": {},
                "visibility": {"hands_visible": False, "shoulders_visible": False},
                "weights_used": {},
                "error": str(exc),
            }

    def close(self) -> None:
        """Release MediaPipe model resources."""
        for landmarker in (
            self._face_landmarker,
            self._hand_landmarker,
            self._pose_landmarker,
        ):
            if landmarker is not None:
                landmarker.close()
