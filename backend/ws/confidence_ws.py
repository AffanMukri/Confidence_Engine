"""
WebSocket endpoint for real-time confidence scoring.

Protocol
--------
CLIENT → SERVER : raw binary JPEG frame (every ~100 ms at 10 FPS)
SERVER → CLIENT : JSON ``score_update`` every 2 seconds
SERVER → CLIENT : JSON ``session_summary`` on disconnect / explicit end

Updated for Section 4 status indicators and Section 5 output contract.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from scoring import ConfidenceEngine
import db
from models.session import ConfidenceSession
import config

logger = logging.getLogger(__name__)
router = APIRouter()


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types, NaN/Inf, or collections to standard JSON types."""
    if obj is None:
        return None
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_to_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, deque)):
        return [_to_json_safe(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Connection manager (maps session_id → state)
# ---------------------------------------------------------------------------
class SessionState:
    """Mutable per-session state held in memory during an active connection."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.engine = ConfidenceEngine()
        self.started_at: float = time.time()
        self.frame_count: int = 0
        self.score_history: List[Dict[str, Any]] = []
        self.all_scores: List[float] = []
        self.signal_accumulators: Dict[str, List[float]] = {}
        self.pending_frames: List[bytes] = []
        self.last_update_time: float = time.time()

    # ---- helpers ----
    def record_score(self, result: dict) -> None:
        """Accumulate a per-frame result for the 2-second window."""
        self.frame_count += 1
        score = result.get("composite_score")
        if score is not None:
            self.all_scores.append(float(score))
            # accumulate per-signal values
            signals = result.get("signals", {})
            for key, val in signals.items():
                if val is not None:
                    self.signal_accumulators.setdefault(key, []).append(float(val))

    def build_window_update(self) -> Dict[str, Any]:
        """Build the 2-second ``score_update`` message.

        Section 4 — Status Indicator:
        - ``"Pose analysis active"`` (green) when pose model running + shoulders visible.
        - ``"Hands not detected"`` / ``"Shoulders not detected"`` (yellow).
        - ``"Pose analysis skipped (performance)"`` (gray) only when
          developer/perf config disables the pose model.
        """
        latest = self.engine.latest_result or {}
        composite = latest.get("composite_score")
        signals = latest.get("signals", {})
        visibility = latest.get("visibility", {})
        weights_used = latest.get("weights_used", {})
        warnings: List[str] = []

        status = latest.get("status", "ok")

        # Section 4: dynamic status indicator
        if status == "no_face":
            warnings.append("No face detected")
        elif status == "pose_skipped":
            warnings.append("Pose analysis skipped (performance)")
        elif status == "shoulders_not_detected":
            warnings.append("Shoulders not detected")
        elif status == "hands_not_detected":
            warnings.append("Hands not detected")
        elif status == "no_pose":
            warnings.append("Pose analysis active — no pose landmarks yet")
        elif status == "model_init_failed":
            err = latest.get("error", "Unknown error")
            warnings.append(f"Analysis engine failed: {err}")
        elif status == "error":
            err = latest.get("error", "Unknown error")
            warnings.append(f"Frame processing error: {err}")

        msg = {
            "type": "score_update",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "composite_score": float(composite) if composite is not None else None,
            "signals": _to_json_safe(signals),
            "visibility": _to_json_safe(visibility),
            "weights_used": _to_json_safe(weights_used),
            "warnings": warnings,
            "frames_analyzed": self.frame_count,
        }

        # Record to timeline
        self.score_history.append(
            {
                "timestamp": msg["timestamp"],
                "score": msg["composite_score"],
            }
        )
        self.last_update_time = time.time()
        return msg

    def build_summary(self) -> Dict[str, Any]:
        """Build the ``session_summary`` message."""
        # Sessions shorter than SCORE_UPDATE_INTERVAL still need a useful
        # timeline point in their final report.
        latest_score = (self.engine.latest_result or {}).get("composite_score")
        if latest_score is not None and (
            not self.score_history
            or self.score_history[-1].get("score") != latest_score
        ):
            self.score_history.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "score": float(latest_score),
                }
            )

        duration = time.time() - self.started_at
        avg_score = (
            float(sum(self.all_scores) / len(self.all_scores))
            if self.all_scores
            else 0.0
        )
        min_score = float(min(self.all_scores)) if self.all_scores else 0.0
        max_score = float(max(self.all_scores)) if self.all_scores else 0.0

        # Compute low-confidence windows (contiguous score_updates < 50)
        low_conf = _find_low_confidence_windows(
            self.score_history, config.LOW_CONFIDENCE_THRESHOLD
        )

        # Per-signal summary with avg, min, max
        signals_summary = {}
        for key, vals in self.signal_accumulators.items():
            if vals:
                signals_summary[key] = {
                    "avg": round(float(sum(vals) / len(vals)), 1),
                    "min": round(float(min(vals)), 1),
                    "max": round(float(max(vals)), 1),
                }
            else:
                signals_summary[key] = {"avg": None, "min": None, "max": None}

        return {
            "type": "session_summary",
            "session_id": self.session_id,
            "average_score": round(avg_score, 1),
            "min_score": round(min_score, 1),
            "max_score": round(max_score, 1),
            "low_confidence_timestamps": _to_json_safe(low_conf),
            "total_frames": self.frame_count,
            "duration_seconds": round(float(duration), 1),
            "signals_summary": _to_json_safe(signals_summary),
            "score_timeline": _to_json_safe(self.score_history),
        }


def _find_low_confidence_windows(
    timeline: List[Dict[str, Any]], threshold: float
) -> List[Dict[str, Any]]:
    """Identify contiguous windows where composite_score < *threshold*."""
    windows: List[Dict[str, Any]] = []
    current_window: Optional[Dict[str, Any]] = None

    for entry in timeline:
        score = entry.get("score")
        ts = entry.get("timestamp", "")

        if score is not None and score < threshold:
            if current_window is None:
                current_window = {
                    "start": ts,
                    "end": ts,
                    "scores": [score],
                }
            else:
                current_window["end"] = ts
                current_window["scores"].append(score)
        else:
            if current_window is not None:
                scores = current_window.pop("scores")
                current_window["avg_score"] = round(
                    float(sum(scores) / len(scores)), 1
                )
                windows.append(current_window)
                current_window = None

    # close any trailing window
    if current_window is not None:
        scores = current_window.pop("scores")
        current_window["avg_score"] = round(float(sum(scores) / len(scores)), 1)
        windows.append(current_window)

    return windows


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------
async def _persist_summary(summary: Dict[str, Any]) -> None:
    """Write session summary to the ``confidence_sessions`` table."""
    try:
        safe_summary = _to_json_safe(summary)
        async with db.AsyncSessionLocal() as session:
            result = await session.execute(
                select(ConfidenceSession).where(
                    ConfidenceSession.session_id == safe_summary["session_id"]
                )
            )
            row = result.scalar_one_or_none()
            values = {
                "average_score": safe_summary["average_score"],
                "min_score": safe_summary["min_score"],
                "max_score": safe_summary["max_score"],
                "score_timeline": safe_summary.get("score_timeline"),
                "low_confidence": safe_summary.get("low_confidence_timestamps"),
                "total_frames": safe_summary["total_frames"],
                "duration_seconds": safe_summary["duration_seconds"],
                "signals_summary": safe_summary.get("signals_summary"),
            }
            if row is None:
                row = ConfidenceSession(
                    session_id=safe_summary["session_id"],
                    **values,
                )
                session.add(row)
            else:
                for key, value in values.items():
                    setattr(row, key, value)
            await session.commit()
            logger.info("Persisted session %s", safe_summary["session_id"])
    except Exception:
        logger.exception(
            "Failed to persist session %s", summary.get("session_id")
        )


# ---------------------------------------------------------------------------
# Frame-receive helper (real-time backpressure)
# ---------------------------------------------------------------------------
async def _receive_latest_frame(
    websocket: WebSocket,
) -> dict:
    """Receive the next message, keeping only the latest binary frame.

    If processing is slower than the client's frame rate, multiple
    binary frames may queue up in the transport buffer.  This helper
    drains them and keeps only the most recent frame, so the engine
    always works on the freshest input.

    Text messages (end_session, JSON) are returned immediately without
    draining.
    """
    msg = await websocket.receive()

    if msg.get("type") == "websocket.disconnect":
        raise WebSocketDisconnect(msg.get("code", 1000))

    # Text messages are control commands — return right away
    if "text" in msg:
        return msg

    # Binary frame: drain any additional buffered frames
    if "bytes" in msg:
        drained = 0
        while True:
            try:
                next_msg = await asyncio.wait_for(
                    websocket.receive(), timeout=0.002
                )
                if next_msg.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect(next_msg.get("code", 1000))
                if "text" in next_msg:
                    return next_msg
                if "bytes" in next_msg:
                    msg = next_msg
                    drained += 1
            except (asyncio.TimeoutError, TimeoutError):
                break
            except WebSocketDisconnect:
                raise
            except Exception:
                # Connection might have closed during drain
                break

        if drained:
            logger.debug("Drained %d stale frame(s)", drained)

    return msg


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
@router.websocket("/ws/confidence/{session_id}")
async def confidence_websocket(websocket: WebSocket, session_id: str) -> None:
    """Stream confidence scores for an interview session."""
    await websocket.accept()
    if not session_id or len(session_id) > config.MAX_SESSION_ID_LENGTH:
        await websocket.close(code=1008, reason="Invalid session ID")
        return

    state = SessionState(session_id)
    logger.info("Session %s connected", session_id)

    # Flag to track if models have been loaded yet
    _models_initialized = False

    try:
        while True:
            # Receive frame (with automatic drain of stale buffered frames)
            try:
                message = await _receive_latest_frame(websocket)
            except WebSocketDisconnect:
                logger.info("Session %s disconnected during receive", session_id)
                break
            except Exception as exc:
                logger.warning("Receive error in session %s: %s", session_id, exc)
                break

            # Check for explicit end-session command
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "")
                    if msg_type == "end_session":
                        logger.info("Session %s ended by client", session_id)
                        break
                except (json.JSONDecodeError, TypeError):
                    pass
                continue

            if "bytes" not in message:
                continue

            frame_bytes: bytes = message["bytes"]
            if len(frame_bytes) > config.MAX_FRAME_BYTES:
                await websocket.send_json(
                    {
                        "type": "score_update",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "composite_score": None,
                        "signals": {},
                        "visibility": {
                            "hands_visible": False,
                            "shoulders_visible": False,
                        },
                        "weights_used": {},
                        "warnings": ["Frame is too large"],
                        "frames_analyzed": state.frame_count,
                    }
                )
                continue

            # Notify client that models are loading (first frame only)
            if not _models_initialized:
                try:
                    await websocket.send_json({
                        "type": "score_update",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "composite_score": None,
                        "signals": {},
                        "visibility": {"hands_visible": False, "shoulders_visible": False},
                        "weights_used": {},
                        "warnings": ["Loading analysis models…"],
                        "frames_analyzed": 0,
                    })
                except Exception:
                    break

            # Process frame in a thread to avoid blocking the event loop
            try:
                result = await asyncio.to_thread(
                    state.engine.process_frame, frame_bytes
                )
                if result is None:
                    continue
                _models_initialized = True
                state.record_score(result)
            except Exception:
                logger.exception(
                    "Frame processing failed for session %s", session_id
                )
                continue

            # Send score update every SCORE_UPDATE_INTERVAL seconds
            elapsed = time.time() - state.last_update_time
            if elapsed >= config.SCORE_UPDATE_INTERVAL:
                update = state.build_window_update()
                try:
                    await websocket.send_json(update)
                except (WebSocketDisconnect, RuntimeError, ConnectionError):
                    logger.info("Session %s: connection closed during send", session_id)
                    break

    except WebSocketDisconnect:
        logger.info("Session %s disconnected", session_id)
    except Exception:
        logger.exception("Error in session %s", session_id)
    finally:
        # Persist before reporting completion so an immediate REST lookup is
        # guaranteed to see the finalized session.
        summary = state.build_summary()
        await _persist_summary(summary)
        try:
            await websocket.send_json(summary)
        except Exception:
            logger.debug("Could not send summary — connection closed")

        try:
            state.engine.close()
        except Exception:
            logger.exception("Failed to release analysis resources for %s", session_id)
        logger.info(
            "Session %s finalized (%d frames, %.1fs)",
            session_id,
            state.frame_count,
            summary["duration_seconds"],
        )
