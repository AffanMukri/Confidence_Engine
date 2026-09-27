"""
Integration test for the WebSocket confidence endpoint.

Uses FastAPI's TestClient to open a WebSocket, send synthetic JPEG frames,
verify score_update and session_summary messages, and check DB persistence.
"""

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_test_jpeg(width: int = 640, height: int = 480) -> bytes:
    """Create a minimal valid JPEG image (solid colour, no face)."""
    import cv2
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (128, 128, 128)  # grey
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def test_app():
    """Create the FastAPI TestClient once for the module.
    Tables are created synchronously at module load in conftest.py.
    """
    # We import here so that mediapipe is only loaded if available
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from main import app
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestWebSocketBasic:
    """Basic WebSocket connection and message handling."""

    def test_connect_and_receive_summary(self, test_app):
        """Connect, send a few frames, then disconnect and get summary."""
        session_id = "test-session-001"
        jpeg = _make_test_jpeg()

        with test_app.websocket_connect(
            f"/ws/confidence/{session_id}"
        ) as ws:
            # Send 5 frames
            for _ in range(5):
                ws.send_bytes(jpeg)

            # Send end_session command
            ws.send_text(json.dumps({"type": "end_session"}))

            message = ws.receive_json()
            assert message["type"] == "session_summary"
            assert message["session_id"] == session_id
            assert message["total_frames"] >= 0
            assert "score_timeline" in message

        # Persistence is part of finalization, not merely a best-effort detail.
        response = test_app.get(f"/api/sessions/{session_id}")
        assert response.status_code == 200
        assert response.json()["session_id"] == session_id

    def test_no_face_handling(self, test_app):
        """A solid-color image (no face) should not crash the pipeline."""
        session_id = "test-session-noface"
        jpeg = _make_test_jpeg()

        with test_app.websocket_connect(
            f"/ws/confidence/{session_id}"
        ) as ws:
            ws.send_bytes(jpeg)
            ws.send_bytes(jpeg)

            # Send end_session
            ws.send_text(json.dumps({"type": "end_session"}))
            summary = ws.receive_json()
            assert summary["type"] == "session_summary"
            assert summary["session_id"] == session_id

    def test_repeated_session_id_updates_persisted_summary(self, test_app):
        """A reconnect/finalize cycle updates instead of violating uniqueness."""
        session_id = "repeat-session-001"

        with test_app.websocket_connect(
            f"/ws/confidence/{session_id}"
        ) as ws:
            ws.send_text(json.dumps({"type": "end_session"}))
            assert ws.receive_json()["type"] == "session_summary"

        with test_app.websocket_connect(
            f"/ws/confidence/{session_id}"
        ) as ws:
            ws.send_bytes(_make_test_jpeg())
            first = ws.receive_json()
            assert first["type"] == "score_update"
            ws.send_text(json.dumps({"type": "end_session"}))
            summary = ws.receive_json()
            assert summary["type"] == "session_summary"
            assert summary["total_frames"] == 1

        response = test_app.get(f"/api/sessions/{session_id}")
        assert response.status_code == 200
        assert response.json()["total_frames"] == 1


class TestRESTEndpoints:
    """Test the REST session retrieval endpoints."""

    def test_get_nonexistent_session(self, test_app):
        """GET a session that doesn't exist → 404."""
        response = test_app.get("/api/sessions/nonexistent-id-12345")
        assert response.status_code == 404

    def test_list_sessions(self, test_app):
        """GET /api/sessions/ should return a valid response."""
        response = test_app.get("/api/sessions/")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    def test_health_check(self, test_app):
        """GET /health should return ok."""
        response = test_app.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
