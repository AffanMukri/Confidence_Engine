import os
import requests

BASE_URL = os.environ.get("TESTSPRITE_TARGET_URL", "https://fruity-windows-agree.loca.lt")


def test_get_missing_session_returns_404() -> None:
    response = requests.get(f"{BASE_URL}/api/sessions/nonexistent-session-id", timeout=30)
    assert response.status_code == 404, f"expected 404, got {response.status_code}"
    payload = response.json()
    assert "detail" in payload, f"missing detail key: {payload}"


test_get_missing_session_returns_404()
