import os
import requests

BASE_URL = os.environ.get("TESTSPRITE_TARGET_URL", "https://fruity-windows-agree.loca.lt")


def test_list_sessions_shape() -> None:
    response = requests.get(f"{BASE_URL}/api/sessions/", timeout=30)
    assert response.status_code == 200, f"expected 200, got {response.status_code}"
    payload = response.json()
    assert "sessions" in payload, f"missing sessions key: {payload}"
    assert "count" in payload, f"missing count key: {payload}"
    assert isinstance(payload["sessions"], list), f"sessions should be a list: {payload}"
    assert payload["count"] == len(payload["sessions"]), f"count mismatch: {payload}"


test_list_sessions_shape()
