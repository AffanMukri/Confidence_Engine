import os
import requests

BASE_URL = os.environ.get("TESTSPRITE_TARGET_URL", "https://fruity-windows-agree.loca.lt")


def test_health_endpoint() -> None:
    response = requests.get(f"{BASE_URL}/health", timeout=30)
    assert response.status_code == 200, f"expected 200, got {response.status_code}"
    payload = response.json()
    assert payload.get("status") == "ok", f"unexpected payload: {payload}"
    assert payload.get("service") == "confidence-engine", f"unexpected service: {payload}"


test_health_endpoint()
