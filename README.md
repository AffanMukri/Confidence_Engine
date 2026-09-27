# Confidence Scoring Engine

Real-time interview confidence analysis via MediaPipe. Captures webcam video, extracts facial, hand, and posture signals, and produces a composite 0–100 confidence score streamed over WebSocket.

## Quick Start

### Prerequisites
- Python 3.10+
- A webcam and a current Chrome or Edge browser
- PostgreSQL 14+ is optional; SQLite is used automatically for local development

### Setup

```bash
# 1. Install dependencies
cd confidence-engine/backend
pip install -r requirements.txt

# 2. (Optional) Use PostgreSQL instead of the built-in SQLite database
export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/confidence_db"
export ALLOWED_ORIGINS="*"

# 3. Start the server
uvicorn main:app --reload --host 127.0.0.1 --port 8765
```

### Demo UI

Open `http://localhost:8765` in Chrome/Edge. Click **Start Session**, grant camera access, and watch the live confidence gauge. Serving the UI through FastAPI keeps camera permissions and the WebSocket on the correct origin.

On Windows, you can also double-click `start.bat` in the project folder, keep the opened server window running, and then visit `http://localhost:8765`.

For a separate frontend deployment, build the frontend with `VITE_API_URL`
set to the backend's public HTTP origin. The browser automatically converts an
`https://` API origin to `wss://` for the live confidence stream.

```bash
# Local static frontend build
VITE_API_URL=http://localhost:8765 npm run build
```

## Production deployment

The production application has two deployment units:

- **Netlify:** static HTML, CSS, and JavaScript from `dist/`
- **Render (or another WebSocket-capable Python host):** FastAPI, MediaPipe,
  REST endpoints, WebSocket analysis, and persistence

Netlify cannot run this Python/MediaPipe backend. The frontend therefore talks
directly to one configured HTTPS backend origin; it never derives a backend
port from the Netlify hostname.

### Backend on Render

The included `render.yaml` configures a Render Web Service. For manual setup,
use these exact settings:

```text
Root directory: backend
Build command:  pip install -r requirements.txt
Start command:  uvicorn main:app --host 0.0.0.0 --port $PORT
Health check:   /health
Python:         3.12
```

Backend environment variables:

```dotenv
ALLOWED_ORIGINS=https://confidence-engine.netlify.app,http://localhost:8765,http://127.0.0.1:8765
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DATABASE
MP_DELEGATE=CPU
```

`PORT` is supplied by Render and read by the app. `DATABASE_URL` is strongly
recommended for persistent production sessions; without it, SQLite uses the
host's temporary filesystem and data can disappear after a restart or deploy.
Standard hosted `postgres://` and `postgresql://` values are automatically
converted to SQLAlchemy's async `postgresql+asyncpg://` form.

After the backend deploys, confirm that
`https://YOUR-BACKEND.onrender.com/health` returns:

```json
{"status":"ok","service":"confidence-engine"}
```

### Frontend on Netlify

The committed `netlify.toml` supplies the build and publish settings:

```text
Base directory:      (blank / repository root)
Build command:       npm run build
Publish directory:   dist
Functions directory: (blank)
```

Add this Netlify environment variable for every deploy context:

```dotenv
VITE_API_URL=https://YOUR-BACKEND.onrender.com
```

Do not include `/health`, `/api`, `/ws`, or a trailing port. Netlify builds are
rejected if this value is missing or does not use HTTPS. Trigger a new deploy
after changing it.

---

## Architecture

```
Browser (getUserMedia → JPEG frames at 10 FPS)
    │
    ▼  WebSocket (binary JPEG)
FastAPI Backend
    ├── Face Mesh (478 landmarks, iris tracking)
    ├── Hands (21 landmarks × 2 hands)
    └── Pose (33 landmarks, optional)
    │
    ▼  Score update every 2 seconds
Browser (animated gauge + 9 signal bars)
    │
    ▼  On session end
PostgreSQL (confidence_sessions table)
```

---

## Scoring Formula

### Sub-Signals (9 total)

| Signal | Source | How It's Calculated | Score Range |
|--------|--------|---------------------|-------------|
| **Eye Contact** | Face Mesh | Iris center offset from eye bounding box center (landmarks 468-477 vs 33/133, 263/362). Centered = 100, extreme deviation = 0. | 0–100 |
| **Blink Rate** | Face Mesh | Eye Aspect Ratio (EAR) tracks blinks/minute. Normal range (15-20/min) = 100. | 0–100 |
| **Brow Tension** | Face Mesh | Vertical distance between inner brow (65, 295) and nose (168), normalized by face height. Relaxed = 100. | 0–100 |
| **Smile** | Face Mesh | Mouth width (61, 291) / face width (234, 454). Gentle smile = bonus above 80 baseline. | 0–100 |
| **Hand Stability** | Hands | Velocity variance of wrist and fingertips over a rolling window. Low variance = 100. | 0–100 |
| **Fidgeting** | Hands | Jerk metric (rapid direction changes). Low jerk = 100. | 0–100 |
| **Hand Openness** | Hands | Fingertip-to-wrist / knuckle-to-wrist distance ratio. Open/relaxed = higher. | 0–100 |
| **Shoulder Alignment** | Pose | Angle of shoulder line vs horizontal. Level = 100, tilted = lower. | 0–100 |
| **Posture** | Pose | Ear-to-shoulder vertical alignment. Upright = 100, forward lean = lower. | 0–100 |

### EAR (Eye Aspect Ratio) Formula

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

Landmarks: Left eye `[362, 385, 386, 263, 374, 380]`, Right eye `[33, 160, 158, 133, 153, 145]`

### Weighted Composite Score

```
Face (50% total):
  Eye Contact:     0.20
  Blink Rate:      0.10
  Brow Tension:    0.10
  Smile:           0.10

Hands (30% total):
  Hand Stability:  0.12
  Fidgeting:       0.12
  Hand Openness:   0.06

Pose (20% total):
  Shoulder Align:  0.10
  Posture:         0.10
```

**Graceful Degradation**: If a signal is `None` (body part not detected), its weight is redistributed proportionally among available signals.

### EMA Smoothing

$$S_t = \alpha \cdot X_t + (1 - \alpha) \cdot S_{t-1}$$

- **α = 0.3** (configurable via `EMA_ALPHA` env var)
- Smooths over approximately 3 windows, reducing jitter while staying responsive

---

## How to Retune Weights

All tuning knobs are in `backend/config.py` and can be overridden via environment variables:

| Parameter | Default | Env Var | Effect |
|-----------|---------|---------|--------|
| Eye Contact weight | 0.20 | — | Edit `WEIGHTS` dict in `config.py` |
| EMA alpha | 0.3 | `EMA_ALPHA` | Higher = faster response, more noise |
| EAR threshold | 0.22 | `EAR_THRESHOLD` | Blink detection sensitivity |
| Blink consec frames | 3 | `BLINK_CONSEC_FRAMES` | Frames below EAR to count a blink |
| Normal blink range | 15-20/min | `NORMAL_BLINK_RATE_MIN/MAX` | What counts as "normal" blinking |
| Low confidence threshold | 50 | `LOW_CONFIDENCE_THRESHOLD` | Below this = flagged as low confidence |
| Score update interval | 2.0s | `SCORE_UPDATE_INTERVAL` | How often scores are pushed to client |

### Changing Weights

Edit the `WEIGHTS` dictionary in `config.py`. The weights **must sum to 1.0**:

```python
WEIGHTS = {
    "eye_contact": 0.25,     # Increase eye contact importance
    "blink_rate": 0.05,      # Decrease blink importance
    "brow_tension": 0.10,
    "smile": 0.10,
    "hand_stability": 0.12,
    "fidgeting": 0.12,
    "hand_openness": 0.06,
    "shoulder_alignment": 0.10,
    "posture": 0.10,
}
```

---

## API Reference

### WebSocket: `ws://localhost:8765/ws/confidence/{session_id}` (local)

Production uses `wss://YOUR-BACKEND/ws/confidence/{session_id}`.

**Client → Server**: Binary JPEG frame (every 100ms)

**Server → Client**: JSON every 2 seconds:
```json
{
  "type": "score_update",
  "timestamp": "2026-07-18T00:20:00Z",
  "composite_score": 73.5,
  "signals": {
    "eye_contact": 85.0,
    "blink_rate": 70.0,
    "brow_tension": 65.0,
    "smile": 80.0,
    "hand_stability": 60.0,
    "fidgeting": 55.0,
    "hand_openness": 75.0,
    "shoulder_alignment": 90.0,
    "posture": 85.0
  },
  "warnings": [],
  "frames_analyzed": 20
}
```

**On session end**: Server sends `session_summary` then persists to DB:
```json
{
  "type": "session_summary",
  "session_id": "abc-123",
  "average_score": 71.2,
  "min_score": 45.0,
  "max_score": 92.0,
  "low_confidence_timestamps": [
    {"start": "2026-07-18T00:22:15Z", "end": "2026-07-18T00:22:45Z", "avg_score": 42.3}
  ],
  "total_frames": 1800,
  "duration_seconds": 180
}
```

### REST

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/sessions/` | List recent sessions (paginated: `?limit=20&offset=0`) |
| `GET` | `/api/sessions/{session_id}` | Get a specific session summary |
| `GET` | `/health` | Health check |

---

## External Integration

### Option 1: WebSocket (separate frontend)

Point any frontend at this engine's WebSocket:

```javascript
const ws = new WebSocket(`wss://engine.example.com/ws/confidence/${yourSessionId}`);
// Send JPEG frames, receive score_update messages
```

### Option 2: Python import (in-process)

```python
from scoring import ConfidenceEngine

engine = ConfidenceEngine()
result = engine.process_frame(jpeg_bytes)
# result["composite_score"]  →  float 0-100
# result["signals"]          →  dict of 9 sub-scores

summary = engine.get_session_summary()
```

### Option 3: REST (post-session)

```bash
curl http://localhost:8765/api/sessions/your-session-id
```

---

## Testing

```bash
cd confidence-engine/backend
pip install -r requirements.txt
pytest tests/ -v --tb=short
```

All tests use synthetic landmarks — no webcam or GPU required.

---

## Project Structure

```
confidence-engine/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # All tunable parameters
│   ├── db.py                      # Async SQLAlchemy engine
│   ├── models/
│   │   └── session.py             # ORM model (confidence_sessions)
│   ├── scoring/
│   │   ├── __init__.py            # ConfidenceEngine public API
│   │   ├── face_analyzer.py       # Face Mesh signals
│   │   ├── hand_analyzer.py       # Hand signals
│   │   ├── pose_analyzer.py       # Pose signals
│   │   ├── composite.py           # Weighted scorer + EMA
│   │   └── frame_processor.py     # MediaPipe orchestrator
│   ├── ws/
│   │   └── confidence_ws.py       # WebSocket endpoint
│   ├── api/
│   │   └── sessions.py            # REST endpoints
│   ├── tests/
│   │   ├── conftest.py            # Shared fixtures
│   │   ├── test_face_analyzer.py
│   │   ├── test_hand_analyzer.py
│   │   ├── test_pose_analyzer.py
│   │   ├── test_composite.py
│   │   └── test_ws_integration.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── README.md
```
