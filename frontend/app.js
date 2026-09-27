/* ================================================================
   Confidence Engine — Frontend Application
   ================================================================ */

const signals = [
    'eye_contact', 'blink_rate', 'brow_tension',
    'smile', 'hand_stability', 'fidgeting',
    'hand_openness', 'shoulder_alignment', 'posture'
];

const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
let wsBase = null;

function websocketToHttp(base) {
    return base.replace(/^wss:/, 'https:').replace(/^ws:/, 'http:');
}

async function isConfidenceBackend(base) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000);
    try {
        const response = await fetch(`${websocketToHttp(base)}/health`, {
            cache: 'no-store',
            signal: controller.signal
        });
        if (!response.ok) return false;
        const payload = await response.json();
        return payload.status === 'ok' && payload.service === 'confidence-engine';
    } catch (error) {
        return false;
    } finally {
        clearTimeout(timeout);
    }
}

async function resolveWebSocketBase() {
    if (window.CONFIDENCE_ENGINE_WS_URL) {
        const configured = window.CONFIDENCE_ENGINE_WS_URL.replace(/\/$/, '');
        if (await isConfidenceBackend(configured)) return configured;
        throw new Error(`Configured backend is unavailable at ${configured}`);
    }

    const hostname = window.location.hostname || 'localhost';
    const candidates = [];
    const isLocalHost = hostname === 'localhost' || hostname === '127.0.0.1';
    if (isLocalHost) {
        candidates.push(`${WS_PROTOCOL}//${hostname}:8765`);
    }
    if (window.location.protocol === 'http:' || window.location.protocol === 'https:') {
        candidates.push(`${WS_PROTOCOL}//${window.location.host}`);
    }
    if (!isLocalHost) {
        candidates.push(`${WS_PROTOCOL}//${hostname}:8765`);
    }

    for (const candidate of [...new Set(candidates)]) {
        if (await isConfidenceBackend(candidate)) return candidate;
    }

    throw new Error(`Start the backend on http://${hostname}:8765 and try again`);
}

// Elements
const videoElement = document.getElementById('videoElement');
const captureCanvas = document.getElementById('captureCanvas');
const fallbackUI = document.getElementById('fallbackUI');
const retryBtn = document.getElementById('retryBtn');
const startBtn = document.getElementById('startBtn');
const endBtn = document.getElementById('endBtn');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const gaugeArc = document.getElementById('gaugeArc');
const scoreValue = document.getElementById('scoreValue');
const signalsList = document.getElementById('signalsList');
const summaryPanel = document.getElementById('summaryPanel');
const toastContainer = document.getElementById('toastContainer');

// State
let ws = null;
let stream = null;
let sessionId = null;
let isSessionActive = false;
let _captureTimer = null;
let _summaryTimer = null;
const FRAME_INTERVAL = 100; // 10 FPS
const BUFFERED_AMOUNT_THRESHOLD = 50000;

// ================================================================
// UI Functions
// ================================================================

function initSignalsUI() {
    signalsList.innerHTML = '';
    signals.forEach(signal => {
        const item = document.createElement('div');
        item.className = 'signal-item';

        const labelStr = signal.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');

        item.innerHTML = `
            <div class="signal-header">
                <span>${labelStr}</span>
                <span id="${signal}Value">--</span>
            </div>
            <div class="signal-bar-bg">
                <div class="signal-bar-fill" id="${signal}Bar"></div>
            </div>
        `;
        signalsList.appendChild(item);
    });
}

function resetAnalysisUI() {
    scoreValue.textContent = '--';
    gaugeArc.style.strokeDashoffset = 236;
    gaugeArc.style.stroke = 'var(--text-secondary)';
    gaugeArc.style.filter = 'none';

    signals.forEach(signal => {
        const valueEl = document.getElementById(`${signal}Value`);
        const barEl = document.getElementById(`${signal}Bar`);
        if (valueEl) valueEl.textContent = '--';
        if (barEl) {
            barEl.style.width = '0%';
            barEl.style.backgroundColor = 'var(--text-secondary)';
        }
    });
}

function generateUUID() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        return window.crypto.randomUUID();
    }
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
}

function getColorForScore(score) {
    if (score >= 70) return 'var(--color-success)';
    if (score >= 40) return 'var(--color-warning)';
    return 'var(--color-danger)';
}

function updateGauge(score) {
    if (score === null || score === undefined || Number.isNaN(score)) {
        scoreValue.textContent = '--';
        gaugeArc.style.strokeDashoffset = 236;
        gaugeArc.style.stroke = 'var(--text-secondary)';
        gaugeArc.style.filter = 'none';
        return;
    }

    const clamped = Math.max(0, Math.min(100, score));
    scoreValue.textContent = Math.round(clamped);
    const offset = 236 - (clamped / 100) * 236;
    gaugeArc.style.strokeDashoffset = offset;

    const color = getColorForScore(clamped);
    gaugeArc.style.stroke = color;
    gaugeArc.style.filter = `drop-shadow(0 0 8px ${color})`;
}

function updateSignals(data) {
    signals.forEach(signal => {
        const val = data[signal];
        const valueEl = document.getElementById(`${signal}Value`);
        const bar = document.getElementById(`${signal}Bar`);
        if (!valueEl || !bar) return;

        if (val === null || val === undefined || Number.isNaN(val)) {
            valueEl.textContent = '--';
            bar.style.width = '0%';
            bar.style.backgroundColor = 'var(--text-secondary)';
            return;
        }

        const clamped = Math.max(0, Math.min(100, val));
        valueEl.textContent = Math.round(clamped);
        bar.style.width = `${clamped}%`;
        bar.style.backgroundColor = getColorForScore(clamped);
    });
}

function showToast(message) {
    const duplicate = Array.from(toastContainer.children)
        .find(item => item.dataset.message === message);
    if (duplicate) return;

    while (toastContainer.children.length >= 3) {
        toastContainer.firstElementChild.remove();
    }

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    toast.dataset.message = message;
    toastContainer.appendChild(toast);

    requestAnimationFrame(() => {
        toast.classList.add('show');
    });

    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

function waitForVideoReady() {
    return new Promise((resolve) => {
        if (videoElement.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
            resolve();
            return;
        }
        videoElement.onloadeddata = () => resolve();
        setTimeout(resolve, 1500);
    });
}

async function requestCamera() {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Camera access requires HTTPS or localhost');
        }
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 960 },
                height: { ideal: 720 },
                facingMode: 'user'
            },
            audio: false
        });
        videoElement.srcObject = stream;
        fallbackUI.style.display = 'none';
        await waitForVideoReady();
        return true;
    } catch (err) {
        console.error('Camera error:', err);
        fallbackUI.style.display = 'flex';
        const message = err && err.message
            ? err.message
            : 'Please allow camera permissions and try again.';
        showToast(`Camera unavailable: ${message}`);
        return false;
    }
}

function startCapture() {
    const ctx = captureCanvas.getContext('2d');
    captureCanvas.width = 960;
    captureCanvas.height = 720;

    if (_captureTimer) {
        clearTimeout(_captureTimer);
        _captureTimer = null;
    }

    function captureFrame() {
        if (!isSessionActive) return;

        if (!ws || ws.readyState !== WebSocket.OPEN || ws.bufferedAmount > BUFFERED_AMOUNT_THRESHOLD) {
            _captureTimer = setTimeout(captureFrame, FRAME_INTERVAL);
            return;
        }
        if (videoElement.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
            _captureTimer = setTimeout(captureFrame, FRAME_INTERVAL);
            return;
        }

        ctx.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);

        captureCanvas.toBlob(blob => {
            if (blob && ws && ws.readyState === WebSocket.OPEN) {
                try { ws.send(blob); } catch (e) { /* ignore */ }
            }
            if (isSessionActive) {
                _captureTimer = setTimeout(captureFrame, FRAME_INTERVAL);
            }
        }, 'image/jpeg', 0.85);
    }

    captureFrame();
}

function stopCapture() {
    if (_captureTimer) {
        clearTimeout(_captureTimer);
        _captureTimer = null;
    }
}

function connectWebSocket() {
    if (!isSessionActive) return;

    const wsUrl = `${wsBase}/ws/confidence/${sessionId}`;
    const socket = new WebSocket(wsUrl);
    ws = socket;

    socket.onopen = () => {
        if (!isSessionActive || ws !== socket) {
            socket.close();
            return;
        }
        statusDot.className = 'status-dot live';
        statusText.textContent = 'Connected — starting models';
        startCapture();
    };

    socket.onmessage = (event) => {
        if (ws !== socket) return;
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'score_update') {
                updateGauge(data.composite_score);
                updateSignals(data.signals || {});

                // Section 4: Dynamic status indicator
                const visibility = data.visibility || {};
                if (data.warnings && data.warnings.length > 0) {
                    const warn = data.warnings[0];
                    if (warn.includes('performance')) {
                        statusDot.className = 'status-dot warning';
                    } else if (warn.includes('not detected')) {
                        statusDot.className = 'status-dot warning';
                    } else if (warn.includes('failed') || warn.includes('error')) {
                        statusDot.className = 'status-dot';
                    }
                    statusText.textContent = warn;

                    // Forward important warnings to user as toast (rate limited)
                    window._lastWarningToast = window._lastWarningToast || 0;
                    const now = Date.now();
                    if (now - window._lastWarningToast > 8000) {
                        window._lastWarningToast = now;
                        if (warn.includes('No face')) {
                            showToast('No face detected — move closer to the camera');
                        } else if (warn.includes('Hands not')) {
                            showToast('Hands not detected — keep hands visible');
                        } else if (warn.includes('Shoulders not')) {
                            showToast('Shoulders not detected — step back slightly');
                        } else if (warn.includes('failed') || warn.includes('error')) {
                            showToast('Analysis engine issue: ' + warn);
                        }
                    }
                } else if (data.composite_score !== null && data.composite_score !== undefined) {
                    if (visibility.shoulders_visible) {
                        statusText.textContent = 'Pose analysis active';
                        statusDot.className = 'status-dot live';
                    } else if (visibility.hands_visible) {
                        statusText.textContent = 'Hands detected';
                        statusDot.className = 'status-dot live';
                    } else {
                        statusText.textContent = 'Live';
                        statusDot.className = 'status-dot live';
                    }
                }

            } else if (data.type === 'session_summary') {
                showSummary(data);
            }
        } catch (e) {
            console.error('Failed to parse message', e);
        }
    };

    socket.onclose = () => {
        if (ws !== socket) return;
        stopCapture();

        if (isSessionActive) {
            isSessionActive = false;
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            videoElement.srcObject = null;
            statusDot.className = 'status-dot';
            statusText.textContent = 'Session stopped';
            startBtn.disabled = false;
            endBtn.disabled = true;
            showToast('Session stopped because the backend connection closed. Start a new session.');
        }
    };

    socket.onerror = () => {
        statusDot.className = 'status-dot';
        statusText.textContent = 'Backend connection error';
    };
}

async function startSession() {
    startBtn.disabled = true;
    statusDot.className = 'status-dot';
    statusText.textContent = 'Checking backend…';

    try {
        wsBase = await resolveWebSocketBase();
    } catch (error) {
        console.error('Backend discovery failed:', error);
        statusText.textContent = 'Backend unavailable';
        startBtn.disabled = false;
        showToast(error.message || 'The confidence backend is unavailable.');
        return;
    }

    const hasCamera = await requestCamera();
    if (!hasCamera) {
        statusText.textContent = 'Camera unavailable';
        startBtn.disabled = false;
        return;
    }

    sessionId = generateUUID();
    isSessionActive = true;
    if (_summaryTimer) {
        clearTimeout(_summaryTimer);
        _summaryTimer = null;
    }

    resetAnalysisUI();
    statusDot.className = 'status-dot';
    statusText.textContent = 'Connecting...';
    endBtn.disabled = false;
    summaryPanel.style.display = 'none';

    connectWebSocket();
}

function endSession() {
    isSessionActive = false;
    stopCapture();

    if (ws && ws.readyState === WebSocket.OPEN) {
        try {
            ws.send(JSON.stringify({ type: 'end_session' }));
        } catch (e) { /* ignore */ }
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    videoElement.srcObject = null;

    statusDot.className = 'status-dot ended';
    statusText.textContent = 'Session Ended';
    startBtn.disabled = false;
    endBtn.disabled = true;

    _summaryTimer = setTimeout(() => {
        _summaryTimer = null;
        if (ws) {
            try { ws.close(); } catch (e) { /* ignore */ }
        }
        if (summaryPanel.style.display === 'none') {
            showSummary({
                session_id: sessionId,
                duration_seconds: 0,
                average_score: 0,
                min_score: 0,
                max_score: 0
            });
        }
    }, 5000);
}

// ================================================================
// Session Summary
// ================================================================
function showSummary(data) {
    isSessionActive = false;
    stopCapture();
    if (_summaryTimer) {
        clearTimeout(_summaryTimer);
        _summaryTimer = null;
    }

    const durationSecs = data.duration_seconds || 0;
    const mins = Math.floor(durationSecs / 60);
    const secs = Math.round(durationSecs % 60);

    const sidEl = document.getElementById('summarySessionId');
    if (sidEl && data.session_id) {
        sidEl.textContent = data.session_id.length > 20
            ? 'ID: ' + data.session_id.slice(0, 20) + '…'
            : 'ID: ' + data.session_id;
    }

    // Overall score out of 10
    const avgScore = data.average_score || 0;
    const overallOutOf10 = (avgScore / 10).toFixed(1);
    const overallArc = document.getElementById('overallScoreArc');
    const overallValue = document.getElementById('overallScoreValue');
    const circumference = 2 * Math.PI * 60; // r=60
    const offset = circumference - (avgScore / 100) * circumference;
    if (overallArc) {
        overallArc.style.strokeDashoffset = offset;
        overallArc.style.stroke = getColorForScore(avgScore);
    }
    if (overallValue) {
        overallValue.textContent = overallOutOf10;
        overallValue.style.color = getColorForScore(avgScore);
    }
    const descEl = document.getElementById('overallScoreDescription');
    if (descEl) {
        if (avgScore >= 75) {
            descEl.textContent = 'Strong visual confidence presentation';
        } else if (avgScore >= 50) {
            descEl.textContent = 'Moderate confidence — room to improve';
        } else {
            descEl.textContent = 'Practice and feedback will help';
        }
    }

    document.getElementById('summaryDuration').textContent = `${mins}m ${secs}s`;
    document.getElementById('summaryAvg').textContent = Math.round(avgScore);
    document.getElementById('summaryFrames').textContent = data.total_frames || 0;

    const rangeMin = Math.round(data.min_score || 0);
    const rangeMax = Math.round(data.max_score || 0);
    const rangeEl = document.getElementById('summaryRange');
    rangeEl.textContent = `${rangeMin} – ${rangeMax}`;
    rangeEl.style.color = getColorForScore((rangeMin + rangeMax) / 2);

    // Timeline
    const timelineEl = document.getElementById('timelineChart');
    timelineEl.innerHTML = '';
    const timeline = data.score_timeline || [];
    if (timeline.length > 0) {
        timeline.forEach(entry => {
            const score = entry.score !== null && entry.score !== undefined ? entry.score : 0;
            const bar = document.createElement('div');
            bar.className = 'timeline-bar';
            const pct = Math.max(3, Math.min(100, score));
            bar.style.height = `${pct}%`;
            bar.style.backgroundColor = getColorForScore(score);
            bar.title = `Score: ${Math.round(score)}`;
            timelineEl.appendChild(bar);
        });
    } else {
        timelineEl.innerHTML = '<span style="color: var(--text-secondary); font-size: 0.85rem;">No timeline data</span>';
    }

    // Breakdown
    const bodyEl = document.getElementById('breakdownBody');
    bodyEl.innerHTML = '';
    const signalsSummary = data.signals_summary || {};
    if (Object.keys(signalsSummary).length > 0) {
        const signalOrder = [
            'eye_contact', 'blink_rate', 'brow_tension', 'smile',
            'hand_stability', 'fidgeting', 'hand_openness',
            'shoulder_alignment', 'posture'
        ];
        signalOrder.forEach(key => {
            const info = signalsSummary[key];
            if (!info) return;
            const row = document.createElement('div');
            row.className = 'breakdown-row';

            const labelStr = key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');

            const avgVal = info.avg !== null && info.avg !== undefined ? Math.round(info.avg) : '--';
            const minVal = info.min !== null && info.min !== undefined ? Math.round(info.min) : '--';
            const maxVal = info.max !== null && info.max !== undefined ? Math.round(info.max) : '--';

            const avgColor = info.avg !== null ? getColorForScore(info.avg) : 'var(--text-secondary)';
            const barWidth = info.avg !== null ? `${Math.max(5, Math.min(100, info.avg))}%` : '0%';

            row.innerHTML = `
                <span class="col-signal">${labelStr}</span>
                <span class="col-avg" style="color: ${avgColor}; font-weight: 600;">${avgVal}</span>
                <span class="col-min">${minVal}</span>
                <span class="col-max">${maxVal}</span>
                <div class="breakdown-bar-bg">
                    <div class="breakdown-bar-fill" style="width: ${barWidth}; background: ${avgColor};"></div>
                </div>
            `;
            bodyEl.appendChild(row);
        });
    } else {
        bodyEl.innerHTML = '<div class="breakdown-empty">No signal data collected</div>';
    }

    // Low confidence periods
    const lowSecEl = document.getElementById('lowConfidenceSection');
    const flagsList = document.getElementById('lowConfidenceFlags');
    flagsList.innerHTML = '';
    if (data.low_confidence_timestamps && data.low_confidence_timestamps.length > 0) {
        lowSecEl.style.display = 'block';
        data.low_confidence_timestamps.forEach(ts => {
            const div = document.createElement('div');
            div.className = 'flag-item';
            const startStr = ts.start ? ts.start.slice(11, 19) : '?';
            const endStr = ts.end ? ts.end.slice(11, 19) : '?';
            div.textContent = `${startStr} → ${endStr}  ·  Avg: ${Math.round(ts.avg_score)}`;
            flagsList.appendChild(div);
        });
    } else {
        lowSecEl.style.display = 'none';
    }

    summaryPanel.style.display = 'block';
    requestAnimationFrame(() => {
        summaryPanel.classList.add('visible');
    });

    if (ws) {
        try { ws.close(); } catch (e) { /* ignore */ }
    }
}

// ================================================================
// Event Listeners
// ================================================================

document.getElementById('closeSummaryBtn').addEventListener('click', () => {
    summaryPanel.classList.remove('visible');
    setTimeout(() => {
        summaryPanel.style.display = 'none';
    }, 400);
});

startBtn.addEventListener('click', startSession);
endBtn.addEventListener('click', endSession);
retryBtn.addEventListener('click', requestCamera);

initSignalsUI();
resetAnalysisUI();
