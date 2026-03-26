# ASR Streaming Server — API Specification

**Base URL:** `http://<host>:8765`

---

## Health & Status

### `GET /healthz`

Liveness probe. Returns 200 if the process is running.

**Response** `200`

```json
{
  "status": "alive",
  "uptime_s": 3421.7
}
```

---

### `GET /readyz`

Readiness probe. Returns 200 only after all models (Nemotron, Silero VAD, Smart Turn) are loaded and warm.

**Response** `200`

```json
{
  "status": "ready",
  "uptime_s": 3421.7
}
```

**Response** `503` (still loading)

```json
{
  "status": "loading"
}
```

---

### `GET /v1/info`

Server configuration and current state.

**Response** `200`

```json
{
  "model": "nvidia/nemotron-speech-streaming-en-0.6b",
  "sample_rate": 16000,
  "chunk_ms": 560,
  "lookahead_ms": 480,
  "vram_limit_gb": 8.0,
  "max_connections": 50,
  "active_connections": 3
}
```

---

### `GET /v1/sessions`

All active streaming sessions across transports.

**Response** `200`

```json
{
  "active": 3,
  "max": 50,
  "sessions": [
    { "id": "10.0.0.5:41022-1711532800000", "transport": "ws", "duration_s": 45.2 },
    { "id": "a1b2c3d4e5f6", "transport": "webrtc", "state": "connected", "duration_s": 12.8 }
  ]
}
```

---

## One-Shot Transcription

### `POST /v1/transcribe`

Upload an audio file, get the full transcript back. No streaming — for batch use and testing.

**Request** `multipart/form-data`

| Field      | Type   | Required | Description                |
|------------|--------|----------|----------------------------|
| `file`     | file   | yes      | Audio file (WAV, FLAC, etc.) |
| `language` | string | no       | Language code (default: `en`) |

**Response** `200`

```json
{
  "text": "I would use a hashmap to store the frequency of each element",
  "duration_s": 4.52,
  "processing_s": 1.203
}
```

**Response** `503`

```json
{
  "error": "models loading"
}
```

---

## WebSocket Streaming

### `WS /v1/ws/transcribe`

Full-duplex streaming ASR. Client sends raw audio, server sends transcription events.

#### Connection

```
ws://<host>:8765/v1/ws/transcribe
```

On successful connection the server sends:

```json
{ "type": "ready" }
```

If the server is at capacity:

```json
{ "type": "error", "message": "server full" }
```

Then the connection is closed.

#### Client → Server

**Binary frames** — raw `int16` PCM audio at **16kHz mono**, little-endian. Any frame size is accepted; the server buffers internally to its required chunk sizes (32ms for VAD, 560ms for ASR).

**JSON frames:**

| Message                | Description                          |
|------------------------|--------------------------------------|
| `{"type": "reset"}`    | End current utterance, start a new one. Flushes buffered audio and emits a `final` if there is pending text. |

#### Server → Client

All messages are JSON with a `type` field.

##### `vad_start`

Speech detected — the user started talking.

```json
{
  "type": "vad_start",
  "text": "",
  "is_final": false,
  "step": 0,
  "turn_probability": 0,
  "timestamp": 1711532845.123
}
```

##### `partial`

Interim transcript. Updates as more audio arrives. Replaces the previous partial.

```json
{
  "type": "partial",
  "text": "I would use a hashmap to store the",
  "is_final": false,
  "step": 5,
  "turn_probability": 0,
  "timestamp": 1711532847.456
}
```

##### `vad_end`

Speech stopped — silence detected after speech.

```json
{
  "type": "vad_end",
  "text": "",
  "is_final": false,
  "step": 0,
  "turn_probability": 0,
  "timestamp": 1711532849.789
}
```

##### `turn_result`

Smart Turn v3 evaluation result. Emitted after every `vad_end`. If probability is above threshold (default 0.5), the turn is considered complete and a `final` follows immediately.

```json
{
  "type": "turn_result",
  "text": "",
  "is_final": false,
  "step": 0,
  "turn_probability": 0.92,
  "timestamp": 1711532849.801
}
```

##### `final`

Confirmed end-of-turn transcript. This is the authoritative text for this utterance. The pipeline resets automatically — no need to send `{"type": "reset"}`.

```json
{
  "type": "final",
  "text": "I would use a hashmap to store the frequency of each element",
  "is_final": true,
  "step": 8,
  "turn_probability": 0.92,
  "timestamp": 1711532849.812
}
```

##### `error`

```json
{
  "type": "error",
  "message": "server full"
}
```

#### Event Sequence (typical)

```
Server: {"type": "ready"}

       [user starts speaking]
Server: {"type": "vad_start"}
Server: {"type": "partial", "text": "I would"}
Server: {"type": "partial", "text": "I would use a"}
Server: {"type": "partial", "text": "I would use a hashmap"}
Server: {"type": "partial", "text": "I would use a hashmap to store the"}

       [user pauses 2 seconds thinking]
Server: {"type": "partial", "text": "I would use a hashmap to store the frequency"}

       [user continues]
Server: {"type": "partial", "text": "I would use a hashmap to store the frequency of each element"}

       [user stops, 300ms silence]
Server: {"type": "vad_end"}
Server: {"type": "turn_result", "turn_probability": 0.92}
Server: {"type": "final", "text": "I would use a hashmap to store the frequency of each element"}

       [pipeline auto-resets, ready for next utterance]
Server: {"type": "vad_start"}
       ...
```

#### Event Sequence (user pauses mid-sentence)

```
Server: {"type": "vad_start"}
Server: {"type": "partial", "text": "so the time complexity is"}

       [user pauses 400ms to think]
Server: {"type": "vad_end"}
Server: {"type": "turn_result", "turn_probability": 0.23}

       [Smart Turn says NOT done — pipeline stays in LISTENING]
       [user resumes]
Server: {"type": "vad_start"}
Server: {"type": "partial", "text": "so the time complexity is O(n log n)"}

       [user finishes]
Server: {"type": "vad_end"}
Server: {"type": "turn_result", "turn_probability": 0.88}
Server: {"type": "final", "text": "so the time complexity is O(n log n)"}
```

---

## WebRTC Streaming

### `POST /v1/webrtc/offer`

Exchange SDP to establish a WebRTC peer connection. The server accepts an audio track and sends transcription events back via an `RTCDataChannel`.

**Request** `application/json`

```json
{
  "sdp": "v=0\r\no=- 4567 2 IN IP4 ...",
  "type": "offer"
}
```

**Response** `200`

```json
{
  "sdp": "v=0\r\no=- 1234 2 IN IP4 ...",
  "type": "answer",
  "session_id": "a1b2c3d4e5f6"
}
```

#### Audio Input

Send an audio track via the peer connection. Any codec supported by the browser (typically Opus) is accepted — the server decodes and resamples to 16kHz mono internally.

#### Transcription Output

Events arrive on the `RTCDataChannel` named `"transcription"`. The channel is created by the server. Messages are JSON strings with the **exact same format** as the WebSocket events above (`partial`, `final`, `vad_start`, `vad_end`, `turn_result`).

On channel open:

```json
{ "type": "ready" }
```

#### Browser Example

```javascript
const pc = new RTCPeerConnection({
  iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
});

// Capture microphone
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
stream.getTracks().forEach(track => pc.addTrack(track, stream));

// Listen for data channel (server creates it)
pc.ondatachannel = (event) => {
  const dc = event.channel;
  dc.onmessage = (msg) => {
    const evt = JSON.parse(msg.data);
    switch (evt.type) {
      case "ready":    console.log("connected"); break;
      case "partial":  updateLiveText(evt.text);  break;
      case "final":    commitText(evt.text);       break;
      case "vad_start": showMicActive();           break;
      case "vad_end":   showMicIdle();             break;
    }
  };
};

// SDP exchange
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

const resp = await fetch("/v1/webrtc/offer", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
});
const answer = await resp.json();
await pc.setRemoteDescription(answer);
```

---

### `POST /v1/webrtc/close`

Tear down a WebRTC session.

**Request** `application/json`

```json
{
  "session_id": "a1b2c3d4e5f6"
}
```

**Response** `200`

```json
{ "status": "closed" }
```

or if not found:

```json
{ "status": "not_found" }
```

---

### `GET /v1/webrtc/sessions`

List active WebRTC sessions.

**Response** `200`

```json
{
  "active": 2,
  "sessions": [
    { "id": "a1b2c3d4e5f6", "state": "connected", "duration_s": 45.2 },
    { "id": "f6e5d4c3b2a1", "state": "connected", "duration_s": 12.1 }
  ]
}
```

---

## Pipeline Internals

For clients that need to understand timing and tuning.

### Audio Requirements

| Parameter    | Value    |
|--------------|----------|
| Sample rate  | 16000 Hz |
| Channels     | 1 (mono) |
| Encoding     | int16 little-endian (WebSocket) or any browser codec (WebRTC) |
| Endianness   | little-endian |

### Processing Chain

```
Audio in
  │
  ▼
Silero VAD          32ms chunks (512 samples), CPU
  │                 Detects speech start/end
  ▼
Nemotron 0.6B       560ms chunks (8960 samples), GPU
  │                 Cache-aware streaming — each frame processed once
  ▼
Smart Turn v3       On VAD silence, evaluates last 8s of audio, CPU
  │                 12ms inference, returns turn-complete probability
  ▼
Code Term Fixer     Regex replacement, <0.1ms
  │
  ▼
Event emitted
```

### Latency Budget

| Stage | Latency |
|-------|---------|
| VAD decision | ~1ms |
| ASR chunk processing | ~50-100ms per 560ms chunk |
| Smart Turn evaluation | ~12ms |
| Code term regex | ~0.1ms |
| Network (local) | ~1ms |
| **Total after speech ends** | **~65-115ms** |

Note: partials are emitted every 560ms during speech. The latency above is only the delay between the user finishing their turn and the `final` event.

### Configuration

Tunables are in `src/asr/config.py`. Key parameters:

| Setting | Default | Effect |
|---------|---------|--------|
| `gpu.vram_limit_gb` | 8.0 | Max VRAM for ASR (of 24GB) |
| `asr.lookahead_frames` | 6 | Chunk size: 0→80ms, 1→160ms, 6→560ms, 13→1120ms |
| `vad.threshold` | 0.5 | VAD sensitivity (lower = more sensitive) |
| `vad.min_silence_ms` | 300 | Silence duration before VAD triggers end |
| `turn.threshold` | 0.5 | Turn completion probability threshold |
| `server.max_connections` | 50 | Max concurrent sessions (WS + WebRTC) |

### Error Handling

| Scenario | Behavior |
|----------|----------|
| Server at capacity | WS: `{"type":"error","message":"server full"}` then close. WebRTC: HTTP 503. |
| Models still loading | WS: close with code 1013. HTTP: 503. WebRTC: 503. |
| Client disconnects mid-stream | Pipeline cleaned up, session removed. No dangling state. |
| Invalid audio format | WS: best-effort processing. WebRTC: aiortc decodes any browser codec. |
| GPU OOM | Process crashes. Use `/healthz` + restart policy in K8s/systemd. |
