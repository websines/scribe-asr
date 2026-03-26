# asr

Streaming ASR server with automatic turn detection. Nemotron 0.6B + Silero VAD + Smart Turn v3.

```
Audio → Silero VAD (32ms, CPU) → Nemotron (560ms, GPU) → Smart Turn v3 (12ms, CPU) → transcript
```

## Stack

| Component | Model | Role | Runs on |
|-----------|-------|------|---------|
| ASR | [Nemotron 0.6B](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) | Cache-aware streaming transcription | GPU |
| VAD | [Silero VAD](https://github.com/snakers4/silero-vad) | Speech start/end detection | CPU |
| Turn Detection | [Smart Turn v3](https://huggingface.co/pipecat-ai/smart-turn-v3) | End-of-turn prediction from audio | CPU |
| Code Fixer | Regex/dict | Fix ASR mishearings of code terms | CPU |

## Transports

- **WebSocket** — `/v1/ws/transcribe` — send raw int16 PCM, receive JSON events
- **WebRTC** — `/v1/webrtc/offer` — SDP exchange, audio track in, data channel out
- **REST** — `/v1/transcribe` — one-shot file upload

Full spec in [apispec.md](apispec.md).

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA (tested on RTX 3090, 8GB VRAM allocated)
- WSL2 or native Linux

## Setup

```bash
# Clone
git clone <repo-url> && cd asr

# Create venv and install
uv venv
uv pip install -e .

# Start server
uv run asr-server
```

Server starts on `http://0.0.0.0:8765`. Models download on first run (~2GB).

## Quick Test

```bash
# Health check
curl http://localhost:8765/healthz

# Readiness (waits for models)
curl http://localhost:8765/readyz

# Server info
curl http://localhost:8765/v1/info

# One-shot transcription
curl -X POST http://localhost:8765/v1/transcribe -F file=@audio.wav

# Stream a file via WebSocket
uv run python test_client.py audio.wav

# Check all endpoints
uv run python test_client.py --health
```

## WebSocket Client (browser)

```javascript
const ws = new WebSocket("ws://localhost:8765/v1/ws/transcribe");
const ctx = new AudioContext({ sampleRate: 16000 });
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const source = ctx.createMediaStreamSource(stream);
const processor = ctx.createScriptProcessor(4096, 1, 1);

source.connect(processor);
processor.connect(ctx.destination);

processor.onaudioprocess = (e) => {
  const float32 = e.inputBuffer.getChannelData(0);
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
  }
  ws.send(int16.buffer);
};

ws.onmessage = (e) => {
  const evt = JSON.parse(e.data);
  if (evt.type === "partial") console.log("...", evt.text);
  if (evt.type === "final")   console.log(">>>", evt.text);
};
```

## WebRTC Client (browser)

```javascript
const pc = new RTCPeerConnection({
  iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
});

const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
stream.getTracks().forEach(t => pc.addTrack(t, stream));

pc.ondatachannel = (e) => {
  e.channel.onmessage = (msg) => {
    const evt = JSON.parse(msg.data);
    if (evt.type === "final") console.log(">>>", evt.text);
  };
};

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

## Events

| Event | When |
|-------|------|
| `ready` | Connection established, pipeline warm |
| `vad_start` | Speech detected |
| `partial` | Interim transcript (updates in-place) |
| `vad_end` | Silence after speech |
| `turn_result` | Smart Turn evaluation (probability 0-1) |
| `final` | Confirmed end-of-turn transcript |

## Configuration

Edit `src/asr/config.py`:

| Setting | Default | Notes |
|---------|---------|-------|
| `gpu.vram_limit_gb` | 8.0 | VRAM cap (of 24GB on 3090) |
| `asr.lookahead_frames` | 6 | 0→80ms, 6→560ms, 13→1120ms |
| `vad.min_silence_ms` | 300 | Silence before evaluating turn |
| `turn.threshold` | 0.5 | Turn-complete probability cutoff |
| `server.max_connections` | 50 | Max concurrent sessions |

## Project Structure

```
src/asr/
├── config.py           Settings (pydantic)
├── vad.py              Silero VAD wrapper
├── turn_detector.py    Smart Turn v3 ONNX wrapper
├── engine.py           Nemotron streaming ASR engine
├── text_fixer.py       Code-term regex replacement
├── pipeline.py         State machine: VAD → ASR → Turn
├── webrtc.py           WebRTC signaling + audio track handler
└── server.py           FastAPI server (HTTP + WS + WebRTC)
```

## License

- Nemotron ASR: [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) (commercial OK, attribution required)
- Smart Turn v3: BSD-2-Clause
- Silero VAD: MIT
