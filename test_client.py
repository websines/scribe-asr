"""Test client — streams a WAV file to the ASR server.

Usage:
    uv run python test_client.py audio.wav
    uv run python test_client.py audio.wav --server ws://192.168.1.100:8765
    uv run python test_client.py --health                # check server health
    uv run python test_client.py --info                  # get server info
    uv run python test_client.py --upload audio.wav      # one-shot transcription
"""

import argparse
import asyncio
import json
import sys
import time

import numpy as np
import soundfile as sf
import websockets
import urllib.request


def check_health(base_url: str):
    url = base_url.replace("ws://", "http://").replace("wss://", "https://")
    for endpoint in ["/healthz", "/readyz", "/v1/info", "/v1/sessions"]:
        try:
            resp = urllib.request.urlopen(f"{url}{endpoint}", timeout=5)
            data = json.loads(resp.read())
            print(f"{endpoint}: {json.dumps(data, indent=2)}")
        except Exception as e:
            print(f"{endpoint}: FAILED — {e}")
        print()


def upload_file(base_url: str, audio_path: str):
    import http.client
    import os

    url = base_url.replace("ws://", "http://").replace("wss://", "https://")
    # Use requests-style multipart, but keep it stdlib-only
    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    filename = os.path.basename(audio_path)

    with open(audio_path, "rb") as f:
        file_data = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{url}/v1/transcribe",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())
    print(json.dumps(data, indent=2))


async def stream_file(audio_path: str, server_url: str, chunk_ms: int):
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        duration = len(audio) / sr
        n = int(duration * 16000)
        x_old = np.linspace(0, duration, len(audio), endpoint=False)
        x_new = np.linspace(0, duration, n, endpoint=False)
        audio = np.interp(x_new, x_old, audio)

    pcm = (audio * 32768).clip(-32768, 32767).astype(np.int16)
    chunk_samples = int(16000 * chunk_ms / 1000)
    total = (len(pcm) + chunk_samples - 1) // chunk_samples

    print(f"Audio: {len(pcm)/16000:.1f}s, {total} chunks of {chunk_ms}ms")
    print(f"Connecting to {server_url}...\n")

    async with websockets.connect(f"{server_url}/v1/ws/transcribe") as ws:
        msg = json.loads(await ws.recv())
        assert msg["type"] == "ready"
        print("Connected. Streaming...\n")

        t0 = time.perf_counter()

        for i in range(0, len(pcm), chunk_samples):
            chunk = pcm[i : i + chunk_samples]
            await ws.send(chunk.tobytes())
            await asyncio.sleep(chunk_ms / 1000)

            try:
                while True:
                    resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.05))
                    rtype = resp["type"]

                    if rtype == "partial":
                        sys.stdout.write(f"\r  partial [{resp['step']:>3}]: {resp['text']}      ")
                        sys.stdout.flush()
                    elif rtype == "final":
                        elapsed = time.perf_counter() - t0
                        print(f"\n  ✓ FINAL [{resp['step']:>3}]: {resp['text']}")
                        print(f"    turn_prob={resp['turn_probability']:.3f}  elapsed={elapsed:.1f}s")
                    elif rtype == "vad_start":
                        print("\n  ▶ speech started")
                    elif rtype == "vad_end":
                        print("\n  ■ speech ended")
                    elif rtype == "turn_result":
                        print(f"    turn_prob={resp['turn_probability']:.3f}")

            except (asyncio.TimeoutError, TimeoutError):
                pass

        # Flush
        await ws.send(json.dumps({"type": "reset"}))
        await asyncio.sleep(1)
        try:
            while True:
                resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
                if resp["type"] == "final":
                    print(f"\n  ✓ FINAL (flush): {resp['text']}")
                    break
        except (asyncio.TimeoutError, TimeoutError):
            pass

        elapsed = time.perf_counter() - t0
        print(f"\nDone in {elapsed:.1f}s")


def main():
    p = argparse.ArgumentParser(description="ASR test client")
    p.add_argument("audio", nargs="?", help="WAV file to stream")
    p.add_argument("--server", default="ws://localhost:8765")
    p.add_argument("--chunk-ms", type=int, default=560)
    p.add_argument("--health", action="store_true", help="Check server health")
    p.add_argument("--info", action="store_true", help="Get server info")
    p.add_argument("--upload", metavar="FILE", help="One-shot file transcription")
    args = p.parse_args()

    if args.health or args.info:
        check_health(args.server)
    elif args.upload:
        upload_file(args.server, args.upload)
    elif args.audio:
        asyncio.run(stream_file(args.audio, args.server, args.chunk_ms))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
