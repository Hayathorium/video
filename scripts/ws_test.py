"""Minimal WebSocket client to smoke-test the RealVideo local speech pipeline.

Connects, sends a text message, and reports the messages received (types and
counts). Useful to confirm LLM -> TTS -> lip-sync without a browser.
"""

import json
import sys
import time

import websocket


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = sys.argv[2] if len(sys.argv) > 2 else "8003"
    text = sys.argv[3] if len(sys.argv) > 3 else "Hello! Please say a short greeting."
    voice_id = sys.argv[4] if len(sys.argv) > 4 else "vivian"
    timeout = int(sys.argv[5]) if len(sys.argv) > 5 else 120

    url = f"ws://{host}:{port}/ws/1"
    ws = websocket.create_connection(url, timeout=timeout)
    print(f"connected: {url}")

    # The DiT only starts generating after an image_config message (it calls
    # lip_sync.start(), which sends the "start" signal that moves the DiT from
    # server_state 0 -> 1). The frontend sends this first with the default
    # reference image; without it, every "text"/audio "update" is ignored.
    image_path = sys.argv[6] if len(sys.argv) > 6 else "resources/cat.png"
    ws.send(json.dumps({"type": "image_config", "image_path": image_path}))
    print(f"sent image_config: {image_path}")

    msg = {
        "type": "text",
        "profile": "You are a helpful assistant.",
        "text": text,
        "voice_id": voice_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    ws.send(json.dumps(msg))
    print(f"sent: {text!r} voice={voice_id}")

    counts = {}
    first_audio_at = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Keepalive: the backend's ws_lifecheck closes the socket after 60s of
        # client silence, so send an app-level ping every ~15s.
        ws.settimeout(15)
        try:
            data = ws.recv()
        except websocket.WebSocketTimeoutException:
            try:
                ws.send(json.dumps({"type": "ping", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}))
            except Exception:
                break
            continue
        except Exception as e:
            print(f"recv ended: {type(e).__name__}: {e}")
            break
        try:
            obj = json.loads(data)
        except Exception:
            obj = {"type": "raw", "len": len(data)}
        t = obj.get("type", "?")
        counts[t] = counts.get(t, 0) + 1
        if t == "audio_image" and first_audio_at is None:
            first_audio_at = time.time()
            print(f"first audio_image frame received")
        if t in ("text", "error", "processing_status"):
            print(f"[{t}] {json.dumps(obj, ensure_ascii=False)[:200]}")

    ws.close()
    print("counts:", json.dumps(counts))
    if counts.get("audio_image", 0) > 0:
        print("RESULT: OK — lip-synced audio/video frames produced")
    else:
        print("RESULT: NO audio_image frames produced")


if __name__ == "__main__":
    main()
