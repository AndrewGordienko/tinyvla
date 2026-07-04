"""Serve the SO-101 MuJoCo arm as a live MJPEG stream on localhost.

Run:
    python3 -m tinyvla.server    # then open http://localhost:8000

All OpenGL rendering happens in a single background thread; HTTP handlers just
serve the latest encoded JPEG, which keeps the MuJoCo GL context happy.
"""
from __future__ import annotations

import io
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import mujoco
from PIL import Image

from .env import SO101Env, JOINT_NAMES

PORT = 8000
FPS = 30

# ---------------------------------------------------------------------------
# Background sim + render thread -> shared latest JPEG
# ---------------------------------------------------------------------------
_latest_jpeg: bytes | None = None
_cond = threading.Condition()


def _sim_loop():
    global _latest_jpeg
    env = SO101Env()
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = 160, -20, 0.9
    cam.lookat[:] = [0, 0, 0.1]

    lo, hi = env.ctrl_range[:, 0], env.ctrl_range[:, 1]
    mid = 0.5 * (lo + hi)
    amp = 0.35 * (hi - lo)
    t0 = time.time()
    frame_dt = 1.0 / FPS

    while True:
        t = time.time() - t0
        # gentle per-joint sinusoid so the arm visibly moves
        target = mid + amp * np.sin(0.5 * t + np.arange(env.nu) * 0.7)
        env.step(target)

        renderer.update_scene(env.data, cam)
        img = renderer.render()
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=80)
        with _cond:
            _latest_jpeg = buf.getvalue()
            _cond.notify_all()
        time.sleep(frame_dt)


PAGE = f"""<!doctype html>
<html><head><title>SO-101 arm (SmolVLA)</title>
<style>
  body{{background:#111;color:#ddd;font-family:system-ui,sans-serif;text-align:center;margin:0;padding:24px}}
  h1{{font-weight:600;font-size:18px}}
  img{{border-radius:8px;box-shadow:0 4px 24px rgba(0,0,0,.5);max-width:100%}}
  .j{{color:#888;font-size:13px;margin-top:12px}}
</style></head>
<body>
  <h1>SO-101 &mdash; the SmolVLA / LeRobot arm</h1>
  <img src="/stream.mjpg" width="640" height="480"/>
  <div class="j">joints: {', '.join(JOINT_NAMES)}</div>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with _cond:
                        _cond.wait(timeout=5)
                        frame = _latest_jpeg
                    if frame is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def main() -> None:
    threading.Thread(target=_sim_loop, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"SO-101 live stream at http://localhost:{PORT}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
