"""Interactive localhost demo: type a command, watch the fine-tuned SmolVLA drive
the SO-101 arm in MuJoCo, live.

Loads the fine-tuned policy once, then serves a page with a text box (and quick
chips for the trained commands). Submitting a command resets the scene, feeds the
policy ONLY the camera image + joint state + your text, and streams the rollout as
MJPEG so you can watch it execute.

Run:  python3 -m tinyvla.demo               # then open http://localhost:8009
      python3 -m tinyvla.demo --device cuda # on a GPU box
"""
from __future__ import annotations

import argparse
import io
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch
import mujoco
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from .task import SO101PickPlaceTask, COMMANDS, COLORS
from .collect import IMG, CAMERAS
from .paths import CHECKPOINTS_ROOT, DATASETS_ROOT
from .runtime import load_runtime, verify_compact_vocabulary

PORT = 8009

# ---------------------------------------------------------------------------
# map free text -> a known command (for cube placement + a success verdict)
# ---------------------------------------------------------------------------
def command_index_from_text(text: str) -> int:
    t = text.lower()
    red_i, blue_i = t.find("red"), t.find("blue")
    has_red, has_blue = red_i >= 0, blue_i >= 0

    def first_color():
        if has_red and (not has_blue or red_i < blue_i):
            return "red"
        return "blue" if has_blue else "red"

    if "top" in t or "stack" in t:                       # stacking
        c = first_color()
        return 4 if c == "red" else 5
    if "box" in t and "plate" in t and has_red and has_blue:   # sorting
        box_i = t.find("box")
        box_color = "red" if abs(red_i - box_i) <= abs(blue_i - box_i) else "blue"
        return 6 if box_color == "red" else 7
    if "plate" in t:                                     # place on plate
        return 2 if first_color() == "red" else 3
    return 0 if first_color() == "red" else 1            # default: place in box


class Demo:
    def __init__(self, model, root, repo_id, device):
        self.device = torch.device(device)
        self.model_path = model
        meta = LeRobotDatasetMetadata(repo_id, root=root)
        runtime = load_runtime(
            model,
            meta=meta,
            dataset_root=root,
            device=self.device,
            stats_source="checkpoint",
        )
        self.policy = runtime.policy.eval()
        self.pre, self.post = runtime.preprocessor, runtime.postprocessor
        self.delta_actions = runtime.delta_actions
        # shared state
        self._jpeg = None
        self._lock = threading.Lock()
        self._pending = None
        self._cond = threading.Condition()
        self.status = {"phase": "idle", "instruction": "", "result": ""}
        threading.Thread(target=self._worker, daemon=True).start()

    # -- public: queue a command -----------------------------------------
    def submit(self, text: str):
        try:
            verify_compact_vocabulary(self.policy, self.model_path, [text])
        except RuntimeError as error:
            self.status = {"phase": "rejected", "instruction": text, "result": str(error)}
            return
        with self._cond:
            self._pending = text
            self.status = {"phase": "queued", "instruction": text, "result": ""}
            self._cond.notify_all()

    def latest_jpeg(self):
        with self._lock:
            return self._jpeg

    # -- worker thread: owns the env, renderers, and rollout -------------
    def _worker(self):
        env = SO101PickPlaceTask(seed=7)
        obs_r = mujoco.Renderer(env.model, height=IMG, width=IMG)
        disp = mujoco.Renderer(env.model, height=480, width=640)

        def publish(e):
            disp.update_scene(e.data, camera="front")
            buf = io.BytesIO()
            Image.fromarray(disp.render()).save(buf, format="JPEG", quality=80)
            with self._lock:
                self._jpeg = buf.getvalue()

        env.reset(command=0)
        publish(env)

        while True:
            with self._cond:
                while self._pending is None:
                    self._cond.wait()
                text = self._pending
                self._pending = None

            idx = command_index_from_text(text)
            env.reset(command=idx)
            env.instruction = text                       # policy sees YOUR words
            self.policy.reset()
            self.status = {"phase": "running", "instruction": text, "result": ""}
            horizon = 130 * len(env.steps)
            for t in range(horizon):
                state = torch.from_numpy(env.data.qpos[:6].copy().astype(np.float32))
                raw = {"observation.state": state.unsqueeze(0).to(self.device), "task": [text]}
                for cam in CAMERAS:
                    obs_r.update_scene(env.data, camera=cam)
                    im = torch.from_numpy(obs_r.render()).permute(2, 0, 1).float() / 255.0
                    raw[f"observation.images.{cam}"] = im.unsqueeze(0).to(self.device)
                batch = self.pre(raw)
                with torch.inference_mode():
                    action = self.policy.select_action(batch)
                action = self.post(action).squeeze(0).cpu().numpy()
                if self.delta_actions:
                    action = action + env.data.qpos[:6].astype(action.dtype)
                env.step(action)
                publish(env)
                if self._pending is not None:            # new command interrupts
                    break
            ok = env.success()
            self.status = {"phase": "done", "instruction": text,
                           "result": "success" if ok else "missed"}
            publish(env)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SmolVLA · SO-101 live</title><style>
  :root{--bg:#fff;--text:#202124;--muted:#5f6368;--line:#dadce0;--accent:#1a73e8;--soft:#e8f0fe}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);
    font-family:"Google Sans",Roboto,-apple-system,system-ui,sans-serif}
  .wrap{max-width:760px;margin:0 auto;padding:36px 24px 60px}
  h1{font-size:22px;font-weight:500;margin:0 0 4px}
  .sub{color:var(--muted);font-size:13.5px;margin:0 0 22px}
  .stage{position:relative;border:1px solid var(--line);border-radius:14px;overflow:hidden;background:#f1f3f4}
  .stage img{display:block;width:100%;aspect-ratio:4/3}
  .badge{position:absolute;top:12px;left:12px;padding:5px 12px;border-radius:999px;
    font-size:12.5px;font-weight:500;background:#fff;border:1px solid var(--line)}
  .row{display:flex;gap:8px;margin-top:16px}
  input{flex:1;padding:12px 14px;border:1px solid var(--line);border-radius:10px;font-size:15px}
  button{padding:12px 20px;border:0;border-radius:10px;background:var(--accent);color:#fff;
    font-size:15px;font-weight:500;cursor:pointer}
  .chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
  .chip{padding:7px 12px;border:1px solid var(--line);border-radius:999px;font-size:12.5px;
    color:var(--text);background:#fff;cursor:pointer}
  .chip:hover{background:var(--soft);border-color:var(--accent);color:var(--accent)}
</style></head><body><div class="wrap">
  <h1>SmolVLA &middot; SO-101 live</h1>
  <p class="sub">Type a command (or tap one below). The fine-tuned policy sees only the
    camera, the joint angles, and your text &mdash; then drives the arm.</p>
  <div class="stage"><span class="badge" id="badge">idle</span>
    <img src="/stream.mjpg"/></div>
  <div class="row">
    <input id="cmd" placeholder="e.g. put the blue cube on the plate" autocomplete="off"/>
    <button onclick="run()">Run</button>
  </div>
  <div class="chips" id="chips"></div>
</div><script>
  const CMDS = __CMDS__;
  const chips = document.getElementById('chips');
  CMDS.forEach(c => { const b=document.createElement('div'); b.className='chip'; b.textContent=c;
    b.onclick=()=>{document.getElementById('cmd').value=c; run();}; chips.appendChild(b); });
  function run(){ const v=document.getElementById('cmd').value.trim(); if(!v)return;
    fetch('/run?cmd='+encodeURIComponent(v)); }
  document.getElementById('cmd').addEventListener('keydown',e=>{if(e.key==='Enter')run();});
  async function poll(){ try{ const s=await (await fetch('/status')).json();
    const b=document.getElementById('badge');
    b.textContent = s.phase==='running' ? '⟳ '+s.instruction
      : s.phase==='done' ? (s.result==='success'?'✓ ':'✗ ')+s.instruction
      : s.phase==='queued' ? '… '+s.instruction : 'idle';
    b.style.color = s.result==='success'?'#188038':(s.result==='missed'?'#c5221f':'#202124');
  }catch(e){} setTimeout(poll,500); } poll();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(CHECKPOINTS_ROOT / "smolvla_pickplace"))
    ap.add_argument("--root", default=str(DATASETS_ROOT / "so101_pickplace"))
    ap.add_argument("--repo-id", default="local/so101_pickplace")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    print("loading fine-tuned SmolVLA (~30s)...")
    demo = Demo(args.model, args.root, args.repo_id, args.device)
    print("ready.")
    import json
    page = PAGE.replace("__CMDS__", json.dumps([c["instruction"] for c in COMMANDS])).encode()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(page, "text/html")
            elif path == "/run":
                q = parse_qs(urlparse(self.path).query)
                demo.submit(q.get("cmd", [""])[0])
                self._send(b"ok", "text/plain")
            elif path == "/status":
                import json as _j
                self._send(_j.dumps(demo.status).encode(), "application/json")
            elif path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        frame = demo.latest_jpeg()
                        if frame:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                            self.wfile.write(frame + b"\r\n")
                        time.sleep(1 / 30)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    print(f"SmolVLA live demo at http://localhost:{args.port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
