"""Live localhost demo with a switchable PolicyAdapter and closed-loop telemetry.

Unlike `tinyvla.demo` (a minimal single-policy viewer) this serves the interview
deliverable: a live Mac/GLFW closed-loop rollout of the ~291.6M student, with a
telemetry panel that reports — from the EXACT loaded checkpoint — the parameter
count, checkpoint SHA, per-step inference latency, the SmolVLA action-chunk queue,
and the physical task stage (approach/grasp/transport/release). A toggle swaps the
student for the 450M teacher so you can compare them live on identical scenes.

Nothing here is a recorded replay: every frame is produced by running the loaded
policy on freshly rendered pixels + joint state.

Honesty note surfaced in the UI: object grasping is currently KINEMATIC — the task
attaches the nearest cube within a 4 cm radius when the gripper closes and carries
it rigidly (see tinyvla/task.py). The arm trajectory is learned; the grasp itself
is simulation scaffolding, not contact-valid physics.

Run:  .venv/bin/python -m tinyvla.live_demo            # http://localhost:8010
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch
import mujoco
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.modeling_smolvla import ACTION

from .task import SO101PickPlaceTask, COMMANDS
from .collect import IMG
from .paths import CHECKPOINTS_ROOT, DATASETS_ROOT, DATA_ROOT
from .runtime import load_runtime, verify_compact_vocabulary, sha256_tree

PORT = 8010
DISPLAY_CAMS = ("front", "wrist")   # rendered for the UI regardless of policy input


def command_index_from_text(text: str) -> int:
    """Map free text -> a known command (for cube placement + a success verdict)."""
    t = text.lower()
    red_i, blue_i = t.find("red"), t.find("blue")
    has_red, has_blue = red_i >= 0, blue_i >= 0

    def first_color():
        if has_red and (not has_blue or red_i < blue_i):
            return "red"
        return "blue" if has_blue else "red"

    if "top" in t or "stack" in t:
        return 4 if first_color() == "red" else 5
    if "box" in t and "plate" in t and has_red and has_blue:
        box_i = t.find("box")
        box_color = "red" if abs(red_i - box_i) <= abs(blue_i - box_i) else "blue"
        return 6 if box_color == "red" else 7
    if "plate" in t:
        return 2 if first_color() == "red" else 3
    return 0 if first_color() == "red" else 1


class LoadedPolicy:
    """One policy loaded from an exact checkpoint, with its provenance telemetry."""

    def __init__(self, name: str, path: str, meta, root: str, device):
        self.name = name
        self.path = str(path)
        rt = load_runtime(path, meta=meta, dataset_root=root, device=device,
                          stats_source="checkpoint")
        self.policy = rt.policy.eval()
        self.pre, self.post = rt.preprocessor, rt.postprocessor
        self.delta_actions = rt.delta_actions
        # provenance, counted from the EXACT loaded checkpoint
        self.param_count = int(sum(p.numel() for p in self.policy.parameters()))
        self.sha = sha256_tree(path) or "unknown"
        cams = list(getattr(getattr(self.policy, "config", None), "image_features", {}) or {})
        self.cameras = [c.removeprefix("observation.images.") for c in cams] or ["front"]
        self.chunk_size = int(getattr(self.policy.config, "n_action_steps", 1))

    def queue_len(self) -> int:
        q = getattr(self.policy, "_queues", {}).get(ACTION)
        return len(q) if q is not None else 0


class PolicyAdapter:
    """Holds several named policies over one embodiment; switches the active one."""

    def __init__(self, specs: dict[str, str], root, repo_id, device):
        self.device = torch.device(device)
        meta = LeRobotDatasetMetadata(repo_id, root=root)
        self.policies: dict[str, LoadedPolicy] = {}
        for name, path in specs.items():
            print(f"loading {name} <- {path}")
            self.policies[name] = LoadedPolicy(name, path, meta, root, self.device)
        self.active = next(iter(self.policies))

    def get(self, name: str | None = None) -> LoadedPolicy:
        return self.policies[name or self.active]

    def names(self):
        return list(self.policies)


class LiveDemo:
    def __init__(self, adapter: PolicyAdapter):
        self.adapter = adapter
        self.device = adapter.device
        self._jpeg = {c: None for c in DISPLAY_CAMS}
        self._lock = threading.Lock()
        self._pending = None
        self._switch_to = None
        self._cond = threading.Condition()
        self.tele = {
            "phase": "idle", "instruction": "", "result": "",
            "active_policy": adapter.active, "step": 0, "horizon": 0,
            "latency_ms": 0.0, "chunk_latency_ms": 0.0,
            "queue_len": 0, "chunk_size": adapter.get().chunk_size,
            "stage": "-", "grasped": None, "action": [0.0] * 6,
            "policies": {n: {"params": p.param_count, "sha": p.sha[:12],
                             "cameras": p.cameras}
                         for n, p in adapter.policies.items()},
        }
        threading.Thread(target=self._worker, daemon=True).start()

    # -- public API ------------------------------------------------------
    def submit(self, text: str):
        lp = self.adapter.get()
        # Feed the raw words if the compact-vocab student can tokenize them; else fall
        # back to the canonical instruction of the nearest command (the student was only
        # trained on those 8 strings — honest, not a silent rewrite of a valid request).
        effective, note = text, ""
        try:
            verify_compact_vocabulary(lp.policy, lp.path, [text])
        except RuntimeError:
            effective = COMMANDS[command_index_from_text(text)]["instruction"]
            note = f'mapped to trained instruction: "{effective}"'
            try:
                verify_compact_vocabulary(lp.policy, lp.path, [effective])
            except RuntimeError as error:
                self.tele.update(phase="rejected", instruction=text, result=str(error))
                return
        with self._cond:
            self._pending = effective
            self.tele.update(phase="queued", instruction=effective, result=note)
            self._cond.notify_all()

    def switch(self, name: str):
        if name in self.adapter.policies:
            with self._cond:
                self._switch_to = name
                self._cond.notify_all()

    def latest_jpeg(self, cam):
        with self._lock:
            return self._jpeg.get(cam)

    # -- live physical stage (mirrors eval_closedloop semantics) ---------
    @staticmethod
    def _stage(env) -> str:
        if bool(env.success()):
            return "release"
        if env.grasped is not None:
            return "transport" if float(env.ee_pos()[2]) > 0.15 else "grasp"
        active = env.active_subtask()[0]
        dist = float(np.linalg.norm(env.ee_pos() - env.cube_pos(active)))
        return "approach" if dist <= 0.05 else "reach"

    # -- worker thread ---------------------------------------------------
    def _worker(self):
        env = SO101PickPlaceTask(seed=7)
        obs_r = mujoco.Renderer(env.model, height=IMG, width=IMG)
        disp = {c: mujoco.Renderer(env.model, height=360, width=480) for c in DISPLAY_CAMS}

        def publish(e):
            for c in DISPLAY_CAMS:
                disp[c].update_scene(e.data, camera=c)
                buf = io.BytesIO()
                Image.fromarray(disp[c].render()).save(buf, format="JPEG", quality=80)
                with self._lock:
                    self._jpeg[c] = buf.getvalue()

        env.reset(command=0)
        publish(env)
        lat_ema = 0.0

        while True:
            with self._cond:
                while self._pending is None and self._switch_to is None:
                    self._cond.wait()
                if self._switch_to is not None:
                    self.adapter.active = self._switch_to
                    self._switch_to = None
                    self.tele.update(active_policy=self.adapter.active,
                                     chunk_size=self.adapter.get().chunk_size)
                    if self._pending is None:
                        continue
                text = self._pending
                self._pending = None

            lp = self.adapter.get()
            idx = command_index_from_text(text)
            env.reset(command=idx)
            env.instruction = text
            lp.policy.reset()
            horizon = 130 * len(env.steps)
            self.tele.update(phase="running", instruction=text, result="",
                             active_policy=lp.name, horizon=horizon,
                             chunk_size=lp.chunk_size)

            for step in range(horizon):
                state = torch.from_numpy(env.data.qpos[:6].copy().astype(np.float32))
                raw = {"observation.state": state.unsqueeze(0).to(self.device), "task": [text]}
                for cam in lp.cameras:                       # policy input = its own cams
                    obs_r.update_scene(env.data, camera=cam)
                    im = torch.from_numpy(obs_r.render()).permute(2, 0, 1).float() / 255.0
                    raw[f"observation.images.{cam}"] = im.unsqueeze(0).to(self.device)
                batch = lp.pre(raw)

                q_before = lp.queue_len()                    # 0 => this call regenerates a chunk
                t0 = time.perf_counter()
                with torch.inference_mode():
                    action = lp.policy.select_action(batch)
                dt = (time.perf_counter() - t0) * 1e3
                lat_ema = dt if lat_ema == 0 else 0.8 * lat_ema + 0.2 * dt

                action = lp.post(action).squeeze(0).cpu().numpy()
                if lp.delta_actions:
                    action = action + env.data.qpos[:6].astype(action.dtype)
                env.step(action)
                publish(env)

                self.tele.update(
                    step=step + 1, latency_ms=round(lat_ema, 1),
                    chunk_latency_ms=round(dt, 1) if q_before == 0 else self.tele["chunk_latency_ms"],
                    queue_len=lp.queue_len(), stage=self._stage(env),
                    grasped=env.grasped, action=[round(float(a), 3) for a in action])

                if self._pending is not None or self._switch_to is not None:
                    break

            ok = bool(env.success())
            self.tele.update(phase="done", result="success" if ok else "missed")
            publish(env)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SmolVLA · SO-101 live (student)</title><style>
  :root{--bg:#0f1115;--panel:#171a21;--text:#e6e8eb;--muted:#9aa0a8;--line:#262b34;
        --accent:#4c8dff;--ok:#39d98a;--bad:#ff6b6b;--warn:#ffc234}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);
    font-family:"SF Pro Text",Roboto,-apple-system,system-ui,sans-serif}
  .wrap{max-width:1080px;margin:0 auto;padding:24px 20px 60px}
  h1{font-size:20px;font-weight:600;margin:0 0 2px}
  .sub{color:var(--muted);font-size:13px;margin:0 0 16px}
  .disc{background:#2a2410;border:1px solid #5a4a12;color:var(--warn);font-size:12.5px;
    padding:9px 12px;border-radius:9px;margin:0 0 16px}
  .grid{display:grid;grid-template-columns:1.4fr 1fr;gap:16px}
  @media(max-width:820px){.grid{grid-template-columns:1fr}}
  .cams{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .cam{position:relative;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#000}
  .cam img{display:block;width:100%;aspect-ratio:4/3}
  .cam .tag{position:absolute;top:8px;left:8px;font-size:11px;color:var(--muted);
    background:rgba(0,0,0,.55);padding:2px 8px;border-radius:999px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
  .toggle{display:flex;gap:6px;margin-bottom:12px}
  .toggle button{flex:1;padding:8px;border:1px solid var(--line);border-radius:8px;background:#12151b;
    color:var(--muted);font-size:13px;font-weight:600;cursor:pointer}
  .toggle button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
  .kv{display:flex;justify-content:space-between;font-size:13px;padding:5px 0;border-bottom:1px solid var(--line)}
  .kv .k{color:var(--muted)} .kv .v{font-variant-numeric:tabular-nums;font-family:ui-monospace,Menlo,monospace}
  .stage{display:flex;gap:4px;margin:10px 0 4px}
  .stage span{flex:1;text-align:center;font-size:11px;padding:5px 2px;border-radius:6px;
    background:#12151b;color:var(--muted);border:1px solid var(--line)}
  .stage span.on{background:var(--accent);color:#fff;border-color:var(--accent)}
  .queue{display:flex;gap:3px;margin-top:6px;flex-wrap:wrap}
  .queue i{width:12px;height:16px;border-radius:3px;background:#2a3240}
  .queue i.f{background:var(--accent)}
  .bars{margin-top:8px}
  .bar{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--muted);margin:3px 0}
  .bar .t{width:52px} .bar .track{flex:1;height:8px;background:#12151b;border-radius:4px;position:relative}
  .bar .fill{position:absolute;top:0;bottom:0;left:50%;background:var(--accent);border-radius:4px}
  .row{display:flex;gap:8px;margin-top:14px}
  input{flex:1;padding:11px 13px;border:1px solid var(--line);border-radius:9px;font-size:14px;
    background:#12151b;color:var(--text)}
  button.run{padding:11px 18px;border:0;border-radius:9px;background:var(--accent);color:#fff;
    font-size:14px;font-weight:600;cursor:pointer}
  .chips{display:flex;flex-wrap:wrap;gap:7px;margin-top:12px}
  .chip{padding:6px 11px;border:1px solid var(--line);border-radius:999px;font-size:12px;
    color:var(--text);background:#12151b;cursor:pointer}
  .chip:hover{border-color:var(--accent);color:var(--accent)}
  .badge{font-size:12px;padding:3px 10px;border-radius:999px;border:1px solid var(--line)}
</style></head><body><div class="wrap">
  <h1>SmolVLA &middot; SO-101 &mdash; live student</h1>
  <p class="sub">Live Mac/GLFW closed-loop inference from the exact loaded checkpoint. Not a replay.</p>
  <div class="disc"><b>Honest disclosure:</b> object grasping is currently KINEMATIC scaffolding &mdash;
    the task rigidly attaches the nearest cube within 4&nbsp;cm when the gripper closes and carries it
    (see task.py). The arm trajectory is learned; the grasp is not contact-valid physics.</div>
  <div class="grid">
    <div>
      <div class="cams">
        <div class="cam"><span class="tag">front</span><img src="/stream_front.mjpg"/></div>
        <div class="cam"><span class="tag">wrist</span><img src="/stream_wrist.mjpg"/></div>
      </div>
      <div class="row">
        <input id="cmd" placeholder="e.g. put the blue cube on the plate" autocomplete="off"/>
        <button class="run" onclick="run()">Run</button>
      </div>
      <div class="chips" id="chips"></div>
    </div>
    <div class="panel">
      <div class="toggle" id="toggle"></div>
      <div class="kv"><span class="k">status</span><span class="v badge" id="st">idle</span></div>
      <div class="kv"><span class="k">parameters</span><span class="v" id="params">–</span></div>
      <div class="kv"><span class="k">checkpoint SHA</span><span class="v" id="sha">–</span></div>
      <div class="kv"><span class="k">inference latency</span><span class="v" id="lat">– ms</span></div>
      <div class="kv"><span class="k">chunk gen latency</span><span class="v" id="clat">– ms</span></div>
      <div class="kv"><span class="k">step</span><span class="v" id="step">0 / 0</span></div>
      <div style="font-size:11px;color:var(--muted);margin:12px 0 2px">TASK STAGE</div>
      <div class="stage" id="stage"></div>
      <div style="font-size:11px;color:var(--muted);margin:12px 0 2px">ACTION QUEUE (<span id="qc">0</span>/<span id="qs">0</span>)</div>
      <div class="queue" id="queue"></div>
      <div style="font-size:11px;color:var(--muted);margin:12px 0 2px">ACTION (6 joints, centered)</div>
      <div class="bars" id="bars"></div>
    </div>
  </div>
</div><script>
  const CMDS = __CMDS__;
  const STAGES = ["reach","approach","grasp","transport","release"];
  const chips = document.getElementById('chips');
  CMDS.forEach(c => { const b=document.createElement('div'); b.className='chip'; b.textContent=c;
    b.onclick=()=>{document.getElementById('cmd').value=c; run();}; chips.appendChild(b); });
  document.getElementById('stage').innerHTML = STAGES.map(s=>`<span data-s="${s}">${s}</span>`).join('');
  for(let i=0;i<6;i++){document.getElementById('bars').insertAdjacentHTML('beforeend',
    `<div class="bar"><span class="t">j${i}</span><div class="track"><div class="fill" id="f${i}"></div></div></div>`);}
  function run(){ const v=document.getElementById('cmd').value.trim(); if(!v)return;
    fetch('/run?cmd='+encodeURIComponent(v)); }
  document.getElementById('cmd').addEventListener('keydown',e=>{if(e.key==='Enter')run();});
  function setToggle(names,active){ const t=document.getElementById('toggle');
    if(t.dataset.built) { [...t.children].forEach(b=>b.classList.toggle('on',b.textContent===active)); return; }
    t.dataset.built=1; names.forEach(n=>{const b=document.createElement('button');
      b.textContent=n; b.className=n===active?'on':''; b.onclick=()=>fetch('/switch?policy='+n); t.appendChild(b);}); }
  function fmt(n){ return n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(1)+'k':n; }
  async function poll(){ try{ const s=await (await fetch('/status')).json();
    setToggle(Object.keys(s.policies), s.active_policy);
    const p=s.policies[s.active_policy]||{};
    const st=document.getElementById('st');
    st.textContent = s.phase==='running'?'⟳ running':s.phase==='done'?(s.result==='success'?'✓ success':'✗ missed'):
      s.phase==='rejected'?'⚠ rejected':s.phase;
    st.style.color = s.result==='success'?'var(--ok)':(s.result==='missed'||s.phase==='rejected')?'var(--bad)':'var(--text)';
    document.getElementById('params').textContent = fmt(p.params||0)+' ('+(p.params||0).toLocaleString()+')';
    document.getElementById('sha').textContent = p.sha||'–';
    document.getElementById('lat').textContent = (s.latency_ms||0)+' ms';
    document.getElementById('clat').textContent = (s.chunk_latency_ms||0)+' ms';
    document.getElementById('step').textContent = (s.step||0)+' / '+(s.horizon||0);
    const si = STAGES.indexOf(s.stage);
    [...document.getElementById('stage').children].forEach((el,i)=>el.classList.toggle('on', i<=si && si>=0));
    document.getElementById('qc').textContent = s.queue_len||0;
    document.getElementById('qs').textContent = s.chunk_size||0;
    const q=document.getElementById('queue'); q.innerHTML='';
    for(let i=0;i<(s.chunk_size||0);i++){q.insertAdjacentHTML('beforeend',`<i class="${i<(s.queue_len||0)?'f':''}"></i>`);}
    (s.action||[]).forEach((a,i)=>{const f=document.getElementById('f'+i); if(!f)return;
      const w=Math.min(50,Math.abs(a)*50); f.style.width=w+'%'; f.style.left=(a>=0?50:50-w)+'%';});
  }catch(e){} setTimeout(poll,300); } poll();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=str(CHECKPOINTS_ROOT / "student291_champion_bf16"))
    ap.add_argument("--teacher", default=str(DATA_ROOT / "checkpoints" / "smolvla_pickplace"))
    ap.add_argument("--root", default=str(DATASETS_ROOT / "so101_pickplace"))
    ap.add_argument("--repo-id", default="local/so101_pickplace")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-teacher", action="store_true", help="load only the student")
    args = ap.parse_args()

    specs = {"student": args.student}
    if not args.no_teacher:
        specs["teacher"] = args.teacher
    print("loading policies (~30-60s)...")
    adapter = PolicyAdapter(specs, args.root, args.repo_id, args.device)
    demo = LiveDemo(adapter)
    for n, lp in adapter.policies.items():
        print(f"  {n}: {lp.param_count:,} params  sha={lp.sha[:12]}  cams={lp.cameras}")
    print("ready.")
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
            elif path == "/switch":
                q = parse_qs(urlparse(self.path).query)
                demo.switch(q.get("policy", [""])[0])
                self._send(b"ok", "text/plain")
            elif path == "/status":
                self._send(json.dumps(demo.tele).encode(), "application/json")
            elif path in ("/stream_front.mjpg", "/stream_wrist.mjpg"):
                cam = "front" if "front" in path else "wrist"
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        frame = demo.latest_jpeg(cam)
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
    print(f"live demo at http://localhost:{args.port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
