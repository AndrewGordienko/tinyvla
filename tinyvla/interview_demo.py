"""Small localhost interview viewer for the trustworthy closed-loop evidence.

It intentionally serves the recorded champion videos and metrics.  The round-0
controller weights were not persisted by the interrupted training process, so
the page says that explicitly instead of presenting a non-reproducible live
checkpoint as a learned demo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "artifacts/truth_harness/deployable_multiview_seed0_short_2026-07-11.json"
DEFAULT_VIDEOS = ROOT / "artifacts/truth_harness/deployable_rollouts_multiview_seed0_short"
MANIFEST = ROOT / "artifacts/truth_harness/datasets/command0_4/scene_manifest.json"
PARAMS = 12_262_086


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_payload(result: Path, videos: Path) -> dict:
    raw = json.loads(result.read_text())
    scenes = json.loads(MANIFEST.read_text())["scenes"]
    # Round 0 is the current champion: it is the only checkpoint that reached 2/4.
    round0 = next((r for r in raw.get("rounds", []) if r.get("round") == 0), None)
    rows = (round0 or {}).get("per_scene", [{"success": 0, "approach": 0, "grasp": 0,
                                               "transport": 0, "release": 0} for _ in scenes])
    return {
        "model_type": "deployable temporal multi-view ResNet-18-class controller",
        "parameter_count": PARAMS,
        "inference_latency_ms": None,
        "checkpoint_sha256": None,
        "checkpoint_status": "round-0 weights were not persisted; videos and metrics are the champion evidence",
        "artifact_sha256": sha256(result),
        "command": "Pick up the red cube and place it in the box.",
        "canonical_radius_m": 0.04,
        "champion_round": 0 if round0 else None,
        "champion_result": {"success": sum(r.get("success", 0) for r in rows), "n": len(rows)},
        "scenes": [{"scene": i, "instruction": s["instruction"], "metrics": rows[i],
                    "video": f"/video/seed0_scene{i}.mp4" if (videos / f"seed0_scene{i}.mp4").exists() else None}
                   for i, s in enumerate(scenes)],
    }


HTML = """<!doctype html><meta charset=utf-8><title>tinyvla interview demo</title>
<style>body{font:16px system-ui;margin:2rem;background:#f6f7f9;color:#17202a}main{max-width:1000px;margin:auto}.card{background:white;border-radius:12px;padding:1rem;margin:1rem 0;box-shadow:0 1px 5px #ccd}video{width:100%;max-height:520px;background:#111}button,select{font:1rem;padding:.55rem;margin:.25rem}.ok{color:#087f23}.bad{color:#b42318}.muted{color:#667085}table{border-collapse:collapse;width:100%}td,th{padding:.45rem;border-bottom:1px solid #ddd;text-align:left}</style>
<main><h1>tinyvla · command-0 controller</h1><div class=card id=meta></div><div class=card>
<label>Deterministic scene <select id=scene></select></label><button id=replay>Replay</button><video id=video controls></video><div id=stage></div></div>
<div class=card><h2>Trustworthy comparison</h2><table><thead><tr><th>Controller</th><th>Result</th><th>Status</th></tr></thead><tbody>
<tr><td>Scripted expert</td><td>4/4</td><td>reference</td></tr><tr><td>Privileged MLP DAgger</td><td>4/4 across 3 seeds</td><td>privileged upper bound</td></tr><tr><td>Single-frame deployable CNN</td><td>3/12</td><td>trustworthy diagnostic</td></tr><tr><td>Temporal multi-view controller</td><td>2/4 round 0; 0/4 aggregate</td><td>current champion / regression</td></tr><tr><td>450M SmolVLA</td><td>not promoted</td><td>deferred</td></tr><tr><td>163M pruned attempt</td><td>invalid vocabulary</td><td>superseded</td></tr></tbody></table></div></main>
<script>let d;const scene=document.querySelector('#scene'),video=document.querySelector('#video'),stage=document.querySelector('#stage');fetch('/api/state').then(r=>r.json()).then(x=>{d=x;document.querySelector('#meta').innerHTML=`<b>${x.model_type}</b> · ${x.parameter_count.toLocaleString()} parameters · latency ${x.inference_latency_ms===null?'not measured (weights unavailable)':x.inference_latency_ms+' ms'}<br>Command: ${x.command}<br>Champion: ${x.champion_result.success}/${x.champion_result.n} · round ${x.champion_round}<br>Checkpoint SHA: ${x.checkpoint_sha256||'not persisted'}<br>Artifact SHA: ${x.artifact_sha256}<br><span class=muted>${x.checkpoint_status}</span>`;x.scenes.forEach(s=>{let o=document.createElement('option');o.value=s.scene;o.textContent='Scene '+s.scene;scene.append(o)});show()});function show(){let s=d.scenes[scene.value||0];video.src=s.video||'';let m=s.metrics;stage.innerHTML=`Scene ${s.scene}: <span class=${m.success?'ok':'bad'}>${m.success?'success':'failure'}</span> · approach ${m.approach?'✓':'✗'} · grasp ${m.grasp?'✓':'✗'} · transport ${m.transport?'✓':'✗'} · release ${m.release?'✓':'✗'}`};scene.onchange=show;document.querySelector('#replay').onclick=()=>{video.currentTime=0;video.play()};</script>"""


def make_handler(result: Path, videos: Path):
    payload = load_payload(result, videos)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            path = unquote(urlparse(self.path).path)
            if path == "/api/state":
                body = json.dumps(payload).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            if path.startswith("/video/"):
                file = videos / Path(path.removeprefix("/video/")).name
                if file.exists() and file.suffix == ".mp4":
                    body = file.read_bytes(); self.send_response(200); self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            body = HTML.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, fmt, *args):
            return
    return Handler


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8768)
    ap.add_argument("--result", type=Path, default=DEFAULT_RESULT); ap.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEOS)
    args = ap.parse_args(); server = ThreadingHTTPServer((args.host, args.port), make_handler(args.result, args.video_dir))
    print(f"tinyvla interview demo: http://{args.host}:{args.port}", flush=True); server.serve_forever()


if __name__ == "__main__":
    main()
