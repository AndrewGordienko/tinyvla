"""Browse the collected SO-101 reach dataset in the browser.

A clean, light grid of looping sample videos, each titled with its language
prompt. Everything is decoded and encoded once at startup, then served static.

Run:  python3 -m tinyvla.dataset_viewer    # open http://localhost:8001
"""
from __future__ import annotations

import argparse
import base64
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .paths import DATASETS_ROOT
from .task import COMMANDS

PORT = 8001
PER_ROW = 4              # example clips per command row
STRIDE = 2              # use every Nth frame in the clip (keeps webp small)


def img_from_sample(sample):
    a = sample["observation.images.front"]           # (3,H,W) float [0,1]
    return (a.numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)


def animated_webp_b64(frames, fps):
    """frames: list of HWC uint8 -> looping animated WebP data-URI."""
    imgs = [Image.fromarray(f) for f in frames]
    buf = io.BytesIO()
    imgs[0].save(buf, format="WEBP", save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps * STRIDE), loop=0, quality=70, method=4)
    return base64.b64encode(buf.getvalue()).decode()


def _dot(task):
    r, b = "red" in task, "blue" in task
    if r and b:
        return "#9c27b0"          # command mentions both cubes (sorting)
    if r:
        return "#ea4335"
    if b:
        return "#4285f4"
    return "#9aa0a6"


def build_page(root, repo_id):
    ds = LeRobotDataset(repo_id, root=root)
    hf = ds.hf_dataset
    ep_idx = np.array(hf["episode_index"])
    fps = ds.fps

    # group episode indices by their instruction (task string)
    by_task = {}
    for e in range(ds.num_episodes):
        g0 = int(np.where(ep_idx == e)[0][0])
        task = ds[g0].get("task", "")
        by_task.setdefault(task, []).append(e)

    # canonical row order = the command set the task supports; then any extras
    order = [spec["instruction"] for spec in COMMANDS]
    tasks_sorted = [t for t in order if t in by_task] + \
                   [t for t in by_task if t not in order]

    def clip_for(e):
        frames_g = np.where(ep_idx == e)[0]
        frames = [img_from_sample(ds[int(gi)]) for gi in frames_g[::STRIDE]]
        return animated_webp_b64(frames, fps)

    rows = []
    for task in tasks_sorted:
        eps = by_task[task][:PER_ROW]
        tiles = "".join(
            f'<figure class="tile"><img src="data:image/webp;base64,{clip_for(e)}" '
            f'alt="ep {e}"/><figcaption>episode {e}</figcaption></figure>' for e in eps)
        rows.append(f"""
        <section class="cmdrow">
          <div class="cmd"><i class="dot" style="background:{_dot(task)}"></i>
            <span class="cmdtext">{task}</span>
            <span class="count">{len(by_task[task])} episodes</span></div>
          <div class="row">{tiles}</div>
        </section>""")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SO-101 manipulation dataset</title><style>
  :root{{
    --bg:#ffffff; --surface:#ffffff; --text:#202124; --muted:#5f6368;
    --line:#dadce0; --accent:#1a73e8; --accent-soft:#e8f0fe;
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--text);
    font-family:"Google Sans","Product Sans",Roboto,-apple-system,system-ui,sans-serif;
    -webkit-font-smoothing:antialiased}}
  header{{padding:40px 48px 8px;max-width:1200px;margin:0 auto}}
  h1{{font-size:26px;font-weight:500;letter-spacing:-.4px;margin:0 0 6px}}
  .lead{{color:var(--muted);font-size:14px;line-height:1.5;margin:0}}
  .stats{{display:flex;gap:10px;margin:18px 0 4px;flex-wrap:wrap}}
  .pill{{background:var(--accent-soft);color:var(--accent);font-size:12.5px;
    font-weight:500;padding:5px 12px;border-radius:999px}}
  main{{max-width:1200px;margin:0 auto;padding:12px 48px 64px}}
  .cmdrow{{padding:22px 0;border-top:1px solid var(--line)}}
  .cmd{{display:flex;align-items:center;gap:9px;margin-bottom:14px}}
  .cmdtext{{font-size:17px;font-weight:500}}
  .count{{font-size:12px;color:var(--muted);margin-left:6px}}
  .dot{{width:11px;height:11px;border-radius:50%;display:inline-block;flex:0 0 auto}}
  .row{{display:grid;grid-template-columns:repeat({PER_ROW},1fr);gap:16px}}
  .tile{{margin:0;background:var(--surface);border:1px solid var(--line);
    border-radius:12px;overflow:hidden;transition:box-shadow .18s,transform .18s}}
  .tile:hover{{box-shadow:0 1px 2px rgba(60,64,67,.15),0 6px 20px rgba(60,64,67,.15);
    transform:translateY(-2px)}}
  .tile img{{display:block;width:100%;aspect-ratio:1/1;object-fit:cover;background:#f1f3f4}}
  .tile figcaption{{padding:9px 12px;font-size:12px;color:var(--muted)}}
</style></head><body>
  <header>
    <h1>SO-101 manipulation &mdash; scripted-expert demonstrations</h1>
    <p class="lead">Each row is one language command SmolVLA is trained on, with four
      example episodes (the camera view the policy sees). Two cubes spawn at random spots;
      the instruction decides what to do &mdash; which cube, and whether to drop it in the
      bin or stack it &mdash; so the policy has to read the words. Task set mirrors
      SmolVLA&rsquo;s SO-100 datasets (pick-place, stacking).</p>
    <div class="stats">
      <span class="pill">{len(tasks_sorted)} commands</span>
      <span class="pill">{ds.num_episodes} episodes</span>
      <span class="pill">{ds.num_frames} frames</span>
      <span class="pill">{fps} fps</span>
      <span class="pill">256&times;256 RGB</span>
    </div>
  </header>
  <main>{''.join(rows)}</main>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATASETS_ROOT / "so101_pickplace"))
    ap.add_argument("--repo-id", default="local/so101_pickplace")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    print("encoding sample clips...")
    page = build_page(args.root, args.repo_id).encode()
    print(f"ready: {len(page)//1024} KB page")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    print(f"SO-101 dataset viewer at http://localhost:{args.port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
