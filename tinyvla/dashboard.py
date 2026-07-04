"""Serve a local benchmark dashboard.

Run:
    python3 -m tinyvla.dashboard --port 8767
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .paths import ARTIFACTS_ROOT

PORT = 8765
BENCHMARKS_DIR = ARTIFACTS_ROOT / "benchmarks"


def _read_benchmark(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    data["_file"] = path.name
    data["_mtime"] = path.stat().st_mtime
    return data


def load_benchmarks() -> list[dict]:
    if not BENCHMARKS_DIR.exists():
        return []
    return sorted(
        (_read_benchmark(path) for path in BENCHMARKS_DIR.glob("*.json")),
        key=lambda row: row["_mtime"],
        reverse=True,
    )


PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>tinyvla benchmarks</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --ink: #1f2933;
      --muted: #697586;
      --line: #d6dbe3;
      --panel: #ffffff;
      --accent: #087f8c;
      --accent-2: #c2410c;
      --good: #1f7a4d;
      --warn: #8a5a00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header { border-bottom: 1px solid var(--line); background: #fff; }
    .wrap { width: min(1180px, calc(100vw - 32px)); margin: 0 auto; }
    .top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 0;
    }
    h1 { margin: 0; font-size: 20px; font-weight: 650; letter-spacing: 0; }
    .sub { margin-top: 2px; color: var(--muted); font-size: 13px; }
    button {
      appearance: none;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); color: var(--accent); }
    main { padding: 20px 0 36px; }
    .stats, .help {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .stat, .help-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .stat { min-height: 74px; }
    .stat label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .stat strong { font-size: 22px; font-weight: 650; letter-spacing: 0; }
    .help-card strong { display: block; margin-bottom: 4px; font-size: 13px; }
    .help-card span { color: var(--muted); font-size: 12px; }
    .run-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      margin-bottom: 14px;
      overflow: hidden;
    }
    .run-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }
    .run-title { font-weight: 650; }
    .run-meta { margin-top: 3px; color: var(--muted); font-size: 12px; }
    .badge {
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      padding: 3px 8px;
      font-size: 12px;
      white-space: nowrap;
    }
    .badge.compare { border-color: #b7d9dd; color: var(--accent); background: #eef6f7; }
    .table-wrap { overflow: auto; }
    table { width: 100%; border-collapse: collapse; min-width: 1040px; }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      background: #fbfcfd;
    }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    tr:last-child td { border-bottom: 0; }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 7px;
      border-radius: 999px;
      background: #eef6f7;
      color: var(--accent);
      font-size: 12px;
      font-weight: 650;
    }
    .pill.base { background: #f4f1ee; color: var(--accent-2); }
    .file { color: var(--muted); font-size: 12px; }
    .empty {
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 24px;
      color: var(--muted);
      background: #fff;
    }
    .note { color: var(--muted); margin-top: 12px; font-size: 12px; }
    .delta.good { color: var(--good); }
    .delta.warn { color: var(--warn); }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    @media (max-width: 900px) {
      .top, .run-head { align-items: flex-start; flex-direction: column; }
      .stats, .help { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 560px) {
      .stats, .help { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap top">
      <div>
        <h1>tinyvla benchmarks</h1>
        <div class="sub">Grouped by benchmark run from <code>artifacts/benchmarks</code></div>
      </div>
      <button type="button" onclick="load()">Refresh</button>
    </div>
  </header>
  <main class="wrap">
    <section class="stats" id="stats"></section>
    <section class="help">
      <div class="help-card">
        <strong>Offline loss</strong>
        <span>The training-style denoising error. Use it to compare models within the same run. Lower is better.</span>
      </div>
      <div class="help-card">
        <strong>Expert MAE/RMSE</strong>
        <span>Action error against the scripted dataset expert. This says whether the model matches the task data.</span>
      </div>
      <div class="help-card">
        <strong>vs 450M MAE/RMSE</strong>
        <span>Action error against the original teacher model. This says whether pruning preserved behavior.</span>
      </div>
      <div class="help-card">
        <strong>Success / final distance</strong>
        <span>Closed-loop simulator metrics. Success means it reached the target; final distance is how far away it ended.</span>
      </div>
    </section>
    <section id="content"></section>
    <div class="note">For head/vocab pruning, the important gate is vs 450M: default pass means teacher MAE <= 0.01 and max absolute action difference <= 0.10 in normalized action space.</div>
  </main>
  <script>
    const fmt = new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 });
    const compact = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 2 });

    function asNumber(value, digits = 3) {
      return value === undefined || value === null ? "—" : fmt.format(Number(Number(value).toFixed(digits)));
    }
    function pct(value) {
      return value === undefined || value === null ? "—" : `${fmt.format(Number((value * 100).toFixed(1)))}%`;
    }
    function params(value) {
      return value === undefined || value === null ? "—" : compact.format(value);
    }
    function modelOrder(row) {
      if (row.name === "base") return 0;
      if (row.name === "pruned") return 1;
      if (!row.pruned) return 2;
      return 3;
    }
    function runsFrom(data) {
      return data.map(run => {
        const rows = Object.entries(run.models || {}).map(([name, model]) => ({
          run: run._file,
          dataset: run.dataset?.root || "—",
          frames: run.dataset?.frames,
          device: run.device || "—",
          name,
          ...model,
        })).sort((a, b) => modelOrder(a) - modelOrder(b) || a.name.localeCompare(b.name));
        return { ...run, rows };
      }).sort((a, b) => {
        const comparisonDelta = (b.rows.length > 1) - (a.rows.length > 1);
        if (comparisonDelta) return comparisonDelta;
        return (b._mtime || 0) - (a._mtime || 0);
      });
    }
    function stat(label, value) {
      return `<div class="stat"><label>${label}</label><strong>${value}</strong></div>`;
    }
    function delta(value, base, lowerIsBetter = true) {
      if (value === undefined || value === null || base === undefined || base === null || value === base) return "—";
      const d = value - base;
      const good = lowerIsBetter ? d < 0 : d > 0;
      const sign = d > 0 ? "+" : "";
      return `<span class="delta ${good ? "good" : "warn"}">${sign}${asNumber(d)}</span>`;
    }
    function tolerance(row) {
      if (row.teacher_within_tolerance === undefined || row.teacher_within_tolerance === null) return "—";
      return `<span class="pill ${row.teacher_within_tolerance ? "" : "base"}">${row.teacher_within_tolerance ? "pass" : "fail"}</span>`;
    }
    function expertMae(row) {
      return row.expert_action_mae ?? row.action_mae;
    }
    function expertRmse(row) {
      return row.expert_action_rmse ?? row.action_rmse;
    }
    function renderRun(run) {
      const base = run.rows.find(row => row.name === "base") || run.rows.find(row => !row.pruned) || run.rows[0];
      const hasTeacher = run.teacher !== undefined && run.teacher !== null;
      const badge = hasTeacher ? "vs 450M teacher" : (run.rows.length > 1 ? "comparison" : "single model");
      return `
        <section class="run-card">
          <div class="run-head">
            <div>
              <div class="run-title">${run._file}</div>
              <div class="run-meta">${run.dataset?.root || "—"} · ${run.dataset?.frames ?? "—"} frames · ${run.device || "—"}</div>
            </div>
            <div class="badge ${(run.rows.length > 1 || hasTeacher) ? "compare" : ""}">${badge}</div>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Model</th><th>Kind</th><th>Params</th><th>Delta Params</th>
                  <th>Offline Loss</th><th>Delta Loss</th><th>Expert MAE</th><th>Expert RMSE</th>
                  <th>vs 450M MAE</th><th>vs 450M RMSE</th><th>vs 450M Max</th><th>Tolerance</th>
                  <th>Success</th><th>Final Dist</th><th>Seconds</th>
                </tr>
              </thead>
              <tbody>
                ${run.rows.map(row => `
                  <tr>
                    <td><div>${row.name}</div><div class="file">${row.path || ""}</div></td>
                    <td><span class="pill ${row.pruned ? "" : "base"}">${row.pruned ? "pruned" : "base"}</span></td>
                    <td class="num">${params(row.parameters)}</td>
                    <td class="num">${delta(row.parameters, base?.parameters)}</td>
                    <td class="num">${asNumber(row.offline_loss_mean)}</td>
                    <td class="num">${delta(row.offline_loss_mean, base?.offline_loss_mean)}</td>
                    <td class="num">${asNumber(expertMae(row))}</td>
                    <td class="num">${asNumber(expertRmse(row))}</td>
                    <td class="num">${asNumber(row.teacher_action_mae, 6)}</td>
                    <td class="num">${asNumber(row.teacher_action_rmse, 6)}</td>
                    <td class="num">${asNumber(row.teacher_action_max_abs, 6)}</td>
                    <td class="num">${tolerance(row)}</td>
                    <td class="num">${pct(row.closed_loop_success_rate)}</td>
                    <td class="num">${asNumber(row.closed_loop_final_dist_mean)}</td>
                    <td class="num">${asNumber(row.offline_seconds ?? row.closed_loop_seconds, 1)}</td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          </div>
        </section>`;
    }
    async function load() {
      const res = await fetch("/api/benchmarks", { cache: "no-store" });
      const data = await res.json();
      const runs = runsFrom(data);
      const rows = runs.flatMap(run => run.rows);
      const pruned = rows.filter(row => row.pruned).length;
      const latest = runs[0]?.rows[0];
      document.getElementById("stats").innerHTML = [
        stat("Runs", data.length),
        stat("Model Rows", rows.length),
        stat("Pruned Rows", pruned),
        stat("First Row Params", latest ? params(latest.parameters) : "—"),
      ].join("");
      if (!rows.length) {
        document.getElementById("content").innerHTML =
          `<div class="empty">No benchmark JSON found. Run <code>python3 -m tinyvla.benchmark ...</code> first.</div>`;
        return;
      }
      document.getElementById("content").innerHTML = runs.map(renderRun).join("");
    }
    load();
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif path == "/api/benchmarks":
            self._send(
                200,
                "application/json; charset=utf-8",
                json.dumps(load_benchmarks(), indent=2).encode(),
            )
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"tinyvla benchmark dashboard at http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
