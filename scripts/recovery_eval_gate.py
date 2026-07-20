"""Promotion gate for a recovery-round champion vs the deployed student + teacher.

Runs the exact existing 30-rollout eval (seed 999, per-command 10, scoped cmds
1,3,4) and applies the promotion criteria:
  - >= 3/4 deterministic scenes (declared fixed (command,episode) scenes)
  - >= 18/30 overall success, or a clearly justified approach to the 66.7% teacher
  - release/place improves materially vs the deployed student
  - no material regression in approach / grasp / transport

Baselines are read from the saved head-to-head evals (deployed student 33.3%,
teacher 66.7%). No expert is used at evaluation — this is pure closed-loop policy.

Run: .venv/bin/python scripts/recovery_eval_gate.py --model <ckpt> [--tag r1]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tinyvla.paths import ARTIFACTS_ROOT

# declared up front, deterministic: one scene per scoped command + a 2nd of cmd3
GATE_SCENES = [(1, 0), (3, 0), (4, 0), (3, 1)]
STUDENT_BASELINE = ARTIFACTS_ROOT / "evaluations" / "headto_student_brain.json"
TEACHER_BASELINE = ARTIFACTS_ROOT / "evaluations" / "headto_teacher_450.json"
REGRESSION_TOL = 0.15  # max allowed drop in approach/grasp/transport


def run_eval(model: str, out: Path) -> dict:
    cmd = [sys.executable, "-m", "tinyvla.eval", "--model", model,
           "--per-command", "10", "--commands", "1,3,4", "--seed", "999",
           "--device", "mps", "--output", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not out.exists():
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit("eval failed")
    return json.load(open(out))["metrics"]


def scene_success(metrics: dict, scenes) -> list[bool]:
    rows = {(int(r["command"]), int(r["episode"])): r for r in metrics.get("stage_rows", [])}
    out = []
    for (c, e) in scenes:
        r = rows.get((c, e))
        out.append(bool(r and r.get("release")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="champion")
    args = ap.parse_args()

    out = ARTIFACTS_ROOT / "evaluations" / f"recovery_gate_{args.tag}.json"
    m = run_eval(args.model, out)
    base = json.load(open(STUDENT_BASELINE))["metrics"] if STUDENT_BASELINE.exists() else None
    teach = json.load(open(TEACHER_BASELINE))["metrics"] if TEACHER_BASELINE.exists() else None

    st = m["stage_completion"]
    b_st = base["stage_completion"] if base else {}
    print(f"\n=== {args.tag}: {args.model}")
    print(f"overall success : {m['success_rate']*100:.1f}%  ({m['successes']}/{m['n']})"
          + (f"   [student {base['success_rate']*100:.0f}%  teacher {teach['success_rate']*100:.0f}%]" if base and teach else ""))
    for c in ("1", "3", "4"):
        pc = m["per_command"][c]
        print(f"  cmd{c}: {pc['success_rate']*100:.0f}%")
    print("stages (new vs deployed student):")
    for s in ("approach", "grasp", "transport", "release"):
        d = st.get(s, 0) - b_st.get(s, 0)
        print(f"  {s:10s} {st.get(s,0):.2f}  (Δ {d:+.2f})")

    scenes = scene_success(m, GATE_SCENES)
    n_scene = sum(scenes)
    print(f"4-scene gate {GATE_SCENES}: {['PASS' if s else 'fail' for s in scenes]} -> {n_scene}/4")

    # promotion criteria
    c_scene = n_scene >= 3
    c_overall = m["successes"] >= 18
    c_release = (st.get("release", 0) - b_st.get("release", 0)) >= 0.10 if base else st.get("release", 0) >= 0.60
    c_noregress = all((st.get(s, 0) - b_st.get(s, 0)) >= -REGRESSION_TOL for s in ("approach", "grasp", "transport")) if base else True
    print("\n=== PROMOTION CRITERIA")
    print(f"  [{'x' if c_scene else ' '}] >=3/4 deterministic scenes           ({n_scene}/4)")
    print(f"  [{'x' if c_overall else ' '}] >=18/30 overall                       ({m['successes']}/30)")
    print(f"  [{'x' if c_release else ' '}] release improves materially           (Δ {st.get('release',0)-b_st.get('release',0):+.2f})")
    print(f"  [{'x' if c_noregress else ' '}] no material regression approach/grasp/transport")
    promoted = c_scene and c_overall and c_release and c_noregress
    print(f"\nVERDICT: {'PROMOTE' if promoted else 'DO NOT PROMOTE (round did not clear the gate)'}")
    # near-miss note: justified approach to teacher even if <18/30
    if not c_overall and m["successes"] >= 14 and c_scene and c_release and c_noregress:
        print("NOTE: below 18/30 but scenes+release+no-regress cleared and overall approaches teacher — "
              "candidate for 'clearly justified best' promotion; review manually.")
    return 0 if promoted else 1


if __name__ == "__main__":
    raise SystemExit(main())
