# tinyvla — definitive held-out benchmark

Device: **Apple M5 Pro (MPS)** · seeds: held-out **3001, 5002** · `n_action_steps=50` · date 2026-07-21

Reproduce: `MUJOCO_GL=glfw .venv/bin/python -m tinyvla.eval --model <ckpt> --commands 0,1,2,3,4,5,6,7 --per-command 25 --seed 3001` then `.venv/bin/python scripts/benchmark_report.py`.

## Footprint, latency, success

| Model | Params | On-disk | MPS working set | Replan | Success (held-out) |
|---|---:|---:|---:|---:|---:|
| Teacher 450M | 450M | 1126 MB | 1199 MB | 225 ms | **42%** (95% CI 38–47%), 8 cmds, n=400 |
| Student 291M bf16 | 292M | **557 MB** | 896 MB | 194 ms | **33%** (95% CI 27–41%), cmds 1/3/4, n=180 |

Head-to-head on the student's trained commands (1,3,4): teacher **46%** (69/150) vs student **33%** (60/180). The student retains ~72% of the teacher here; CIs overlap.

## Teacher per-command (8 commands)

| cmd | task | success |
|---:|---|---:|
| 0 | red cube -> box | 20/50 = 40% |
| 1 | blue cube -> box | 19/50 = 38% |
| 2 | red cube -> plate | 28/50 = 56% |
| 3 | blue cube -> plate | 27/50 = 54% |
| 4 | red on top of blue | 23/50 = 46% |
| 5 | blue on top of red | 16/50 = 32% |
| 6 | red->box + blue->plate (2-step) | 15/50 = 30% |
| 7 | blue->box + red->plate (2-step) | 21/50 = 42% |

## Where failures happen (fraction of failed episodes stuck at each stage)

- **Teacher:** transport 60%, reach 37%, approach 3%
- **Student:** transport 49%, reach 45%, approach 6%

**Reach** = gripper never got within 5 cm of the cube; **transport** = grasped then dropped/misplaced mid-carry. Placement precision is *not* the dominant failure on hard held-out scenes — reaching + transport are (~90% combined). This points to a data-distribution gap (hard object poses), not model capacity.

## Checkpoint hashes (sha256 of model.safetensors)

- `teacher_450M`: `38d4b15a4743b99070017e6a4b5282a345fecf69a65fe24a564668f1aa5250bc`
- `student_291M_bf16`: `d38cfa1e8bcd606df720fa06ae4c2b0b491fb058b7f54f57f86ee5284ec8ebfd`
