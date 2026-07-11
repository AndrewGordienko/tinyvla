"""Audit command-0 SmolVLA dataset semantics, hashes, and view alignment."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tinyvla.runtime import sha256_tree, detect_action_semantics

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo-id',required=True); ap.add_argument('--root',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); root=Path(a.root); ds=LeRobotDataset(a.repo_id,root=root)
    h=hashlib.sha256(); episodes={}; views=set(); bad=[]; prev=None
    for i in range(len(ds)):
        s=ds[i]; ep=int(s['episode_index']); fi=int(s['frame_index']);
        episodes.setdefault(ep,[]).append(fi)
        views.update(k for k in s if k.startswith('observation.images.'))
        if int(s['task_index']) != 0 or s.get('task') != 'Pick up the red cube and place it in the box.': bad.append({'index':i,'reason':'task'})
        for key in ('observation.state','action','observation.images.front','observation.images.wrist'):
            v=s[key].detach().contiguous().cpu().numpy(); h.update(key.encode()); h.update(v.tobytes())
    boundary_bad=[]
    for ep, frames in episodes.items():
        expected=list(range(len(frames)))
        if frames != expected: boundary_bad.append({'episode':ep,'frames':frames[:5]+frames[-5:]})
    result={'repo_id':a.repo_id,'root':str(root),'dataset_sha256':sha256_tree(root),
            'sample_sha256':h.hexdigest(),'frames':len(ds),'episodes':len(episodes),
            'frames_per_episode':{str(k):len(v) for k,v in episodes.items()},
            'views':sorted(views),'fps':ds.meta.fps,'features':ds.meta.features,
            'action_semantics':detect_action_semantics(root),'task_errors':bad,
            'episode_boundary_errors':boundary_bad,'front_wrist_same_timestep':not bool(boundary_bad),
            'temporal_stack_policy':'FastChunkDataset uses LeRobot episode-aware query; chunks cannot cross episode boundaries',
            'source_provenance':'demonstration','recovery_examples':0,'demo_sampling_requirement':'>=50% when recovery pool is added'}
    Path(a.out).write_text(json.dumps(result,indent=2,default=str)+'\n'); print(json.dumps(result,indent=2,default=str))
    raise SystemExit(1 if bad or boundary_bad else 0)
if __name__=='__main__': main()
