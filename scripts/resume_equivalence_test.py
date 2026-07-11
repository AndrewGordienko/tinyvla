"""Compare uninterrupted and interrupted fixed-tuple training trajectories."""
from __future__ import annotations
import argparse, copy, hashlib, json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from tinyvla.fast_dataset import FastChunkDataset
from tinyvla.runtime import load_runtime
from tinyvla.trainability import set_trainable

def make_run(model, meta, root, device, seed=1234):
    rt=load_runtime(model,meta=meta,dataset_root=root,device=device,stats_source='dataset',base_checkpoint=True); p=rt.policy; set_trainable(p,'expert'); return rt,p
def state_hash(state):
    h=hashlib.sha256()
    def add(k,v):
        h.update(str(k).encode())
        if torch.is_tensor(v): h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,dict):
            for kk in sorted(v): add(kk,v[kk])
        else: h.update(str(v).encode())
    for k in sorted(state): add(k,state[k])
    return h.hexdigest()
def train_step(p,opt,sched,batch,noise,timestep):
    p.train(); opt.zero_grad(set_to_none=True); loss,_=p.forward(batch,noise=noise,time=timestep); loss.backward(); pre=torch.nn.utils.clip_grad_norm_([q for q in p.parameters() if q.requires_grad],10.0); opt.step(); sched.step(); return float(loss),float(pre)
def snapshot(p,opt,sched):
    return {'model':copy.deepcopy(p.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'scheduler':copy.deepcopy(sched.state_dict()),'rng':torch.get_rng_state().clone()}
def compare(a,b):
    diffs=[]
    for k in a:
        if isinstance(a[k],dict): diffs.extend(compare(a[k],b[k]))
        elif torch.is_tensor(a[k]):
            d=float((a[k].float()-b[k].float()).abs().max());
            if d: diffs.append((k,d))
        elif a[k]!=b[k]: diffs.append((k,a[k],b[k]))
    return diffs
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--root',required=True); ap.add_argument('--repo-id',required=True); ap.add_argument('--device',default='mps'); ap.add_argument('--out',required=True)
    a=ap.parse_args(); t0=time.time(); device=torch.device(a.device); meta=LeRobotDatasetMetadata(a.repo_id,root=a.root); ds=FastChunkDataset(a.repo_id,root=a.root,delta_timestamps={'action':[i/meta.fps for i in range(SmolVLAConfig().chunk_size)]}); raw=next(iter(DataLoader(ds,batch_size=1,shuffle=False,num_workers=0))); fixed_hash=state_hash({k:v for k,v in raw.items() if torch.is_tensor(v)})
    torch.manual_seed(4242); rt,p=make_run(a.model,meta,a.root,device); batch=rt.preprocessor(dict(raw)); noise=torch.randn((1,p.config.chunk_size,p.config.max_action_dim),device=device); timestep=torch.full((1,),.37,device=device); tuple_hash=state_hash({**batch,'noise':noise,'timestep':timestep}); params=[q for q in p.parameters() if q.requires_grad]; opt=torch.optim.AdamW(params,lr=3e-5,betas=(.9,.95),weight_decay=1e-10); sched=p.config.get_scheduler_preset().build(opt,num_training_steps=25); base=snapshot(p,opt,sched)
    uninterrupted=[]; torch.set_rng_state(base['rng']);
    for step in range(1,27):
        uninterrupted.append(train_step(p,opt,sched,batch,noise,timestep))
    u26=snapshot(p,opt,sched); u_loss,_=p.forward(batch,noise=noise,time=timestep)
    # A separate base-weight policy simulates stop/reload at step 10.
    torch.manual_seed(4242); rt2,p2=make_run(a.model,meta,a.root,device); b2=rt2.preprocessor(dict(raw)); torch.manual_seed(4242); noise2=torch.randn_like(noise); t2=torch.full_like(timestep,.37); opt2=torch.optim.AdamW([q for q in p2.parameters() if q.requires_grad],lr=3e-5,betas=(.9,.95),weight_decay=1e-10); sched2=p2.config.get_scheduler_preset().build(opt2,num_training_steps=25); 
    torch.set_rng_state(base['rng'])
    for _ in range(10): train_step(p2,opt2,sched2,b2,noise2,t2)
    saved=snapshot(p2,opt2,sched2); torch.save(saved,Path(a.out).with_suffix('.resume.pt')); restored=torch.load(Path(a.out).with_suffix('.resume.pt'),map_location='cpu',weights_only=False); p2.load_state_dict(restored['model']); opt2.load_state_dict(restored['optimizer']); sched2.load_state_dict(restored['scheduler']); torch.set_rng_state(restored['rng'])
    resumed=[]
    for _ in range(15): resumed.append(train_step(p2,opt2,sched2,b2,noise2,t2))
    resumed.append(train_step(p2,opt2,sched2,b2,noise2,t2)); r26=snapshot(p2,opt2,sched2); r_loss,_=p2.forward(b2,noise=noise2,time=t2)
    diffs=compare(u26,r26); result={'planned_total_steps':25,'resume_step':10,'final_step':26,'learning_rate':3e-5,'dataset_tuple_hash':fixed_hash,'processed_tuple_hash':tuple_hash,'uninterrupted_step25_loss':uninterrupted[24][0],'resumed_step25_loss':resumed[14][0],'uninterrupted_step26_loss':uninterrupted[25][0],'resumed_step26_loss':resumed[15][0],'max_state_difference':max([x[1] for x in diffs if len(x)>1 and isinstance(x[1],float)] or [0.0]),'difference_count':len(diffs),'exact_or_tight_equivalence':len(diffs)==0,'rng_equal':torch.equal(u26['rng'],r26['rng']),'runtime_seconds':time.time()-t0,'artifact':str(Path(a.out).with_suffix('.resume.pt'))}
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
