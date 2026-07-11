"""Compare fixed-tuple learning rates without changing the base checkpoint."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from tinyvla.fast_dataset import FastChunkDataset
from tinyvla.runtime import load_runtime
from tinyvla.trainability import set_trainable

def digest(values):
    h=hashlib.sha256()
    for k,v in sorted(values.items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes() if torch.is_tensor(v) else str(v).encode())
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--repo-id',required=True); ap.add_argument('--model',required=True); ap.add_argument('--device',default='mps'); ap.add_argument('--steps',type=int,default=25); ap.add_argument('--out',required=True)
    a=ap.parse_args(); device=torch.device(a.device); meta=LeRobotDatasetMetadata(a.repo_id,root=a.root)
    ds=FastChunkDataset(a.repo_id,root=a.root,delta_timestamps={'action':[i/meta.fps for i in range(SmolVLAConfig().chunk_size)]}); raw=next(iter(DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)))
    results={}
    for lr in (1e-5,3e-5,1e-4):
        rt=load_runtime(a.model,meta=meta,dataset_root=a.root,device=device,stats_source='dataset',base_checkpoint=True); p=rt.policy; set_trainable(p,'expert'); p.train(); batch=rt.preprocessor(dict(raw)); torch.manual_seed(4242); noise=torch.randn((1,p.config.chunk_size,p.config.max_action_dim),device=device); timestep=torch.full((1,),.37,device=device)
        target_velocity=F.pad(batch['action'],(0,noise.shape[-1]-batch['action'].shape[-1]))-noise
        hashes={'raw_indices':digest({k:v for k,v in raw.items() if 'index' in k or 'episode' in k}),'processed':digest(batch),'noise':digest({'noise':noise}),'time':digest({'time':timestep}),'padding_mask':digest({k:v for k,v in batch.items() if 'pad' in k}),'language_tokens':digest({k:v for k,v in batch.items() if 'token' in k or 'mask' in k}),'target_velocity':digest({'target_velocity':target_velocity}),'normalization_stats':hashlib.sha256(Path(a.root,'meta','stats.json').read_bytes()).hexdigest(),'tuple':digest({**batch,'noise':noise,'time':timestep})}; opt=torch.optim.AdamW([q for q in p.parameters() if q.requires_grad],lr=lr,betas=(.9,.95),weight_decay=1e-10); initial={k:v.detach().clone() for k,v in p.state_dict().items()}; curve=[]; finite=True; t0=time.time()
        for step in range(a.steps+1):
            p.eval();
            with torch.inference_mode(): ev,_=p.forward(batch,noise=noise,time=timestep)
            if step in (0,10,a.steps): hashes[f'step_{step}']=digest({**batch,'noise':noise,'time':timestep,'eval_loss':ev.detach()}); curve.append({'step':step,'eval_loss':float(ev)})
            if step==a.steps: break
            p.train(); p.zero_grad(set_to_none=True); loss,_=p.forward(batch,noise=noise,time=timestep); loss.backward(); pre=float(torch.nn.utils.clip_grad_norm_([q for q in p.parameters() if q.requires_grad],10.0)); post=float(torch.sqrt(sum((q.grad.detach()**2).sum() for q in p.parameters() if q.grad is not None))); finite=finite and bool(torch.isfinite(loss)) and torch.isfinite(torch.tensor(pre)) and torch.isfinite(torch.tensor(post)); opt.step()
        update=torch.sqrt(sum(((p.state_dict()[k].float()-v.float())**2).sum() for k,v in initial.items())); max_update=max(float((p.state_dict()[k].float()-v.float()).abs().max()) for k,v in initial.items()); out=Path(a.out)/f'lr_{lr:g}'; out.mkdir(parents=True,exist_ok=True); torch.save(p.state_dict(),out/'state.pt'); re=load_runtime(a.model,meta=meta,dataset_root=a.root,device=device,stats_source='dataset',base_checkpoint=True).policy; re.load_state_dict(torch.load(out/'state.pt',map_location=device,weights_only=True)); re.eval(); p.eval();
        torch.manual_seed(8181)
        with torch.inference_mode(): act=p.select_action(batch)
        torch.manual_seed(8181)
        with torch.inference_mode(): act2=re.select_action(batch)
        results[str(lr)]={'curve':curve,'hashes':hashes,'gradient_finite':finite,'update_norm':float(update),'max_parameter_update':max_update,'save_reload_action_diff':float((act-act2).abs().max()),'runtime_seconds':time.time()-t0}
    Path(a.out).mkdir(parents=True,exist_ok=True); Path(a.out,'diagnostic.json').write_text(json.dumps(results,indent=2,default=str)+'\n'); print(json.dumps(results,indent=2,default=str))
if __name__=='__main__': main()
