"""Verify fixed-batch, checkpoint, resume, and short rollout properties."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import imageio.v3 as iio
import numpy as np, torch, mujoco
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from tinyvla.fast_dataset import FastChunkDataset
from tinyvla.runtime import load_runtime, sha256_tree, sha256_file
from tinyvla.task import SO101PickPlaceTask, COMMANDS

def load_policy(path, meta, root, device):
    rt=load_runtime(path,meta=meta,dataset_root=root,device=device,stats_source='dataset',base_checkpoint=(Path(path).name=='smolvla_base'))
    return rt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--repo-id',required=True); ap.add_argument('--base',required=True); ap.add_argument('--step10',required=True); ap.add_argument('--step25',required=True); ap.add_argument('--video',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); t0=time.time(); device=torch.device('cpu'); meta=LeRobotDatasetMetadata(a.repo_id,root=a.root)
    ds=FastChunkDataset(a.repo_id,root=a.root,delta_timestamps={'action':[i/meta.fps for i in range(SmolVLAConfig().chunk_size)]})
    raw=next(iter(DataLoader(ds,batch_size=1,shuffle=False,num_workers=0))); fixed_hash=hashlib.sha256()
    for k,v in raw.items():
        if torch.is_tensor(v): fixed_hash.update(k.encode()); fixed_hash.update(v.numpy().tobytes())
    def fixed(path):
        rt=load_policy(path,meta,a.root,device); b=rt.preprocessor(dict(raw)); rt.policy.train(); torch.manual_seed(1234 + 991)
        noise=torch.randn((1,rt.policy.config.chunk_size,rt.policy.config.max_action_dim)); tm=torch.full((1,),.37)
        with torch.inference_mode(): loss,_=rt.policy.forward(b,noise=noise,time=tm); action=rt.policy.select_action(b)
        return rt,float(loss),action.detach().cpu(),b,noise,tm
    base_loss=None
    base_rt,base_loss,_,_,_,_=fixed(a.base)
    r10,l10,act10,_,_,_=fixed(a.step10); r25,l25,act25,b,noise,tm=fixed(a.step25)
    _,l25_reload,act25_reload,_,_,_=fixed(a.step25)
    sbase=base_rt.policy.state_dict(); sfinal=r25.policy.state_dict(); sq=0.; n=0
    for k in sbase:
        d=(sfinal[k].float()-sbase[k].float()); sq+=float((d*d).sum()); n+=d.numel()
    st10=torch.load(Path(a.step10)/'training_state.pt',map_location='cpu',weights_only=False); st25=torch.load(Path(a.step25)/'training_state.pt',map_location='cpu',weights_only=False)
    opt_steps10=sorted({int(v['step']) for v in st10['optimizer']['state'].values() if 'step' in v}); opt_steps25=sorted({int(v['step']) for v in st25['optimizer']['state'].values() if 'step' in v})
    # Short real rollout; front video is the canonical viewer artifact.
    env=SO101PickPlaceTask(seed=1234); env.reset(command=0); renderer=mujoco.Renderer(env.model,height=256,width=256); frames=[]; actions=[]; rt=r25; policy=rt.policy; prep=rt.preprocessor; policy.eval()
    for _ in range(30):
        renderer.update_scene(env.data,camera='front'); frames.append(renderer.render().copy()); renderer.update_scene(env.data,camera='wrist')
        from tinyvla.eval_closedloop import build_obs
        obs=prep(build_obs(env,{'front':renderer,'wrist':renderer},COMMANDS[0]['instruction'],device))
        with torch.inference_mode(): action=rt.postprocessor(policy.select_action(obs)).squeeze(0).numpy()
        actions.append(action.tolist()); env.step(action)
    renderer.close(); Path(a.video).parent.mkdir(parents=True,exist_ok=True); iio.imwrite(a.video,np.stack(frames),fps=25)
    result={'dataset_sha256':sha256_tree(a.root),'base_checkpoint_sha256':sha256_tree(a.base),'git_sha':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),'fixed_batch_sha256':fixed_hash.hexdigest(),'initial_fixed_loss':base_loss,'step10_fixed_loss':l10,'final_fixed_loss':l25,'loss_improved':l25<base_loss,'parameter_update_norm':sq**0.5,'parameter_rms_update':(sq/n)**0.5,'save_reload_action_max_abs_diff':float((act25-act25_reload).abs().max()),'save_reload_loss_abs_diff':abs(l25-l25_reload),'resume_global_steps':[st10['global_step'],st25['global_step']],'optimizer_steps':[opt_steps10,opt_steps25],'scheduler_state_step10':st10.get('scheduler'),'scheduler_state_step25':st25.get('scheduler'),'action_min':float(np.min(actions)),'action_max':float(np.max(actions)),'rollout_video':a.video,'runtime_seconds':time.time()-t0,'mps_fallback':'PYTORCH_ENABLE_MPS_FALLBACK=1 (training logs captured separately)','checkpoint_read_only_hash':sha256_tree(a.step25)}
    Path(a.out).write_text(json.dumps(result,indent=2,default=str)+'\n'); print(json.dumps(result,indent=2,default=str))
if __name__=='__main__': main()
