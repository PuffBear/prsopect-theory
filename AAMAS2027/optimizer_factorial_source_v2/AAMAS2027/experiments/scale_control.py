"""Prospective scalar-vs-asymmetric controls; calibration and confirmation kept separate."""
from pathlib import Path
import sys,json,argparse,time,hashlib
import numpy as np,torch,supersuit as ss
import horizon_supply as h
from instrumented_ppo import InstrumentedPPO,GroupwiseClipPPO,SeparateNoClipPPO,RolloutDiagnostics
from reward_diagnostics import moments,scale_constant,transform
WS=[0.,.4,.5,.6,.7,.85,1.]

class ExplicitDemandRNG(h.DemandRNG):
 """Bind cloned SciPy distributions to an explicitly seeded per-environment stream."""
 def __init__(self,env,seed):
  super().__init__(env,seed);self.stream=np.random.RandomState(seed)
 def reset(self,seed=None,options=None):
  if seed is not None:self.stream.seed(seed)
  return super().reset(seed=seed,options=options)
 def invoke(self,fn,*args,**kwargs):
  distributions=[]
  for _,_,data in self.env.unwrapped.unwrapped_env.graph.edges(data=True):
   dist=data.get('demand_dist')
   if dist is not None and all(dist is not x[0] for x in distributions):
    distributions.append((dist,dist.random_state));dist.random_state=self.stream
  try:return super().invoke(fn,*args,**kwargs)
  finally:
   for dist,state in distributions:dist.random_state=state

def build(w,arm,seed,c=1.,reference=0.):
 env=h.build(w,'null' if arm=='linear' else ('lambda7' if arm in ['A','N'] else 'curvature'),seed)
 diag=env.env.env;diag.env=ExplicitDemandRNG(diag.env.env,seed)
 env.env.output_scale=c if arm=='N' else (1/c if arm in ['S','R'] else 1.)
 env.env.reference_point=reference
 return env

def vector(w,arm,seed,c=1.,reference=0.):
 vec=ss.pettingzoo_env_to_vec_env_v1(ss.pad_action_space_v0(build(w,arm,seed,c,reference)))
 vec=ss.concat_vec_envs_v1(vec,1,num_cpus=0,base_class='stable_baselines3');vec.seed=lambda s:None
 return vec

class CheckpointPPO:
 def train(self):
  super().train()
  for checkpoint in self.protocol['checkpoints']:
   if checkpoint<=self.num_timesteps and checkpoint not in self.completed:
    self.completed.append(checkpoint);out=Path(self.output)
    r=h.evaluate(self,self.protocol,10);r.update(requested_checkpoint=checkpoint,actual_agent_transitions=self.num_timesteps,elapsed_seconds=time.monotonic()-self.started)
    (out/f'evaluation_{checkpoint}.json').write_text(json.dumps(r,indent=2));h.HorizonPPO.checkpoint(self,out/f'checkpoint_{checkpoint}')

class ScalePPO(CheckpointPPO,InstrumentedPPO):
 pass

class GroupwiseScalePPO(CheckpointPPO,GroupwiseClipPPO):
 pass

def create(out,w,arm,seed,c=1.,reference=0.,budget=300000,phase='confirmation',optimizer_mode='global'):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
 if optimizer_mode not in ['global','groupwise']:raise ValueError(optimizer_mode)
 cls=ScalePPO if optimizer_mode=='global' else GroupwiseScalePPO
 m=cls('MlpPolicy',vector(w,arm,seed,c,reference),n_steps=128,batch_size=256,seed=seed,verbose=0,device='cpu')
 m.diagnostics_dir=str(out/'diagnostics');m.output=str(out);m.started=time.monotonic();m.completed=[]
 m.protocol=dict(w=w,arm=arm,c=c,reference=reference,seed=seed,condition='lambda7' if arm in ['A','N'] else 'curvature',checkpoints=[budget],total_budget=budget,phase=phase,optimizer_mode=optimizer_mode,max_grad_norm=float(m.max_grad_norm))
 m.verify_upstream();(out/'protocol.json').write_text(json.dumps(m.protocol,indent=2));manifest=h.manifest();manifest['control_sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('instrumented_ppo.py'),Path(__file__).with_name('reward_diagnostics.py')]};(out/'manifest.json').write_text(json.dumps(manifest,indent=2));return m

def calibrate(out,index):
 assert 0<=index<35
 w=WS[index//5];seed=100+index%5;m=create(out,w,'B',seed,budget=20000,phase='calibration')
 m.learn(total_timesteps=20000,callback=RolloutDiagnostics());env=build(w,'B',seed);stream=[]
 state=h.rng_state()
 try:
  for ep in range(10):
   obs,_=env.reset(seed=300000+seed*10+ep)
   while env.agents:
    actions={a:m.predict(obs[a],deterministic=True)[0][:env.action_space(a).shape[0]] for a in env.agents}
    obs,_,_,_,infos=env.step(actions)
    for a,info in infos.items():stream.append(dict(agent=a,episode=ep,blend=info['blended_reward']))
 finally:env.close();h.restore_rng(state)
 record=dict(w=w,seed=seed,phase='untreated-calibration',policy_budget_requested=20000,policy_budget_actual=m.num_timesteps,stream=stream)
 Path(out,'calibration_stream.json').write_text(json.dumps(record));print('CALIBRATION COMPLETE',index,w,seed,flush=True)

def freeze(root,out):
 root=Path(root);records=[]
 for i in range(35):
  p=root/f'run_{i:03d}/calibration_stream.json'
  if not p.exists():raise ValueError(f'Incomplete calibration: {p}')
  r=json.loads(p.read_text());assert r['seed']==100+i%5 and r['w']==WS[i//5];records.append((p,r))
 result=dict(protocol='fixed untreated B policies; 20k requested transitions; ten evaluation episodes per seed, seeds100–104; no confirmation data',constants={},input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p,r in records})
 for w in WS:
  vals=np.array([v['blend'] for p,r in records if r['w']==w for v in r['stream']]);result['constants'][str(w)]={str(ref):dict(c=scale_constant(vals-ref),moments=moments(vals-ref)) for ref in [0.,-6.77]}
 with Path(out).open('x') as f:json.dump(result,f,indent=2)
 print('FROZEN',out)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--calibrate',type=int);p.add_argument('--freeze');p.add_argument('--constants');p.add_argument('--index',type=int);p.add_argument('--reference',type=float,default=0.);a=p.parse_args()
 if a.calibrate is not None:return calibrate(a.out,a.calibrate)
 if a.freeze:return freeze(a.freeze,a.out)
 if a.index is None or not a.constants:raise ValueError('Confirmation needs index and frozen constants')
 assert 0<=a.index<560
 w=WS[a.index//80];arm=['B','A','S','N'][(a.index%80)//20];seed=42+a.index%20
 c=json.loads(Path(a.constants).read_text())['constants'][str(w)][str(a.reference)]['c'];m=create(a.out,w,arm,seed,c,a.reference)
 (Path(a.out)/'calibration.sha256').write_text(hashlib.sha256(Path(a.constants).read_bytes()).hexdigest())
 m.learn(total_timesteps=300000,callback=RolloutDiagnostics());print('COMPLETE',a.index,m.num_timesteps)
if __name__=='__main__':main()
