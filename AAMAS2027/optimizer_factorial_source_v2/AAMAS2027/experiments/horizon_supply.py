"""New, isolated supply horizon protocol. Post-update checkpoints; full resume state.
Never appends to legacy CSVs. Run smoke before scheduling any pilot/confirmation.
"""
from pathlib import Path
import sys,os,json,time,random,hashlib,copy,argparse,platform
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'or-gym'),str(ROOT),str(ROOT/'experiments')]
import numpy as np,torch,cloudpickle
import supersuit as ss
from stable_baselines3 import PPO
from pettingzoo.utils import BaseParallelWrapper
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper
CONDITIONS={'null':(1.,1.,1.),'curvature':(.88,.88,1.),'lambda7':(.88,.88,7.)}

def rng_state():return (random.getstate(),np.random.get_state(),torch.get_rng_state())
def restore_rng(s):random.setstate(s[0]);np.random.set_state(s[1]);torch.set_rng_state(s[2])
class DemandRNG(BaseParallelWrapper):
    """Isolate legacy global-NumPy simulator draws from PPO/evaluation draws."""
    def __init__(self,env,seed):super().__init__(env);self.demand_rng=np.random.RandomState(seed).get_state()
    def invoke(self,fn,*args,**kwargs):
        outside=np.random.get_state();np.random.set_state(self.demand_rng)
        try:return fn(*args,**kwargs)
        finally:self.demand_rng=np.random.get_state();np.random.set_state(outside)
    def reset(self,seed=None,options=None):
        if seed is not None:self.demand_rng=np.random.RandomState(seed).get_state()
        return self.invoke(self.env.reset,seed=seed,options=options)
    def step(self,actions):return self.invoke(self.env.step,actions)

def build(w,condition,seed):
    state=rng_state()
    try:
        raw=MultiAgentNetInvMgmt();isolated=DemandRNG(raw,seed);diag=DiagnosticWrapper(isolated,scripted_nodes=['node_1'])
        a,b,l=CONDITIONS[condition];params={i:dict(alpha=a,beta=b,**{'lambda':l}) for i in diag.possible_agents}
        return ScaledBaseStockWrapper(CPTRewardWrapper(diag,params,reward_scale=1.,global_reward_weight=w),low_bound=0.,high_bound=500.)
    finally:restore_rng(state)

def vector(w,condition,seed):
    vec=ss.pettingzoo_env_to_vec_env_v1(ss.pad_action_space_v0(build(w,condition,seed)))
    vec=ss.concat_vec_envs_v1(vec,1,num_cpus=0,base_class='stable_baselines3')
    # Simulator RNG was initialized explicitly; SB3 seeds optimizer/action RNG normally.
    vec.seed=lambda s:None
    return vec

def evaluate(model,config,episodes=10):
    state=rng_state();env=build(config['w'],config['condition'],config['seed']);results=[];traceS=[]
    try:
        for ep in range(episodes):
            obs,_=env.reset(seed=config['seed']*10+ep);profit=0.;orders=[];targets=[]
            while env.agents:
                actions={}
                for a in env.agents:
                    action,_=model.predict(obs[a],deterministic=True);actions[a]=action[:env.action_space(a).shape[0]];targets.append(float(250*(np.clip(action[0],-1,1)+1)))
                obs,rewards,_,_,infos=env.step(actions);profit+=sum(infos[a]['true_reward'] for a in rewards)
                base=env.unwrapped.unwrapped_env;t=base.period-1
                for a in env.possible_agents:
                    node=env.unwrapped.agent_name_to_id[a]
                    orders += [float(base.R.loc[t,link]) for link in base.reorder_links if link[1]==node]
            base=env.unwrapped.unwrapped_env;demand=float(base.D.to_numpy().sum());lost=float(base.U.to_numpy().sum())
            results.append(dict(episode=ep,evaluation_seed=config['seed']*10+ep,profit=profit,mean_order=float(np.mean(orders)),lost_sales=lost/demand if demand else 0,trajectory_mean_S=float(np.mean(targets))))
        obs,_=env.reset(seed=config['seed']*10);resetS=[]
        for a in env.agents:
            action,_=model.predict(obs[a],deterministic=True);resetS.append(float(250*(action[0]+1)))
        summary={k:float(np.mean([r[k] for r in results])) for k in ['profit','mean_order','lost_sales','trajectory_mean_S']};summary['mean_S']=float(np.mean(resetS));summary['collapsed']=bool(summary['mean_S']<10 and summary['profit']<=-127.1)
        return dict(summary=summary,episodes=results)
    finally:env.close();restore_rng(state)

def manifest():
    paths=[Path(__file__),ROOT/'env/marl_or_gym_wrapper.py',ROOT/'env/diagnostic_wrapper.py',ROOT/'env/scaled_base_stock_wrapper.py',ROOT/'agents/cpt_wrapper.py',ROOT/'or-gym/or_gym/envs/supply_chain/network_management.py']
    import importlib.metadata as md
    return dict(python=sys.version,packages={p:md.version(p) for p in ['numpy','torch','stable-baselines3','supersuit','pettingzoo','gymnasium','pandas','scipy']},sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},counter='agent transitions; 5 lanes; rollout 128; round requested checkpoints up to completed rollout')
class StageComplete(Exception):
    pass

class HorizonPPO(PPO):
    def train(self):
        super().train()
        for checkpoint in self.protocol['checkpoints']:
            if checkpoint<=self.num_timesteps and checkpoint not in self.completed:
                self.completed.append(checkpoint)
                out=Path(self.output);out.mkdir(parents=True,exist_ok=True)
                if self.protocol.get('evaluate',True):
                    report=evaluate(self,self.protocol,10)
                    if checkpoint in [900000,1800000,3600000]:report['sensitivity_50']=evaluate(self,self.protocol,50)
                    report.update(requested_checkpoint=checkpoint,actual_agent_transitions=self.num_timesteps,joint_ticks=self.num_timesteps//5,elapsed_seconds=time.monotonic()-self.started)
                    (out/f'evaluation_{checkpoint}.json').write_text(json.dumps(report,indent=2))
                self.checkpoint(out/f'checkpoint_{checkpoint}')
        if getattr(self,'stop_after',0) and self.num_timesteps>=self.stop_after:
            raise StageComplete()
    def checkpoint(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        self.save(path/'model.zip')
        with (path/'environment.pkl').open('wb') as f:cloudpickle.dump(self.env,f)
        with (path/'rng.pkl').open('wb') as f:cloudpickle.dump(rng_state(),f)
        (path/'state.json').write_text(json.dumps(dict(num_timesteps=self.num_timesteps,completed=self.completed,protocol=self.protocol),indent=2))

def create(config,out):
    torch.set_num_threads(1);vec=vector(config['w'],config['condition'],config['seed'])
    m=HorizonPPO('MlpPolicy',vec,n_steps=128,batch_size=256,seed=config['seed'],verbose=0,device='cpu')
    m.protocol=config;m.output=str(out);m.completed=[];m.started=time.monotonic();return m

def resume(path,out):
    path=Path(path)
    with (path/'environment.pkl').open('rb') as f:env=cloudpickle.load(f)
    m=HorizonPPO.load(path/'model.zip',env=env,device='cpu',force_reset=False)
    with (path/'rng.pkl').open('rb') as f:state=cloudpickle.load(f)
    restore_rng(state);m.output=str(out);m.started=time.monotonic();return m

def smoke(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    config=dict(w=.6,condition='curvature',seed=100,checkpoints=[640,1280],evaluate=True,total_budget=1280,phase='smoke')
    m=create(config,out/'continuous');m.learn(total_timesteps=1280)
    expected={k:v.detach().clone() for k,v in m.policy.state_dict().items()}
    r=resume(out/'continuous/checkpoint_640',out/'resumed');r.learn(total_timesteps=1280-r.num_timesteps,reset_num_timesteps=False)
    exact=all(torch.equal(v,r.policy.state_dict()[k]) for k,v in expected.items())
    # Evaluation should not change any training RNG state.
    before=cloudpickle.dumps(rng_state());evaluate(r,config,2);after=cloudpickle.dumps(rng_state());rng_ok=before==after
    # Do not rely on serialized Torch byte representation for equality; compare states.
    state=rng_state();evaluate(r,config,2);post=rng_state();rng_ok=state[0]==post[0] and np.array_equal(state[1][1],post[1][1]) and state[1][2:]==post[1][2:] and torch.equal(state[2],post[2])
    check=dict(exact_resume_policy=exact,evaluation_rng_unchanged=rng_ok,actual_transitions=r.num_timesteps,manifest=manifest())
    (out/'smoke_result.json').write_text(json.dumps(check,indent=2));print(json.dumps(check),flush=True)
    assert exact and rng_ok
    # Measure post-smoke training throughput with production settings, no evaluation.
    cfg=dict(w=.6,condition='null',seed=101,checkpoints=[],evaluate=False,total_budget=20000,phase='timing')
    t=create(cfg,out/'timing');start=time.monotonic();t.learn(total_timesteps=20000);elapsed=time.monotonic()-start
    stats=dict(actual_transitions=t.num_timesteps,seconds=elapsed,transitions_per_second=t.num_timesteps/elapsed,base_420_training_cpu_hours=1512000000/(t.num_timesteps/elapsed)/3600,evaluation_overhead_excluded=True)
    (out/'timing.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume');ap.add_argument('--stop-after',type=int,default=0);ap.add_argument('--seed',type=int,default=100);ap.add_argument('--w',type=float,default=.6);ap.add_argument('--condition',choices=CONDITIONS,default='null');ap.add_argument('--budget',type=int,default=3600000);ap.add_argument('--phase',choices=['pilot','confirmation'],default='pilot');a=ap.parse_args()
    if a.smoke:return smoke(a.out)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    if a.resume:m=resume(a.resume,out)
    else:
        config=dict(w=a.w,condition=a.condition,seed=a.seed,checkpoints=list(range(150000,a.budget+1,150000)),total_budget=a.budget,phase=a.phase,evaluate=True)
        m=create(config,out)
    (out/'manifest.json').write_text(json.dumps(manifest(),indent=2));(out/'protocol.json').write_text(json.dumps(m.protocol,indent=2))
    m.stop_after=a.stop_after
    try:m.learn(total_timesteps=m.protocol['total_budget']-m.num_timesteps,reset_num_timesteps=(not a.resume))
    except StageComplete:print('STAGED',m.num_timesteps,flush=True);return
    print('COMPLETE',m.num_timesteps,flush=True)
if __name__=='__main__':main()
