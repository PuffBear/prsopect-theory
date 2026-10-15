"""Remote runtime integration tests: instrumentation must leave learning unchanged."""
from pathlib import Path
import argparse,json,gzip
import numpy as np,torch
import horizon_supply as h
from scale_control import create,build,vector
from instrumented_ppo import RolloutDiagnostics
from reward_diagnostics import transform
p=argparse.ArgumentParser();p.add_argument('--out',required=True);out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
cfg=dict(w=.6,condition='curvature',seed=103,checkpoints=[],evaluate=False,total_budget=1280,phase='smoke')
torch.set_num_threads(1)
a=h.HorizonPPO('MlpPolicy',vector(.6,'B',103),n_steps=128,batch_size=256,seed=103,verbose=0,device='cpu');a.protocol=cfg;a.completed=[];a.output=str(out/'plain');a.learn(total_timesteps=1280);expected={k:v.detach().clone() for k,v in a.policy.state_dict().items()};state=h.rng_state()
b=create(out/'instrumented',.6,'B',103,budget=1280,phase='smoke');b.learn(total_timesteps=1280,callback=RolloutDiagnostics())
assert all(torch.equal(v,b.policy.state_dict()[k]) for k,v in expected.items()),'Instrumentation changed policy'
post=h.rng_state();assert state[0]==post[0] and np.array_equal(state[1][1],post[1][1]) and state[1][2:]==post[1][2:] and torch.equal(state[2],post[2])
for arm in ['B','A','S','N']:
 e=build(.6,arm,103,2.5);obs,_=e.reset(seed=42)
 for _ in range(5):
  obs,rewards,_,_,infos=e.step({ag:np.zeros(e.action_space(ag).shape) for ag in e.agents})
  for ag,r in rewards.items():np.testing.assert_allclose(r,transform(infos[ag]['scaled_reward'],arm,2.5))
 e.close()
for file in ['rollouts.jsonl.gz','updates.jsonl.gz']:
 with gzip.open(out/'instrumented/diagnostics'/file,'rt') as f:rows=[json.loads(x) for x in f]
 assert len(rows)==2
 if file.startswith('rollouts'):
  assert rows[0]['groups'][0]['rewards']['blend']['count']==640
  assert len(rows[0]['groups'])==6
 else:assert len(rows[0]['minibatches'])==30
# A cloudpickle roundtrip must not detach demand sampling from the seeded stream.
import cloudpickle
from stable_baselines3 import PPO
v1=vector(.6,'B',105);v2=vector(.6,'B',105)
x1=v1.reset();np.random.random(37);x2=v2.reset();np.testing.assert_array_equal(x1,x2)
for _ in range(15):
 actions=np.zeros((5,3));o1,r1,_,_=v1.step(actions);np.random.random(37);o2,r2,_,_=v2.step(actions)
 np.testing.assert_array_equal(o1,o2);np.testing.assert_array_equal(r1,r2)
# Different explicit seeds must change observations under the same actions.
v3=vector(.6,'B',106);v3.reset();o3,_,_,_=v3.step(np.zeros((5,3)))
v4=vector(.6,'B',105);v4.reset();o4,_,_,_=v4.step(np.zeros((5,3)));assert not np.array_equal(o3,o4)
(out/'test_result.json').write_text(json.dumps(dict(instrumentation_policy_identical=True,training_rng_identical=True,four_arm_wrapper_equivalence=True,two_rollouts=True,thirty_minibatches_per_update=True,vectorized_same_seed_reproducible=True,different_seed_changes_demand=True),indent=2));print('PASS instrumentation integration',flush=True)
