"""SB3 2.8.0 PPO update with observational diagnostics only.
Update body copied from inspected installed PPO.train; additions do not draw randomness.
Gradients are norms of disjoint actor/critic/shared parameter groups of total PPO loss.
"""
from pathlib import Path
import json,gzip,hashlib,inspect
import numpy as np
import torch as th
from torch.nn import functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.callbacks import BaseCallback
from reward_diagnostics import moments
UPSTREAM_TRAIN_SHA256 = '1dd52af7dfa6cc043520013d6ef5ad3c4af542f303f6341e8abbf1932afa8f76'

def append_record(path,record):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(path,'at') as f:f.write(json.dumps(record,allow_nan=True)+'\n')

class RolloutDiagnostics(BaseCallback):
 def _on_rollout_start(self):self.samples=[]
 def _on_step(self):
  for lane,info in enumerate(self.locals['infos']):
   self.samples.append(dict(agent=f'node_{lane+2}',local=info['raw_local_reward'],global_reward=info['raw_global_reward'],blend=info['blended_reward'],centered=info['scaled_reward'],transformed=info['cpt_reward']))
  return True
 def _on_rollout_end(self):
  rows=[]
  for agent in ['pooled']+[f'node_{i}' for i in range(2,7)]:
   selected=self.samples if agent=='pooled' else [r for r in self.samples if r['agent']==agent]
   row={'agent':agent,'rewards':{k:moments([r[k] for r in selected]) for k in ['local','global_reward','blend','centered','transformed']}}
   for k in ['advantages','returns','rewards']:
    a=getattr(self.model.rollout_buffer,k);v=a if agent=='pooled' else a[:,int(agent[-1])-2]
    row[k+'_buffer']=moments(v)
   rows.append(row)
  append_record(Path(self.model.diagnostics_dir)/'rollouts.jsonl.gz',{'actual_transitions':self.num_timesteps,'scope':'rollout; population SD; pooled global reward repeats once per agent','groups':rows})

class InstrumentedPPO(PPO):
 def verify_upstream(self):
  assert hashlib.sha256(inspect.getsource(PPO.train).rstrip().encode()).hexdigest()==UPSTREAM_TRAIN_SHA256,'Installed PPO update differs from the audited implementation'
 def before_clip(self):
  groups={'actor':[],'critic':[],'shared':[]}
  for name,p in self.policy.named_parameters():
   if p.grad is None:continue
   group='critic' if ('value_net' in name) else ('actor' if ('policy_net' in name or 'action_net' in name or 'log_std' in name) else 'shared')
   groups[group].append(float(th.sum(p.grad.detach()**2)))
  sums={k:sum(v) for k,v in groups.items()};norm=float(np.sqrt(sum(sums.values())))
  actor_share=sums['actor']/sum(sums.values()) if sum(sums.values()) else 0.
  return {**{k+'_grad_norm':float(np.sqrt(v)) for k,v in sums.items()},'total_grad_norm':norm,'actor_grad_energy_share':float(actor_share),'gradient_clipped':bool(norm>self.max_grad_norm)}

 def after_clip(self):
  groups={'actor':[],'critic':[],'shared':[]}
  for name,p in self.policy.named_parameters():
   if p.grad is None:continue
   group='critic' if ('value_net' in name) else ('actor' if ('policy_net' in name or 'action_net' in name or 'log_std' in name) else 'shared')
   groups[group].append(float(th.sum(p.grad.detach()**2)))
  sums={k:sum(v) for k,v in groups.items()};total=sum(sums.values())
  return {**{'post_'+k+'_grad_norm':float(np.sqrt(v)) for k,v in sums.items()},'post_total_grad_norm':float(np.sqrt(total)),'post_actor_grad_energy_share':float(sums['actor']/total if total else 0.)}

 def update_optimizers(self):
  self._update_learning_rate(self.policy.optimizer)

 def zero_gradients(self):
  self.policy.optimizer.zero_grad()

 def step_optimizers(self):
  self.policy.optimizer.step()

 def clip_gradients(self):
  """Apply the upstream global clipping rule and report its pre-clip norm."""
  pre_norm=th.nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm)
  return dict(clip_mode='global',global_clip_pre_norm=float(pre_norm))

 def train(self) -> None:
     """
     Update policy using the currently gathered rollout buffer.
     """
     # Switch to train mode (this affects batch norm / dropout)
     self.policy.set_training_mode(True)
     # Update optimizer learning rate
     self.update_optimizers()
     # Compute current clip range
     clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
     # Optional: clip range for the value function
     if self.clip_range_vf is not None:
         clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

     diagnostic_batches = []
     entropy_losses = []
     pg_losses, value_losses = [], []
     clip_fractions = []

     continue_training = True
     # train for n_epochs epochs
     for epoch in range(self.n_epochs):
         approx_kl_divs = []
         # Do a complete pass on the rollout buffer
         for rollout_data in self.rollout_buffer.get(self.batch_size):
             actions = rollout_data.actions
             if isinstance(self.action_space, spaces.Discrete):
                 # Convert discrete action from float to long
                 actions = rollout_data.actions.long().flatten()

             values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
             values = values.flatten()
             # Normalize advantage
             advantages = rollout_data.advantages
             # Normalization does not make sense if mini batchsize == 1, see GH issue #325
             if self.normalize_advantage and len(advantages) > 1:
                 advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

             batch_diagnostic = dict(raw_advantage=moments(rollout_data.advantages.detach().cpu().numpy()),normalized_advantage=moments(advantages.detach().cpu().numpy()),return_target=moments(rollout_data.returns.detach().cpu().numpy()))

             # ratio between old and new policy, should be one at the first iteration
             ratio = th.exp(log_prob - rollout_data.old_log_prob)

             # clipped surrogate loss
             policy_loss_1 = advantages * ratio
             policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
             policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

             # Logging
             pg_losses.append(policy_loss.item())
             clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
             clip_fractions.append(clip_fraction)

             if self.clip_range_vf is None:
                 # No clipping
                 values_pred = values
             else:
                 # Clip the difference between old and new value
                 # NOTE: this depends on the reward scaling
                 values_pred = rollout_data.old_values + th.clamp(
                     values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                 )
             # Value loss using the TD(gae_lambda) target
             value_loss = F.mse_loss(rollout_data.returns, values_pred)
             value_losses.append(value_loss.item())

             # Entropy loss favor exploration
             if entropy is None:
                 # Approximate entropy when no analytical form
                 entropy_loss = -th.mean(-log_prob)
             else:
                 entropy_loss = -th.mean(entropy)

             entropy_losses.append(entropy_loss.item())

             loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

             # Calculate approximate form of reverse KL Divergence for early stopping
             # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
             # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
             # and Schulman blog: http://joschu.net/blog/kl-approx.html
             with th.no_grad():
                 log_ratio = log_prob - rollout_data.old_log_prob
                 approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                 approx_kl_divs.append(approx_kl_div)

             if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                 continue_training = False
                 if self.verbose >= 1:
                     print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                 break

             # Optimization step
             self.zero_gradients()
             loss.backward()
             batch_diagnostic.update(self.before_clip())
             batch_diagnostic.update(value_loss=float(value_loss.item()),approx_kl=float(approx_kl_div),clip_fraction=clip_fraction,entropy=float(-entropy_loss.item()),epoch=epoch)
             diagnostic_batches.append(batch_diagnostic)
             # Clip grad norm. The default method is exactly the upstream global
             # operation; subclasses may replace it with a declared intervention.
             batch_diagnostic.update(self.clip_gradients())
             batch_diagnostic.update(self.after_clip())
             self.step_optimizers()

         self._n_updates += 1
         if not continue_training:
             break

     explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

     # Logs
     self.logger.record("train/entropy_loss", np.mean(entropy_losses))
     self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
     self.logger.record("train/value_loss", np.mean(value_losses))
     self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
     self.logger.record("train/clip_fraction", np.mean(clip_fractions))
     self.logger.record("train/loss", loss.item())
     self.logger.record("train/explained_variance", explained_var)
     if hasattr(self.policy, "log_std"):
         self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

     self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
     self.logger.record("train/clip_range", clip_range)
     if self.clip_range_vf is not None:
         self.logger.record("train/clip_range_vf", clip_range_vf)

     append_record(Path(self.diagnostics_dir)/'updates.jsonl.gz',dict(actual_transitions=self.num_timesteps,n_updates=self._n_updates,scope='actual optimizer minibatches; advantages use upstream sample-SD normalization, diagnostics population SD',explained_variance=float(explained_var),gradient_clipping_frequency=float(np.mean([r['gradient_clipped'] for r in diagnostic_batches])),minibatches=diagnostic_batches))

class GroupwiseClipPPO(InstrumentedPPO):
 """PPO intervention that prevents critic gradients from shrinking actor updates.

 The loss, optimizer, rollout data and maximum norm are unchanged. Actor and
 critic parameter groups are clipped independently immediately before the same
 optimizer step. The flat-observation MlpPolicy used in this experiment has no
 trainable shared feature parameters; execution aborts if that assumption fails.
 """
 def parameter_groups(self):
  groups={'actor':[],'critic':[],'shared':[]}
  for name,p in self.policy.named_parameters():
   if not p.requires_grad:continue
   group='critic' if ('value_net' in name) else ('actor' if ('policy_net' in name or 'action_net' in name or 'log_std' in name) else 'shared')
   groups[group].append(p)
  if groups['shared']:
   names=[name for name,p in self.policy.named_parameters() if p.requires_grad and all(p is not q for q in groups['actor']+groups['critic'])]
   raise RuntimeError(f'Groupwise clipping requires no trainable shared parameters; found {names}')
  if not groups['actor'] or not groups['critic']:
   raise RuntimeError('Actor/critic parameter grouping is empty')
  if len({id(p) for p in groups['actor']+groups['critic']}) != len(groups['actor'])+len(groups['critic']):
   raise RuntimeError('Actor/critic parameter groups overlap')
  return groups

 def clip_gradients(self):
  groups=self.parameter_groups()
  actor_pre=th.nn.utils.clip_grad_norm_(groups['actor'],self.max_grad_norm)
  critic_pre=th.nn.utils.clip_grad_norm_(groups['critic'],self.max_grad_norm)
  return dict(
   clip_mode='groupwise',
   actor_clip_pre_norm=float(actor_pre),
   critic_clip_pre_norm=float(critic_pre),
   actor_group_clipped=bool(float(actor_pre)>self.max_grad_norm),
   critic_group_clipped=bool(float(critic_pre)>self.max_grad_norm),
  )

class SeparateNoClipPPO(GroupwiseClipPPO):
 """PPO with disjoint actor/critic Adam optimizers and no gradient clipping.

 The declared flat-observation MLP must have no trainable shared parameters.
 With disjoint parameters, the two Adam instances remove the optimizer/clip
 coupling while preserving the upstream losses, schedules, epochs and batches.
 """
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  groups=self.parameter_groups()
  optimizer_class=self.policy.optimizer_class
  optimizer_kwargs=self.policy.optimizer_kwargs
  lr=self.lr_schedule(1)
  self.actor_optimizer=optimizer_class(groups['actor'],lr=lr,**optimizer_kwargs)
  self.critic_optimizer=optimizer_class(groups['critic'],lr=lr,**optimizer_kwargs)

 def update_optimizers(self):
  self._update_learning_rate([self.actor_optimizer,self.critic_optimizer])

 def zero_gradients(self):
  self.actor_optimizer.zero_grad();self.critic_optimizer.zero_grad()

 def clip_gradients(self):
  return dict(clip_mode='separate_optimizers_no_clip')

 def step_optimizers(self):
  self.actor_optimizer.step();self.critic_optimizer.step()
