/* shared rendering helpers + verbatim run-archive data, used by every page */
document.addEventListener('DOMContentLoaded', function(){
  renderMathInElement(document.body, {delimiters:[
    {left:'$$', right:'$$', display:true},
    {left:'$', right:'$', display:false}
  ], throwOnError:false});
});

/* ---------- shared style ---------- */
const C = {paper:'#f7f4ed', paper2:'#f1ecdf', ink:'#1a1712', muted:'#6f6757',
           rule:'#d9d2c2', accent:'#a33327', blue:'#2b4f8a', gold:'#8a6d1f'};
const FONT = {family:'IBM Plex Mono, monospace', size:12, color:C.muted};
function baseLayout(extra){
  return Object.assign({
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:FONT, margin:{l:62,r:30,t:28,b:54},
    xaxis:{gridcolor:C.rule, zerolinecolor:C.rule, linecolor:C.muted, ticks:'outside', tickcolor:C.muted},
    yaxis:{gridcolor:C.rule, zerolinecolor:C.rule, linecolor:C.muted, ticks:'outside', tickcolor:C.muted},
    hoverlabel:{font:{family:'IBM Plex Mono, monospace', size:12}}
  }, extra||{});
}
const CFG = {displayModeBar:false, responsive:true};
function wilson(k,n){
  const p=k/n, z=1.96, d=1+z*z/n;
  const ctr=(p+z*z/(2*n))/d, hw=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;
  return [Math.max(0,ctr-hw), Math.min(1,ctr+hw)];
}
const sig = z => 1/(1+Math.exp(-z));
function linspace(a,b,n){const o=[];for(let i=0;i<n;i++)o.push(a+(b-a)*i/(n-1));return o;}
function erf(x){const s=x<0?-1:1;x=Math.abs(x);const t=1/(1+0.3275911*x);
  const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x);
  return s*y;}
const Phi = z => 0.5*(1+erf(z/Math.SQRT2));

/* ---------- data (verbatim from run archives) ----------
   PPO supply-chain dense lambda sweep (or-gym, alpha=beta=0.88):
   lambda in {1,3,7} from Exp B (20 seeds x 5 w-cells), {2,4,5,6} from the
   completed dense sweep. Per-cell k/n are collapse counts. */
const expA = {w:[0.2,0.3,0.4,0.5,0.6,0.7,0.8], k:[20,20,20,19,16,8,0], n:20, wc:0.666, slope:22.046};
const expN = {w:[0.0,0.2,0.4,0.5,0.6,0.7,0.8,0.9,1.0], k:[20,20,20,20,17,10,0,0,0], n:20, wc:0.685, slope:27.349};

/* full 7-point PPO lambda sweep: per-cell collapse counts (k of n=20) */
const expB = {
  1:{w:[0.5,0.65,0.75,0.85,0.95], k:[17,3,2,1,0],  n:20, wc:0.586, slope:15.621},
  2:{w:[0.5,0.65,0.75,0.85,0.95], k:[20,17,6,0,0],  n:20, wc:0.715, slope:29.793},
  3:{w:[0.5,0.65,0.75,0.85,0.95], k:[20,19,15,1,0], n:20, wc:0.774, slope:32.775},
  4:{w:[0.5,0.65,0.75,0.85,0.95], k:[20,20,18,3,0], n:20, wc:0.806, slope:40.667},
  5:{w:[0.5,0.65,0.75,0.85,0.95], k:[20,20,20,3,0], n:20, wc:0.839, slope:60},
  6:{w:[0.5,0.65,0.75,0.85,0.95], k:[20,20,17,10,0],n:20, wc:0.836, slope:26.828},
  7:{w:[0.5,0.65,0.75,0.85,0.95], k:[20,20,20,11,0],n:20, wc:0.851, slope:60}
};
/* per-lambda bootstrap w_crit + 95% CI (n=2000), PPO supply chain, all 7 points */
const wcB = {lam:[1,2,3,4,5,6,7],
  wc:[0.586,0.715,0.774,0.806,0.839,0.836,0.851],
  lo:[0.539,0.687,0.751,0.766,0.788,0.806,0.845],
  hi:[0.624,0.745,0.798,0.841,0.845,0.860,0.858]};

/* cross-environment / cross-algorithm w_crit(lambda) (Firth logit, bootstrap n=2000) */
const xenv = {
  ppoSupply:  {lam:[1,2,3,4,5,6,7], wc:[0.585,0.714,0.775,0.805,0.820,0.835,0.854],
               lo:[0.538,0.688,0.751,0.782,0.802,0.808,0.834], hi:[0.622,0.739,0.797,0.828,0.835,0.862,0.873]},
  mappoSupply:{lam:[1,2,3,4,5,6,7], wc:[0.796,0.747,0.795,0.800,0.851,0.861,0.890],
               lo:[0.741,0.708,0.765,0.772,0.822,0.832,0.866], hi:[0.866,0.786,0.826,0.829,0.879,0.894,0.914]},
  mappoMPE:   {lam:[1,2,3,4,5,6,7], wc:[0.756,0.741,0.812,0.771,0.863,0.924,0.956],
               lo:[0.691,0.690,0.761,0.722,0.811,0.859,0.893], hi:[0.829,0.803,0.868,0.825,0.926,1.033,1.059]}
};
