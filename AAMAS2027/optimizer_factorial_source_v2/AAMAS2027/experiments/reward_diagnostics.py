"""Pure numerical diagnostics; no sampling and no RNG consumption."""
import numpy as np

def moments(values):
 x=np.asarray(values,dtype=float).reshape(-1);x=x[np.isfinite(x)]
 if not len(x):return {'count':0}
 return dict(count=len(x),mean=float(x.mean()),std=float(x.std(ddof=0)),rms=float(np.sqrt(np.mean(x*x))),quantiles=dict(zip(['p01','p10','p25','p50','p75','p90','p99'],np.quantile(x,[.01,.1,.25,.5,.75,.9,.99]).tolist())),positive=float(np.mean(x>0)),zero=float(np.mean(x==0)),negative=float(np.mean(x<0)))

def phi(u):
 u=np.asarray(u,dtype=float);return np.sign(u)*np.abs(u)**.88

def transform(u,arm='B',c=1.):
 if not np.isfinite(c) or c<=0:raise ValueError('c must be positive and finite')
 u=np.asarray(u,dtype=float);base=phi(u);asym=np.where(u<0,7*base,base)
 if arm=='B':return base
 if arm=='A':return asym
 if arm=='S':return c*base
 if arm=='N':return asym/c
 raise ValueError(arm)

def scale_constant(u):
 den=np.mean(phi(u)**2)
 if not den>0:raise ValueError('calibration stream has zero RMS')
 return float(np.sqrt(np.mean(transform(u,'A')**2)/den))
