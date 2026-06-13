#!/usr/bin/env python3
"""
fit_scaling_laws.py — pre-committed law-fitting pipeline for the dense lambda sweep.

FROZEN BEFORE SWEEP RESULTS. Decision rule (stated in the blog, Section 06):
  a law survives iff every per-lambda bootstrap w_crit falls inside its pointwise
  95% band; the winner must beat the runner-up by dAICc >= 4. If no law survives,
  that is the result.

Usage:
  python fit_scaling_laws.py sweep_raw_data.csv [--boot 2000] [--out results/]

Input CSV schema (same as expB_reduced_raw_data.csv):
  run_idx, w, lambda_loss, seed, mean_S, profit, lost_sales, collapsed, ...

Pipeline:
  1. per-lambda logistic regression of collapsed on w  -> w_crit = -b0/b1
  2. run-level bootstrap (n=--boot, resample within lambda) -> CI per lambda
  3. fit all 20 laws by CI-weighted least squares
  4. rank by AICc + leave-one-lambda-out CV; apply survival rule
"""
import sys, argparse, warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm
import statsmodels.api as sm

S = lambda z: 1/(1+np.exp(-z))

# ---------------------------------------------------------------- w_crit per lambda
def wcrit_logistic(df):
    X = sm.add_constant(df['w'].values)
    y = df['collapsed'].astype(int).values
    if y.min() == y.max():
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = sm.Logit(y, X).fit(disp=0, maxiter=200)
        b0, b1 = fit.params
        return -b0/b1
    except Exception:
        return np.nan

def bootstrap_wcrit(df, n_boot, rng):
    runs = df.reset_index(drop=True)
    est = wcrit_logistic(runs)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(runs), len(runs))
        wc = wcrit_logistic(runs.iloc[idx])
        if np.isfinite(wc) and -1 < wc < 2:
            boots.append(wc)
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    med = np.median(boots)
    return (est if np.isfinite(est) else med), lo, hi

# ---------------------------------------------------------------- the 20 laws
def boxcox_term(l, g):
    return np.where(np.abs(g) < 1e-8, np.log(l), (np.power(l, g) - 1)/g)

LAWS = [
 ("L1  log",            2, lambda p,l: p[0]+p[1]*np.log(l),                          [0.6,0.12]),
 ("L2  power-shift",    3, lambda p,l: p[0]+p[1]*np.power(np.maximum(l-1,1e-12),p[2]),[0.59,0.15,0.3]),
 ("L3  odds-power",     2, lambda p,l: S(p[0]+p[1]*np.log(l)),                       [0.4,0.7]),
 ("L4  Mobius",         3, lambda p,l: (p[0]+p[1]*l)/(1+p[2]*l),                     [0.0,1.7,1.8]),
 ("L5  sat-exp",        3, lambda p,l: p[0]-(p[0]-p[1])*np.exp(-p[2]*(l-1)),         [0.86,0.59,0.58]),
 ("L6  Michaelis-Menten",2,lambda p,l: p[0]*l/(p[1]+l),                              [0.92,0.57]),
 ("L7  MM-shifted",     3, lambda p,l: p[0]+(p[1]-p[0])*(l-1)/(p[2]+l-1),            [0.59,0.92,1.5]),
 ("L8  sqrt",           2, lambda p,l: p[0]+p[1]*np.sqrt(np.maximum(l-1,0)),         [0.62,0.10]),
 ("L9  linear",         2, lambda p,l: p[0]+p[1]*(l-1),                              [0.67,0.03]),
 ("L10 quad-log",       3, lambda p,l: p[0]+p[1]*np.log(l)+p[2]*np.log(l)**2,        [0.59,0.22,-0.04]),
 ("L11 tanh",           3, lambda p,l: p[0]+p[1]*np.tanh(p[2]*(l-1)),                [0.59,0.27,0.44]),
 ("L12 arctan",         3, lambda p,l: p[0]+p[1]*np.arctan(p[2]*(l-1)),              [0.59,0.20,0.70]),
 ("L13 Gompertz",       3, lambda p,l: p[0]*np.exp(-p[1]*np.exp(-p[2]*l)),           [0.86,0.73,0.66]),
 ("L14 probit-log",     2, lambda p,l: norm.cdf(p[0]+p[1]*np.log(l)),                [0.26,0.40]),
 ("L15 odds-linear",    2, lambda p,l: (p[0]+p[1]*l)/(1+p[0]+p[1]*l),                [0.85,0.71]),
 ("L16 stretched-exp",  4, lambda p,l: p[0]-(p[0]-p[1])*np.exp(-p[2]*np.power(np.maximum(l-1,1e-12),p[3])),[0.87,0.59,0.61,0.79]),
 ("L17 pure-power",     2, lambda p,l: p[0]*np.power(l,p[1]),                        [0.63,0.15]),
 ("L18 reciprocal",     2, lambda p,l: p[0]-p[1]/l,                                  [0.90,0.32]),
 ("L19 neg-power",      3, lambda p,l: p[0]+p[1]*(1-np.power(l,-p[2])),              [0.59,0.37,0.64]),
 ("L20 Box-Cox",        3, lambda p,l: p[0]+p[1]*boxcox_term(l,p[2]),                [0.59,0.24,-0.64]),
]

def fit_law(f, p0, lam, wc, sigma):
    res = least_squares(lambda p: (f(p, lam) - wc)/sigma, p0, max_nfev=50000)
    k = len(p0); n = len(lam)
    chi2 = float(np.sum(res.fun**2))
    # Gaussian quasi-likelihood AICc
    aic = chi2 + 2*k
    aicc = aic + (2*k*(k+1)/(n-k-1)) if n-k-1 > 0 else np.inf
    return res.x, chi2, aicc

def loo_cv(f, p0, lam, wc, sigma):
    errs = []
    for i in range(len(lam)):
        m = np.ones(len(lam), bool); m[i] = False
        try:
            p,_,_ = fit_law(f, p0, lam[m], wc[m], sigma[m])
            errs.append(((f(p, lam[i:i+1])[0]-wc[i])/sigma[i])**2)
        except Exception:
            errs.append(np.inf)
    return float(np.mean(errs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv'); ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.csv)
    if df['collapsed'].dtype == object:
        df['collapsed'] = df['collapsed'].astype(str).str.lower().isin(['true','1'])

    rows = []
    for lam, g in sorted(df.groupby('lambda_loss')):
        est, lo, hi = bootstrap_wcrit(g, args.boot, rng)
        rows.append(dict(lam=lam, wcrit=est, lo=lo, hi=hi, n=len(g)))
        print(f"lambda={lam:<6g} w_crit={est:.4f}  CI=[{lo:.4f}, {hi:.4f}]  n={len(g)}")
    pts = pd.DataFrame(rows).dropna()
    lam = pts['lam'].values; wc = pts['wcrit'].values
    sigma = (pts['hi'].values - pts['lo'].values)/3.92
    sigma = np.maximum(sigma, 1e-3)

    out = []
    for name, k, f, p0 in LAWS:
        try:
            p, chi2, aicc = fit_law(f, p0, lam, wc, sigma)
            pred = f(p, lam)
            inside = bool(np.all((pred >= pts['lo'].values) & (pred <= pts['hi'].values)))
            cv = loo_cv(f, p0, lam, wc, sigma)
            out.append(dict(law=name, k=k, chi2=chi2, AICc=aicc, LOO=cv,
                            survives_CI=inside, params=np.round(p,4).tolist(),
                            pred_l10=float(f(p, np.array([10.]))[0]),
                            pred_l20=float(f(p, np.array([20.]))[0])))
        except Exception as e:
            out.append(dict(law=name, k=k, chi2=np.nan, AICc=np.inf, LOO=np.inf,
                            survives_CI=False, params=str(e)[:40], pred_l10=np.nan, pred_l20=np.nan))
    res = pd.DataFrame(out).sort_values('AICc')
    pd.set_option('display.width', 200)
    print('\n' + res.to_string(index=False))

    surv = res[res.survives_CI].sort_values('AICc')
    print('\n==== SURVIVAL RULE ====')
    if len(surv) == 0:
        print('NO LAW SURVIVES. That is the result.')
    else:
        d = surv.iloc[1]['AICc'] - surv.iloc[0]['AICc'] if len(surv) > 1 else np.inf
        print(f"survivors: {list(surv.law)}")
        print(f"leader: {surv.iloc[0]['law']}  dAICc over runner-up = {d:.2f} "
              f"({'WINNER (>=4)' if d >= 4 else 'NOT decisive (<4) — report finalists'})")
    res.to_csv('law_fits_results.csv', index=False)
    pts.to_csv('wcrit_by_lambda.csv', index=False)
    print("\nwrote law_fits_results.csv, wcrit_by_lambda.csv")

if __name__ == '__main__':
    main()
