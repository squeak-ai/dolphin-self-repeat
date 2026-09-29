"""m03c — exact-repeat prevalence and per-feature similarity under the 60-s block null (the defensible numbers)."""
import numpy as np, pandas as pd
from m00_lib import *
rng = np.random.default_rng(5)
c2 = pd.read_csv(os.path.join(OUT, 'calls2.csv'))
FEATS = {'dur': 'dur', 'pulses': 'f_number_of_pulses', 'f_max': 'f_maximum_frequency', 'f_start': 'f_start_frequency',
         'f_end': 'f_end_frequency', 'loud': 'f_maximum_loudness', 'pulse_len': 'f_length_of_pulse'}
X = c2[list(FEATS.values())].copy(); X.columns = list(FEATS)
X['dur'] = np.log(X.dur.clip(0.05)); X['pulse_len'] = np.log(X.pulse_len.clip(1))
ok = X.notna().all(1).values; Z = ((X - X.mean()) / X.std()).values
groups = [(s, c, g.index.values, g.t0.values) for (s, c), g in c2[ok].groupby(['session', 'caller']) if len(g) >= 5]
NIT = 200; B = 60
def block_perm(t):
    p = np.arange(len(t)); b = (t // B).astype(int)
    for u in np.unique(b):
        m = np.where(b == u)[0]
        if len(m) > 1: p[m] = m[rng.permutation(len(m))]
    return p
# threshold p5 of all within-caller-session distances (same as m03)
D = []
for s, c, ix, t in groups:
    i, j = np.triu_indices(len(t), 1); D.append(np.sqrt(((Z[ix][i] - Z[ix][j]) ** 2).sum(1)))
thr = np.percentile(np.concatenate(D), 5)
LAGB = [(5, 10), (10, 20), (20, 30), (30, 60)]
ex_obs = np.zeros(len(LAGB)); ex_null = np.zeros((NIT, len(LAGB))); po = 0; pn = np.zeros(NIT); nc = 0
fo = np.zeros(len(FEATS)); fn = np.zeros((NIT, len(FEATS))); npf = 0
per = []
for s, c, ix, t in groups:
    Zg = Z[ix]; n = len(t); i, j = np.triu_indices(n, 1); lag = t[j] - t[i]; m520 = (lag >= 5) & (lag < 20)
    def cnt(Zx):
        d = np.sqrt(((Zx[i] - Zx[j]) ** 2).sum(1)); e = d <= thr
        out = np.array([(e & (lag >= a) & (lag < b)).sum() for a, b in LAGB])
        pm = e & (lag >= 5) & (lag < 30); part = len(np.unique(np.concatenate([i[pm], j[pm]])))
        f = np.abs(Zx[i[m520]] - Zx[j[m520]]).sum(0) if m520.any() else np.zeros(len(FEATS))
        return out, part, f
    o, p, f = cnt(Zg); ex_obs += o; po += p; nc += n; fo += f; npf += m520.sum()
    nn = np.zeros((NIT, len(LAGB)))
    for it in range(NIT):
        o, p, f = cnt(Zg[block_perm(t)]); ex_null[it] += o; pn[it] += p; fn[it] += f; nn[it] = o
    if ex_obs is not None and (lag < 30).sum() >= 10:
        per.append(dict(session=int(s), caller=c, obs=int(cnt(Zg)[0][:3].sum()), null=float(nn[:, :3].sum(1).mean())))
per = pd.DataFrame(per)
res = dict(threshold=float(thr), lags=[f'{a}–{b}' for a, b in LAGB], obs=ex_obs.tolist(), null=ex_null.mean(0).tolist(), sd=ex_null.std(0).tolist(),
           ratio=(ex_obs / ex_null.mean(0)).tolist(), z=((ex_obs - ex_null.mean(0)) / ex_null.std(0)).tolist(),
           participation=dict(obs=int(po), null=float(pn.mean()), ncalls=int(nc), obs_pct=100 * po / nc, null_pct=100 * pn.mean() / nc, excess_pct=100 * (po - pn.mean()) / nc),
           per_feature_5_20=dict(feature=list(FEATS), ratio=(fo / fn.mean(0)).round(4).tolist(), z=((fo - fn.mean(0)) / fn.std(0)).round(1).tolist()),
           per_caller=dict(n=len(per), gt_null=int((per.obs > per.null).sum()), excess_pairs=int(per.obs.sum() - per.null.sum())))
print('exact pairs (≤p5) under 60-s block null:')
for k in range(len(LAGB)): print(f"  {res['lags'][k]:>6} s obs {int(ex_obs[k]):5d} null {ex_null.mean(0)[k]:7.1f} ± {ex_null.std(0)[k]:4.1f} ratio {res['ratio'][k]:.3f} z {res['z'][k]:+.1f}")
p = res['participation']; print(f"participation 5–30 s: {p['obs_pct']:.1f}% vs null {p['null_pct']:.1f}% → excess {p['excess_pct']:.2f} pct points; caller-sessions above null {res['per_caller']['gt_null']}/{res['per_caller']['n']}")
print('per-feature |Δ| 5–20 s vs block null:', dict(zip(FEATS, res['per_feature_5_20']['ratio'])))
dump('m03c_exact_local', res)
