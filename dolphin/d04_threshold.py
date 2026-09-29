"""d04b: threshold sweep of the exact-repeat excess COUNT.

The p5 DTW-abs threshold is a convention. A stricter threshold raises the excess *ratio* (p1: 1.64x) but catches fewer
repeats; a looser one admits more chance matches. The excess *count* (observed - null) should rise with the threshold
while genuine repeats are still being admitted and plateau once only chance pairs are added. The plateau estimates the
total number of detectable repeat pairs / participating whistles, i.e. bout prevalence, independent of the p5 choice.

Sweep percentiles 1..20 of the pairwise DTW-abs distribution; count exact non-overlapping pairs within 10 s, observed
vs within-minute-file onset permutation (headline) and session permutation (for reference), 50 draws each, and the
excess whistles participating. Also per-class excess pairs (both whistles same class) at each threshold.
"""
import numpy as np, json, time
from d04_lib import *

NDRAW = 50
df, F = load(); rng = np.random.default_rng(4242)
Da = np.load(os.path.join(OUT, 'dtw_abs.npy')); Ds = np.load(os.path.join(OUT, 'dtw_shape.npy'))
P = Pairs(df, Da, Ds)
PCTS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50]
thrs = {p: P.thr(p) for p in PCTS}
cls = df.cls.astype(str).values; same = cls[P.I] == cls[P.J]; cls_i = cls[P.I]
CLASSES = ['rise', 'fall', 'arch', 'U']


def counts(t0v, t1v):
    lag, gap, ov = P.geometry(t0v, t1v)
    win = (lag <= 10) & ~ov
    out = []
    for p in PCTS:
        ex = (P.da <= thrs[p]) & win
        out.append(dict(pairs=int(ex.sum()), part=P.participation(ex),
                        by_class=[int((ex & same & (cls_i == c)).sum()) for c in CLASSES]))
    return out


obs = counts(P.t0, P.t1)
nulls = {}
for name, groups in [('session', P.sess_groups), ('file', P.file_groups)]:
    tic = time.time(); nulls[name] = [counts(*P.permuted(rng, groups)) for _ in range(NDRAW)]
    print(f'{name} null: {NDRAW} draws in {time.time()-tic:.1f}s')

rows = []
print(f'{"pct":>4s} {"thr":>6s} | {"obs":>5s} {"file-null":>10s} {"excess":>7s} {"ratio":>5s} {"z":>5s} | {"part":>5s} {"null":>6s} {"excess":>6s} {"%mod":>5s} | session: excess ratio')
for k, p in enumerate(PCTS):
    r = dict(pct=p, thr=thrs[p], obs_pairs=obs[k]['pairs'], obs_part=obs[k]['part'], obs_by_class=obs[k]['by_class'])
    for name in nulls:
        v = np.array([d[k]['pairs'] for d in nulls[name]], float); w = np.array([d[k]['part'] for d in nulls[name]], float)
        bc = np.array([d[k]['by_class'] for d in nulls[name]], float)
        r[f'null_pairs_{name}'] = v.mean(); r[f'null_pairs_{name}_sd'] = v.std(); r[f'excess_pairs_{name}'] = obs[k]['pairs'] - v.mean()
        r[f'ratio_{name}'] = obs[k]['pairs'] / v.mean(); r[f'z_{name}'] = (obs[k]['pairs'] - v.mean()) / v.std()
        r[f'null_part_{name}'] = w.mean(); r[f'null_part_{name}_sd'] = w.std(); r[f'excess_part_{name}'] = obs[k]['part'] - w.mean()
        r[f'excess_part_frac_{name}'] = (obs[k]['part'] - w.mean()) / P.n_mod
        r[f'excess_by_class_{name}'] = (np.array(obs[k]['by_class']) - bc.mean(0)).tolist()
        r[f'ratio_by_class_{name}'] = (np.array(obs[k]['by_class']) / bc.mean(0)).tolist()
    rows.append(r)
    print(f'{p:4d} {thrs[p]:6.2f} | {r["obs_pairs"]:5d} {r["null_pairs_file"]:7.0f}±{r["null_pairs_file_sd"]:3.0f} {r["excess_pairs_file"]:+7.0f} {r["ratio_file"]:5.2f} {r["z_file"]:5.1f} | '
          f'{r["obs_part"]:5d} {r["null_part_file"]:6.0f} {r["excess_part_file"]:+6.0f} {r["excess_part_frac_file"]:5.1%} | {r["excess_pairs_session"]:+6.0f} {r["ratio_session"]:.2f}')
print('\nper-class excess pairs (file null), rise/fall/arch/U:')
for r in rows:
    print(f'  p{r["pct"]:<3d}', ' '.join(f'{c}:{e:+5.0f} ({q:.2f}x)' for c, e, q in zip(CLASSES, r['excess_by_class_file'], r['ratio_by_class_file'])))

# marginal excess per percentile point (the slope of the excess curve); the plateau is where it reaches the noise floor
ex = np.array([r['excess_pairs_file'] for r in rows]); sd = np.array([r['null_pairs_file_sd'] for r in rows]); pc = np.array(PCTS, float)
inc = np.diff(ex) / np.diff(pc); inc_sd = np.sqrt(sd[1:] ** 2 + sd[:-1] ** 2) / np.diff(pc)
for k in range(len(inc)): rows[k + 1]['marginal_excess_per_pct'] = float(inc[k]); rows[k + 1]['marginal_excess_per_pct_sd'] = float(inc_sd[k])
rows[0]['marginal_excess_per_pct'] = float(ex[0] / pc[0]); rows[0]['marginal_excess_per_pct_sd'] = float(sd[0] / pc[0])
ipk = int(np.argmax([r['excess_part_file'] for r in rows]))
res = dict(n_draws=NDRAW, n_mod=P.n_mod, n_pairs=int(len(P.I)), classes=CLASSES, rows=rows,
           summary=dict(excess_pairs_p5=float(ex[PCTS.index(5)]), excess_pairs_p20=float(ex[PCTS.index(20)]), excess_pairs_max=float(ex.max()), pct_max=PCTS[int(np.argmax(ex))],
                        part_peak_pct=PCTS[ipk], part_peak_excess=float(rows[ipk]['excess_part_file']), part_peak_frac=float(rows[ipk]['excess_part_frac_file'])))
print('\nmarginal excess pairs per percentile point:', ' '.join(f'p{int(p)}:{v:.0f}±{e:.0f}' for p, v, e in zip(pc, [r['marginal_excess_per_pct'] for r in rows], [r['marginal_excess_per_pct_sd'] for r in rows])))
print(f'excess pairs: p5 {ex[PCTS.index(5)]:.0f}, p20 {ex[PCTS.index(20)]:.0f}, max {ex.max():.0f} at p{PCTS[int(np.argmax(ex))]}; participation excess peaks at p{PCTS[ipk]} ({rows[ipk]["excess_part_file"]:.0f} whistles, {rows[ipk]["excess_part_frac_file"]:.1%})')
json.dump(res, open(os.path.join(OUT, 'd04_threshold.json'), 'w'), indent=1, default=float)
print('saved d04_threshold.json')
