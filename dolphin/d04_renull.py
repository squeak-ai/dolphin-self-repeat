"""d04a: re-null the d02 pairwise self-repeat statistics.

d02 measured the short-lag (<= 10 s) excess against a 60-600 s within-session baseline (and d02_part against a
session-level onset permutation). The d03 triangle analysis showed that a within-1-minute-file onset permutation
nearly doubles the chance count: minute-scale co-presence explains about half of the session-level effect.

Here every d02 headline number is recomputed against three baselines, side by side:
  (i)   'long'    - the original 60-600 s within-session pairs (d02 convention; for the conditional table only)
  (ii)  'session' - onset permutation within session (d02_part / d03 convention)
  (iii) 'file'    - onset permutation within 1-minute file (d03_triangles2 convention)  <- headline
Permutation: whistle (t0, t1) slots shuffled among the modulated whistles of a file/session, so identities
(shape, pitch, class) stay attached to the whistle and only *when* it occurred changes; within-file composition
and minute-scale drift are preserved under (iii).

Statistics: exact pairs within 10 s (non-overlapping), participation, conditional P(match | |df| bin) table,
gap-resolved exact excess, lag-resolved exact excess, per-session replication. 50 draws per null.
Pair construction identical to d02_repeats_decomp.py / d02_repeats_part.py (sessions with >= 20 modulated whistles).
"""
import numpy as np, pandas as pd, json, time
from scipy.stats import wilcoxon
from d01_lib import *

NDRAW = 50
df, F = load(); rng = np.random.default_rng(2024)
Ds = np.load(os.path.join(OUT, 'dtw_shape.npy')); Da = np.load(os.path.join(OUT, 'dtw_abs.npy'))
mod = (df.cls != 'constant').values
n_mod = int(mod.sum())
t0 = df.abs_t0.values; t1 = df.abs_t1.values; fm = df.f_mean.values / 1000
files = df.file.values; sessions_all = df.session.values

# ---- all within-session pairs among modulated whistles (sessions with >= 20) ----
PI, PJ, PS = [], [], []
sess_ix = {}
for s, d in df.groupby('session'):
    ix = d.index.values[mod[d.index.values]]
    if len(ix) < 20: continue
    sess_ix[s] = ix
    I, J = np.triu_indices(len(ix), 1)
    PI.append(ix[I]); PJ.append(ix[J]); PS.append(np.full(len(I), s))
PI = np.concatenate(PI); PJ = np.concatenate(PJ); PS = np.concatenate(PS)
da = Da[PI, PJ]; ds = Ds[PI, PJ]; dfm = np.abs(fm[PI] - fm[PJ])
thr_a = np.percentile(da, 5); thr_s = np.percentile(ds, 5)
exact = da <= thr_a; shape = ds <= thr_s
sess_list = sorted(sess_ix)
sess_pos = {s: np.where(PS == s)[0] for s in sess_list}
print(f'{len(PI)} pairs, {len(sess_list)} sessions, thr_abs {thr_a:.3f} thr_shape {thr_s:.3f}, n_mod {n_mod}')

DF_BINS = [0, 0.5, 1, 2, 3, 5, 15]
GAP_BINS = [(0, 0.25), (0.25, 1), (1, 3), (3, 10)]
LAG_BINS = [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 5), (5, 10)]
file_groups = [d.index.values for _, d in df[mod].groupby('file')]
sess_groups = list(sess_ix.values())


def geometry(t0v, t1v):
    """lag, gap, overlap for every pair, oriented so 'first' is the earlier onset."""
    a0 = t0v[PI]; b0 = t0v[PJ]
    first_i = a0 <= b0
    lag = np.abs(b0 - a0)
    first_end = np.where(first_i, t1v[PI], t1v[PJ]); second_on = np.where(first_i, b0, a0)
    gap = second_on - first_end
    return lag, gap, gap < 0


def stats(t0v, t1v):
    lag, gap, ov = geometry(t0v, t1v)
    short = lag <= 10                      # d02_decomp 'short' (includes overlapping)
    short_no = short & ~ov                 # d02_part / d03 pair unit
    r = {}
    ex10 = exact & short_no
    r['exact_pairs'] = int(ex10.sum())
    r['part'] = int(len(np.unique(np.concatenate([PI[ex10], PJ[ex10]]))))
    r['per_session'] = {int(s): int(ex10[p].sum()) for s, p in sess_pos.items()}
    # conditional table: P(shape match | df bin, short)
    r['cond'] = [float(shape[short & (dfm >= lo) & (dfm < hi)].mean()) for lo, hi in zip(DF_BINS[:-1], DF_BINS[1:])]
    r['cond_n'] = [int((short & (dfm >= lo) & (dfm < hi)).sum()) for lo, hi in zip(DF_BINS[:-1], DF_BINS[1:])]
    # gap-resolved exact count (non-overlapping) and pair count
    r['gap_exact'] = [int((exact & short_no & (gap >= lo) & (gap < hi)).sum()) for lo, hi in GAP_BINS]
    r['gap_n'] = [int((short_no & (gap >= lo) & (gap < hi)).sum()) for lo, hi in GAP_BINS]
    r['overlap_exact'] = int((exact & ov & short).sum()); r['overlap_n'] = int((ov & short).sum())
    # lag-resolved exact count (all pairs, d02_decomp lag profile convention)
    r['lag_exact'] = [int((exact & (lag >= lo) & (lag < hi)).sum()) for lo, hi in LAG_BINS]
    r['lag_n'] = [int(((lag >= lo) & (lag < hi)).sum()) for lo, hi in LAG_BINS]
    # per-session gap-bin exact counts for replication of the gap curve
    r['per_session_gap'] = {int(s): [int((exact & short_no & (gap >= lo) & (gap < hi))[p].sum()) for lo, hi in GAP_BINS] for s, p in sess_pos.items()}
    # long-lag baseline (d02 convention), only meaningful for observed times
    lng = (lag >= 60) & (lag <= 600)
    r['long_cond'] = [float(shape[lng & (dfm >= lo) & (dfm < hi)].mean()) for lo, hi in zip(DF_BINS[:-1], DF_BINS[1:])]
    r['long_exact_rate'] = float(exact[lng].mean()); r['long_n'] = int(lng.sum())
    r['short_exact_rate_no'] = float(exact[short_no].mean())
    r['short_no_n'] = int(short_no.sum())
    return r


def draw(groups):
    t0p = t0.copy(); t1p = t1.copy()
    for ix in groups:
        perm = rng.permutation(len(ix)); t0p[ix] = t0[ix][perm]; t1p[ix] = t1[ix][perm]
    return stats(t0p, t1p)


tic = time.time()
obs = stats(t0, t1)
print(f'observed in {time.time()-tic:.1f}s: exact pairs {obs["exact_pairs"]}, participation {obs["part"]} ({obs["part"]/n_mod:.1%})')
nulls = {}
for name, groups in [('session', sess_groups), ('file', file_groups)]:
    tic = time.time(); nulls[name] = [draw(groups) for _ in range(NDRAW)]
    print(f'{name} null: {NDRAW} draws in {time.time()-tic:.1f}s')


def agg(name, key, idx=None):
    v = np.array([(d[key] if idx is None else d[key][idx]) for d in nulls[name]], float)
    return float(v.mean()), float(v.std())


def ratio_ci(o, m, sd):
    """approximate 95% band for obs/null: Poisson on obs + null draw sd."""
    if m <= 0 or o <= 0: return [np.nan, np.nan]
    rel = np.sqrt(1 / o + (sd / m) ** 2); r = o / m
    return [float(r * np.exp(-1.96 * rel)), float(r * np.exp(1.96 * rel))]


res = dict(n_pairs=int(len(PI)), n_mod=n_mod, n_sessions=len(sess_list), thr_abs=float(thr_a), thr_shape=float(thr_s), n_draws=NDRAW,
           original_d02=dict(exact_pairs_ratio_session=2194 / 1499.7, participation_excess_session=(1922 - 1628.3) / 3864,
                             note='d02_repeats_part.json: 2194 pairs vs 1499.7 session-null; 1922 vs 1628.3 whistles'))

# ---- 1. exact pairs within 10 s and participation ----
print('\n== exact non-overlapping pairs within 10 s / participation')
sec = {}
for name in ['session', 'file']:
    m, sd = agg(name, 'exact_pairs'); pm, psd = agg(name, 'part')
    rep = sum(obs['per_session'][s] > np.mean([d['per_session'][s] for d in nulls[name]]) for s in obs['per_session'])
    per_null = {s: float(np.mean([d['per_session'][s] for d in nulls[name]])) for s in obs['per_session']}
    per_ratio = np.array([obs['per_session'][s] / per_null[s] for s in obs['per_session'] if per_null[s] >= 5 and obs['per_session'][s] > 0])
    wp = float(wilcoxon(np.log(per_ratio)).pvalue) if len(per_ratio) >= 6 else np.nan
    sec[name] = dict(pairs_null=m, pairs_sd=sd, pairs_ratio=obs['exact_pairs'] / m, pairs_ratio_ci=ratio_ci(obs['exact_pairs'], m, sd),
                     pairs_z=(obs['exact_pairs'] - m) / sd, pairs_excess=obs['exact_pairs'] - m,
                     part_null=pm, part_sd=psd, part_excess=obs['part'] - pm, part_excess_frac=(obs['part'] - pm) / n_mod,
                     part_obs_frac=obs['part'] / n_mod, part_null_frac=pm / n_mod,
                     sessions_gt=int(rep), sessions_tested=len(obs['per_session']), per_session_null=per_null,
                     sessions_ratio_tested=int(len(per_ratio)), sessions_ratio_gt1=int((per_ratio > 1).sum()), wilcoxon_p=wp,
                     per_session_median_ratio=float(np.median(per_ratio)) if len(per_ratio) else np.nan)
    print(f'  {name:8s} pairs {obs["exact_pairs"]} vs {m:.0f} ± {sd:.0f} -> {obs["exact_pairs"]/m:.2f}x [{sec[name]["pairs_ratio_ci"][0]:.2f}-{sec[name]["pairs_ratio_ci"][1]:.2f}] z {(obs["exact_pairs"]-m)/sd:.1f}; excess {obs["exact_pairs"]-m:.0f}'
          f' | whistles {obs["part"]} vs {pm:.0f} ± {psd:.0f}: excess {obs["part"]-pm:.0f} = {(obs["part"]-pm)/n_mod:.1%} | sessions obs>null {rep}/{len(obs["per_session"])}; ratio>1 {int((per_ratio>1).sum())}/{len(per_ratio)} (p={wp:.3g}, median {np.median(per_ratio):.2f})')
res['pairs'] = dict(obs_pairs=obs['exact_pairs'], obs_part=obs['part'], per_session_obs=obs['per_session'], **{k: v for k, v in sec.items()})

# ---- 2. conditional table ----
print('\n== conditional P(shape match | |df| bin, short) / baseline')
tab = []
for k, (lo, hi) in enumerate(zip(DF_BINS[:-1], DF_BINS[1:])):
    row = dict(lo=lo, hi=hi, n_short=obs['cond_n'][k], p_short=obs['cond'][k], p_long=obs['long_cond'][k], ratio_long=obs['cond'][k] / obs['long_cond'][k])
    # Wilson CI on p_short
    n = obs['cond_n'][k]; p = obs['cond'][k]; z = 1.96
    den = 1 + z * z / n; ctr = (p + z * z / (2 * n)) / den; half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    row['p_short_ci'] = [ctr - half, ctr + half]
    for name in ['session', 'file']:
        m, sd = agg(name, 'cond', k)
        row[f'p_null_{name}'] = m; row[f'p_null_{name}_sd'] = sd; row[f'ratio_{name}'] = p / m
        row[f'ratio_{name}_ci'] = [(ctr - half) / m, (ctr + half) / m]
    tab.append(row)
    print(f'  {lo:4.1f}-{hi:4.1f} n={row["n_short"]:5d}: P(short) {p:.4f} | long {row["p_long"]:.4f} -> {row["ratio_long"]:.2f} | session-null {row["p_null_session"]:.4f} -> {row["ratio_session"]:.2f} | file-null {row["p_null_file"]:.4f} -> {row["ratio_file"]:.2f} [{row["ratio_file_ci"][0]:.2f}-{row["ratio_file_ci"][1]:.2f}]')
res['conditional'] = tab

# ---- 3. gap-resolved exact excess ----
print('\n== gap-resolved exact excess (non-overlapping)')
gt = []
base_long = obs['long_exact_rate']
for k, (lo, hi) in enumerate(GAP_BINS):
    o = obs['gap_exact'][k]; n = obs['gap_n'][k]
    row = dict(lo=lo, hi=hi, n=n, obs=o, ratio_long=o / (n * base_long), expected_long=n * base_long)
    for name in ['session', 'file']:
        m, sd = agg(name, 'gap_exact', k)
        row[f'null_{name}'] = m; row[f'null_{name}_sd'] = sd; row[f'ratio_{name}'] = o / m; row[f'ratio_{name}_ci'] = ratio_ci(o, m, sd)
        row[f'excess_{name}'] = o - m
        per_null = {s: np.mean([d['per_session_gap'][s][k] for d in nulls[name]]) for s in obs['per_session_gap']}
        tested = [s for s in per_null if per_null[s] >= 3]
        row[f'sessions_gt_{name}'] = int(sum(obs['per_session_gap'][s][k] > per_null[s] for s in tested)); row[f'sessions_tested_{name}'] = len(tested)
    gt.append(row)
    print(f'  gap {lo:5.2f}-{hi:5.2f} n={n:6d} exact {o:5d}: long-base {row["ratio_long"]:.2f} | session {row["ratio_session"]:.2f} | file {row["ratio_file"]:.2f} [{row["ratio_file_ci"][0]:.2f}-{row["ratio_file_ci"][1]:.2f}] excess {row["excess_file"]:+.0f}; sessions>null {row["sessions_gt_file"]}/{row["sessions_tested_file"]}')
tot_f = sum(r['excess_file'] for r in gt); tot_s = sum(r['excess_session'] for r in gt)
shares = dict(file=[r['excess_file'] / tot_f for r in gt], session=[r['excess_session'] / tot_s for r in gt])
print(f'  excess mass at gaps >= 1 s: file {shares["file"][2]+shares["file"][3]:.0%}, session {shares["session"][2]+shares["session"][3]:.0%}; total excess pairs file {tot_f:.0f}, session {tot_s:.0f}')
ovm, ovsd = agg('file', 'overlap_exact')
res['gap'] = dict(table=gt, excess_share=shares, total_excess_file=tot_f, total_excess_session=tot_s,
                  overlap=dict(obs=obs['overlap_exact'], n=obs['overlap_n'], null_file=ovm, null_file_sd=ovsd, ratio_file=obs['overlap_exact'] / ovm, ratio_file_ci=ratio_ci(obs['overlap_exact'], ovm, ovsd)))
print(f'  overlapping pairs: exact {obs["overlap_exact"]} vs file-null {ovm:.0f} ± {ovsd:.0f} -> {obs["overlap_exact"]/ovm:.2f}')

# ---- 4. lag-resolved (all pairs) ----
print('\n== lag-resolved exact excess (all pairs incl. overlapping; d02 lag_profile convention)')
lt = []
for k, (lo, hi) in enumerate(LAG_BINS):
    o = obs['lag_exact'][k]; n = obs['lag_n'][k]
    row = dict(lo=lo, hi=hi, n=n, obs=o, ratio_long=o / (n * base_long))
    for name in ['session', 'file']:
        m, sd = agg(name, 'lag_exact', k); row[f'null_{name}'] = m; row[f'ratio_{name}'] = o / m; row[f'ratio_{name}_ci'] = ratio_ci(o, m, sd)
    lt.append(row)
    print(f'  lag {lo:4.1f}-{hi:4.1f} n={n:6d} exact {o:5d}: long-base {row["ratio_long"]:.2f} | session {row["ratio_session"]:.2f} | file {row["ratio_file"]:.2f}')
res['lag'] = lt

json.dump(res, open(os.path.join(OUT, 'd04_renull.json'), 'w'), indent=1, default=float)
print('saved d04_renull.json')
