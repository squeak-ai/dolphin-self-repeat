"""d05a: does the exact self-repeat excess vary with behavioural condition?  (+ Task 3: bouts x condition)

Tests the arousal / serial-redundancy reading (dolphin-self-repeat-analysis.md §1.2 ii, §6 item 1).

Unit: encounter (year_ID_group from the Part II observation sheet). Pair unit as d02/d03/d04: unordered pairs of
modulated whistles within a session (sessions >= 20 modulated whistles); here additionally both members in the
same encounter. "Exact" = DTW-abs <= p5 (thr 7.77), non-overlapping, onset lag <= 10 s.
Null: onset permutation within 1-minute file (d04 headline), 50 draws. Per-encounter ratio = observed / mean null.

Covariates per encounter (whistle-weighted; most are constant within an encounter, a few vary across files):
  behaviour (mode), group_size (mean), distance (mean, m), playback (mode; also fraction of whistles under playback),
  sonar_noise (fraction), fishing_net (mode), rate = modulated whistles per recorded minute (files with >= 1 whistle).
Tests: Kruskal-Wallis on the per-encounter ratio (categorical), Spearman (continuous). Encounters with expected
null count < 5 pairs are excluded from the ratio tests (ratio too unstable); they are listed.

Second view (more power, exact covariates): pool pairs by the *file-level* condition of the first whistle and
compute observed / null with an approximate CI. Files carry the observation-sheet row, so the covariates are exact.

Task 3: bout membership per whistle (connected components of exact pairs, d03_bouts recipe, within session) crossed
with file-level behaviour / playback / net / sonar; observed fraction of whistles in bouts (>= 2, >= 3) vs the same
fraction under the file null.
"""
import numpy as np, pandas as pd, json, time
from scipy.stats import kruskal, spearmanr, mannwhitneyu
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from d01_lib import *
from d04_lib import Pairs

NDRAW = 50; MIN_NULL = 5
df, F = load(); rng = np.random.default_rng(5)
Da = np.load(os.path.join(OUT, 'dtw_abs.npy')); Ds = np.load(os.path.join(OUT, 'dtw_shape.npy'))
df['enc'] = df.enc.fillna(df.year.astype(str) + '_unobserved')
df['net'] = df.fishing_net.astype(str)
P = Pairs(df, Da, Ds)
thr_a = P.thr(5, 'abs'); exact = P.da <= thr_a
enc = df.enc.values; same_enc = enc[P.I] == enc[P.J]
file_of = df.file.values
N = len(df); mod = P.mod
print(f'{len(P.I)} pairs, same-encounter {same_enc.sum()}; thr_abs {thr_a:.3f}')

# ---------------- covariates ----------------
def wmode(s): return s.value_counts().idxmax()
files = df[mod].groupby('file').agg(enc=('enc', 'first'), n=('idx', 'size'), behaviour=('behaviour', 'first'), group_size=('group_size', 'first'),
                                    distance=('distance', 'first'), playback=('playback', 'first'), sonar=('sonar_noise', 'first'), net=('net', 'first'))
E = df[mod].groupby('enc').agg(n_mod=('idx', 'size'), n_files=('file', 'nunique'), n_sessions=('session', 'nunique'), behaviour=('behaviour', wmode),
                               behaviour_mixed=('behaviour', 'nunique'), group_size=('group_size', 'mean'), distance=('distance', 'mean'),
                               playback=('playback', wmode), playback_frac=('playback', lambda s: float((s != 'silent').mean())),
                               sonar_frac=('sonar_noise', 'mean'), net=('net', wmode), year=('year', 'first'))
E['rate'] = E.n_mod / E.n_files          # modulated whistles per recorded minute (only files with whistles exist in the table)
E['behaviour_mixed'] = E.behaviour_mixed > 1

# ---------------- exact pairs per encounter, observed and null ----------------
enc_list = list(E.index)
def counts(t0v, t1v):
    lag, gap, ov = P.geometry(t0v, t1v)
    sel = exact & same_enc & ~ov & (lag <= 10)
    e = pd.Series(enc[P.I[sel]]).value_counts()
    return e.reindex(enc_list).fillna(0).values

obs = counts(P.t0, P.t1)
tic = time.time(); nulls = np.array([counts(*P.permuted(rng, P.file_groups)) for _ in range(NDRAW)])
print(f'null {NDRAW} draws in {time.time()-tic:.1f}s')
E['obs'] = obs; E['null'] = nulls.mean(0); E['null_sd'] = nulls.std(0)
E['ratio'] = E.obs / E.null.replace(0, np.nan); E['excess'] = E.obs - E.null
E['z'] = (E.obs - E.null) / E.null_sd.replace(0, np.nan)
E['excess_per_whistle'] = E.excess / E.n_mod
ok = E.null >= MIN_NULL
print(f'\nencounters total {len(E)}, with null >= {MIN_NULL}: {ok.sum()}; excluded: {", ".join(f"{e} (null {E.null[e]:.1f})" for e in E.index[~ok])}')
T = E[ok].copy()
cols = ['n_mod', 'n_files', 'behaviour', 'group_size', 'distance', 'playback', 'playback_frac', 'sonar_frac', 'net', 'rate', 'obs', 'null', 'ratio', 'z']
print(T[cols].sort_values('ratio', ascending=False).to_string(float_format=lambda x: f'{x:.2f}'))
print(f'\nratio: median {T.ratio.median():.2f}, IQR {T.ratio.quantile(.25):.2f}-{T.ratio.quantile(.75):.2f}; >1 in {(T.ratio>1).sum()}/{len(T)}')

res = dict(n_encounters=len(E), n_tested=int(ok.sum()), min_null=MIN_NULL, n_draws=NDRAW, thr_abs=float(thr_a),
           encounters=E.reset_index().to_dict('records'))

# ---------------- encounter-level tests ----------------
print('\n== encounter-level tests (per-encounter ratio)')
tests = {}
for cat in ['behaviour', 'playback', 'net']:
    g = T.groupby(cat).ratio; groups = [v.values for k, v in g if len(v) >= 2]; names = [k for k, v in g if len(v) >= 2]
    summ = {str(k): dict(n=int(len(v)), median=float(v.median()), mean=float(v.mean()), min=float(v.min()), max=float(v.max())) for k, v in g}
    if len(groups) >= 2:
        H, p = kruskal(*groups); tests[cat] = dict(test='kruskal', H=float(H), p=float(p), groups=names, summary=summ)
        print(f'  {cat:10s} Kruskal H={H:.2f} p={p:.3f}  ' + '; '.join(f'{k}: n={v["n"]} med {v["median"]:.2f}' for k, v in summ.items()))
    else:
        tests[cat] = dict(test='kruskal', p=np.nan, note='fewer than two groups with n>=2', summary=summ)
        print(f'  {cat:10s} — fewer than two groups with n >= 2: ' + '; '.join(f'{k}: n={v["n"]}' for k, v in summ.items()))
for cont in ['group_size', 'distance', 'rate', 'playback_frac', 'sonar_frac', 'n_mod']:
    x = T[cont].values; y = T.ratio.values; m = np.isfinite(x) & np.isfinite(y)
    if m.sum() >= 5 and np.nanstd(x[m]) > 0:
        rho, p = spearmanr(x[m], y[m]); tests[cont] = dict(test='spearman', rho=float(rho), p=float(p), n=int(m.sum()))
        print(f'  {cont:13s} Spearman rho={rho:+.2f} p={p:.3f} (n={m.sum()})')
    else:
        tests[cont] = dict(test='spearman', p=np.nan, n=int(m.sum()), note='insufficient variation')
        print(f'  {cont:13s} — insufficient variation (n={m.sum()})')
# sonar and net as two-group where possible
for flag, lab in [('sonar_frac', 'sonar'), ('playback_frac', 'playback_any')]:
    a = T.ratio[T[flag] > 0.5].values; b = T.ratio[T[flag] <= 0.5].values
    if len(a) >= 2 and len(b) >= 2:
        U, p = mannwhitneyu(a, b); tests[lab + '_mw'] = dict(test='mannwhitney', p=float(p), n_hi=len(a), n_lo=len(b), med_hi=float(np.median(a)), med_lo=float(np.median(b)))
        print(f'  {lab:13s} MW p={p:.3f}: {flag}>0.5 n={len(a)} med {np.median(a):.2f} | <=0.5 n={len(b)} med {np.median(b):.2f}')
res['encounter_tests'] = tests

# ---------------- pooled pair-level view by file condition ----------------
print('\n== pooled pairs by file-level condition (obs / file-null), first-whistle file')
def ratio_ci(o, m, sd):
    if m <= 0 or o <= 0: return [np.nan, np.nan]
    rel = np.sqrt(1 / o + (sd / m) ** 2); r = o / m
    return [float(r * np.exp(-1.96 * rel)), float(r * np.exp(1.96 * rel))]
first_file = np.where(P.t0[P.I] <= P.t0[P.J], file_of[P.I], file_of[P.J])   # observed orientation; identity attached to pair
fcov = files[['behaviour', 'playback', 'sonar', 'net', 'group_size', 'distance']].copy()
fcov['group_bin'] = pd.cut(fcov.group_size, [0, 4, 9, 100], labels=['2-4', '5-9', '10+'])
fcov['dist_bin'] = pd.cut(fcov.distance, [0, 9, 29, 1000], labels=['<10 m', '10-29 m', '30+ m'])
fcov['sonar'] = fcov.sonar.map({0: 'no sonar', 1: 'sonar'})
pair_cov = fcov.reindex(first_file)
def pooled_counts(t0v, t1v):
    lag, gap, ov = P.geometry(t0v, t1v)
    return exact & same_enc & ~ov & (lag <= 10)
sel_obs = pooled_counts(P.t0, P.t1)
sel_null = [pooled_counts(*P.permuted(rng, P.file_groups)) for _ in range(NDRAW)]
pooled = {}
for cat in ['behaviour', 'playback', 'sonar', 'net', 'group_bin', 'dist_bin']:
    vals = pair_cov[cat].astype(str).values; pooled[cat] = {}
    for lev in pd.unique(vals):
        m = vals == lev; o = int(sel_obs[m].sum()); nv = np.array([s[m].sum() for s in sel_null], float)
        n_wh = int(files.n[fcov[cat].astype(str) == lev].sum())
        pooled[cat][lev] = dict(obs=o, null=float(nv.mean()), null_sd=float(nv.std()), ratio=o / nv.mean() if nv.mean() else np.nan,
                                ci=ratio_ci(o, nv.mean(), nv.std()), n_whistles=n_wh, excess_per_100_whistles=100 * (o - nv.mean()) / max(n_wh, 1))
    print(f'  {cat}:')
    for lev, r in sorted(pooled[cat].items(), key=lambda kv: -kv[1]['n_whistles']):
        print(f'    {lev:12s} whistles {r["n_whistles"]:5d}  exact {r["obs"]:5d} vs {r["null"]:7.1f} ± {r["null_sd"]:4.1f}  ratio {r["ratio"]:.2f} [{r["ci"][0]:.2f}-{r["ci"][1]:.2f}]  excess/100 wh {r["excess_per_100_whistles"]:+.2f}')
res['pooled_by_file_condition'] = pooled

# ---------------- Task 3: bouts x condition ----------------
print('\n== bouts x condition (connected components of exact pairs within session, d03 recipe)')
def bout_size(t0v, t1v):
    lag, gap, ov = P.geometry(t0v, t1v)
    ok = exact & ~ov & (lag <= 10)
    g = coo_matrix((np.ones(ok.sum()), (P.I[ok], P.J[ok])), shape=(N, N))
    _, lab = connected_components(g, directed=False)
    return np.bincount(lab)[lab]      # component size per whistle (1 = not in a bout)
bs_obs = bout_size(P.t0, P.t1)
bs_null = [bout_size(*P.permuted(rng, P.file_groups)) for _ in range(NDRAW)]
wh = df[mod].copy(); wh['bout2'] = bs_obs[mod] >= 2; wh['bout3'] = bs_obs[mod] >= 3
wh['sonar'] = wh.sonar_noise.map({0: 'no sonar', 1: 'sonar'})
wh['group_bin'] = pd.cut(wh.group_size, [0, 4, 9, 100], labels=['2-4', '5-9', '10+']).astype(str)
wh['dist_bin'] = pd.cut(wh.distance, [0, 9, 29, 1000], labels=['<10 m', '10-29 m', '30+ m']).astype(str)
bouts = {}
for cat in ['behaviour', 'playback', 'sonar', 'net', 'group_bin', 'dist_bin']:
    bouts[cat] = {}
    for lev in pd.unique(wh[cat].astype(str)):
        m = (wh[cat].astype(str) == lev).values; idx = wh.index.values[m]
        o2 = float((bs_obs[idx] >= 2).mean()); o3 = float((bs_obs[idx] >= 3).mean())
        n2 = np.array([(b[idx] >= 2).mean() for b in bs_null]); n3 = np.array([(b[idx] >= 3).mean() for b in bs_null])
        bouts[cat][lev] = dict(n=int(m.sum()), frac2=o2, null2=float(n2.mean()), null2_sd=float(n2.std()), ratio2=o2 / n2.mean() if n2.mean() else np.nan,
                               excess2=o2 - n2.mean(), frac3=o3, null3=float(n3.mean()), null3_sd=float(n3.std()), ratio3=o3 / n3.mean() if n3.mean() else np.nan,
                               excess3=o3 - n3.mean(), z3=(o3 - n3.mean()) / n3.std() if n3.std() > 0 else np.nan)
    print(f'  {cat}:')
    for lev, r in sorted(bouts[cat].items(), key=lambda kv: -kv[1]['n']):
        print(f'    {lev:12s} n={r["n"]:5d}  in bout>=2: {r["frac2"]:.3f} vs null {r["null2"]:.3f} (ratio {r["ratio2"]:.2f}, excess {r["excess2"]:+.3f})  | bout>=3: {r["frac3"]:.3f} vs {r["null3"]:.3f} (ratio {r["ratio3"]:.2f}, excess {r["excess3"]:+.3f}, z {r["z3"]:+.1f})')
# raw cross-tab for the record (observed fraction only, chi-square across behaviour)
from scipy.stats import chi2_contingency
ct = pd.crosstab(wh.behaviour, wh.bout3); chi, pchi, dof, _ = chi2_contingency(ct.values)
print(f'  raw bout>=3 x behaviour chi2 {chi:.1f} p={pchi:.2g} (confounded by rate/co-presence; the null-corrected excess above is the honest number)')
res['bouts_by_condition'] = bouts; res['bouts_raw_chi2'] = dict(chi2=float(chi), p=float(pchi), table=ct.to_dict())
json.dump(res, open(os.path.join(OUT, 'd05_condition.json'), 'w'), indent=1, default=float)
E[cols + ['excess', 'excess_per_whistle', 'n_sessions', 'year']].to_csv(os.path.join(OUT, 'd05_condition_encounters.csv'))
print('saved d05_condition.json, d05_condition_encounters.csv')
