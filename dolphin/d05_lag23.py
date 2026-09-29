"""d05b: the 2-3 s lag bin.

d02's lag profile (long-lag baseline, 60-600 s) showed the 2-3 s onset-to-onset bin as the only bin with a
shape-only (shape-matched but NOT exact) excess above 1: 1.07 at n = 3,554, and cond(f far) = 1.07. That is
Janik's vocal-matching window (Tursiops respond with a copy within ~1-3 s). Everywhere else transposed
shape matches sit at or below chance. If a rare matching signal exists in this chorus, this is where it would hide.

Here the bin is taken apart:
  1. shape-match / transposed-shape-match / exact rates in 2-3 s vs the neighbouring 3-5 s bin (local control)
     and vs the 1-2 s bin on the other side;
  2. against the within-1-minute-file onset permutation (d04 headline null, 50 draws) - the baseline that
     already absorbed half of the exact-repeat effect;
  3. per-session replication (sign test, 2-3 vs 3-5);
  4. session concentration of the 2-3 s shape matches;
  5. class-pair composition of shape matches, 2-3 vs 3-5 (chi-square on the class-pair table);
  6. a finer lag grid (0.25 s) across 1-5 s to see whether anything peaks inside the bin or the 1.07 was
     bin-edge luck.
Pair unit: all within-session pairs among modulated whistles, sessions >= 20 (d02 lag_profile convention,
overlapping pairs included; at 2-3 s lag overlap is rare, ~0.84 s mean duration). Non-overlapping variant reported too.
"""
import numpy as np, pandas as pd, json, time
from scipy.stats import binomtest, chi2_contingency, norm
from d01_lib import *
from d04_lib import Pairs

NDRAW = 50
df, F = load(); rng = np.random.default_rng(23)
Ds = np.load(os.path.join(OUT, 'dtw_shape.npy')); Da = np.load(os.path.join(OUT, 'dtw_abs.npy'))
P = Pairs(df, Da, Ds)
thr_a = P.thr(5, 'abs'); thr_s = P.thr(5, 'shape')
exact = P.da <= thr_a; shape = P.ds <= thr_s
shape_only = shape & ~exact                 # d02 definition
transposed = shape & (P.dfm > 1.0)          # shape match at > 1 kHz mean-frequency shift (matching would live here)
cls = df.cls.astype(str).values
sess_list = sorted(P.sess_ix)
print(f'{len(P.I)} pairs, {len(sess_list)} sessions; thr_abs {thr_a:.3f} thr_shape {thr_s:.3f}')

BINS = {'1-2': (1, 2), '2-3': (2, 3), '3-5': (3, 5), '5-10': (5, 10)}
FINE = [(a / 4, (a + 1) / 4) for a in range(4, 20)]   # 1.00-5.00 s in 0.25 s steps


def rates(lag, ov, sel_extra=None):
    """per lag bin: n, and rate of exact / shape / shape-only / transposed among pairs in the bin."""
    out = {}
    for name, (lo, hi) in BINS.items():
        m = (lag >= lo) & (lag < hi)
        if sel_extra is not None: m &= sel_extra
        n = int(m.sum())
        out[name] = dict(n=n, exact=int(exact[m].sum()), shape=int(shape[m].sum()), shape_only=int(shape_only[m].sum()),
                         transposed=int(transposed[m].sum()))
    return out


def per_session(lag):
    r = {}
    for s in sess_list:
        p = P.sess_pos[s]; l = lag[p]
        a = (l >= 2) & (l < 3); b = (l >= 3) & (l < 5)
        r[s] = dict(n23=int(a.sum()), n35=int(b.sum()), shape23=int(shape[p][a].sum()), shape35=int(shape[p][b].sum()),
                    trans23=int(transposed[p][a].sum()), trans35=int(transposed[p][b].sum()),
                    so23=int(shape_only[p][a].sum()), so35=int(shape_only[p][b].sum()))
    return r


def fine(lag):
    return [dict(lo=lo, hi=hi, n=int(((lag >= lo) & (lag < hi)).sum()),
                 shape=int(shape[(lag >= lo) & (lag < hi)].sum()), shape_only=int(shape_only[(lag >= lo) & (lag < hi)].sum()),
                 transposed=int(transposed[(lag >= lo) & (lag < hi)].sum()), exact=int(exact[(lag >= lo) & (lag < hi)].sum()))
            for lo, hi in FINE]


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n; ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (ctr - half, ctr + half)


def two_prop(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2); se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (k1 / n1 - k2 / n2) / se if se > 0 else 0.0
    return float(z), float(2 * norm.sf(abs(z)))


# ---------------- observed ----------------
lag, gap, ov = P.geometry(P.t0, P.t1)
obs_all = rates(lag, ov); obs_no = rates(lag, ov, ~ov)
obs_ps = per_session(lag); obs_fine = fine(lag)
# long-lag baseline for the d02 shape-only number (reproduce 1.07)
lng = (lag >= 60) & (lag <= 600)
b_so = shape_only[lng].mean(); b_sh = shape[lng].mean(); b_tr = transposed[lng].mean(); b_ex = exact[lng].mean()
res = dict(n_pairs=int(len(P.I)), n_sessions=len(sess_list), thr_abs=float(thr_a), thr_shape=float(thr_s), n_draws=NDRAW,
           long_baseline=dict(n=int(lng.sum()), shape_only=float(b_so), shape=float(b_sh), transposed=float(b_tr), exact=float(b_ex)))

print('\n== rates by lag bin (all pairs) vs long-lag baseline (d02 convention)')
tab = {}
for name, r in obs_all.items():
    n = r['n']
    row = dict(n=n, exact_rate=r['exact'] / n, shape_rate=r['shape'] / n, shape_only_rate=r['shape_only'] / n, transposed_rate=r['transposed'] / n,
               exact_vs_long=r['exact'] / n / b_ex, shape_only_vs_long=r['shape_only'] / n / b_so, transposed_vs_long=r['transposed'] / n / b_tr,
               shape_only_ci=wilson(r['shape_only'], n), transposed_ci=wilson(r['transposed'], n))
    tab[name] = row
    print(f'  {name:5s} n={n:6d}: exact {row["exact_vs_long"]:.2f} | shape-only {row["shape_only_vs_long"]:.2f} (rate {row["shape_only_rate"]:.4f}, k={r["shape_only"]}) | transposed {row["transposed_vs_long"]:.2f} (rate {row["transposed_rate"]:.4f}, k={r["transposed"]})')
res['bins_vs_long'] = tab

# ---------------- 1. local control: 2-3 vs 3-5 and vs 1-2 ----------------
print('\n== 2-3 s vs neighbours (two-proportion z)')
loc = {}
for key in ['shape', 'shape_only', 'transposed', 'exact']:
    a = obs_all['2-3']; row = {}
    for other in ['3-5', '1-2']:
        b = obs_all[other]
        z, p = two_prop(a[key], a['n'], b[key], b['n'])
        row[other] = dict(rate_23=a[key] / a['n'], rate_other=b[key] / b['n'], ratio=(a[key] / a['n']) / (b[key] / b['n']), z=z, p=p)
    loc[key] = row
    print(f'  {key:10s} 2-3: {a[key]/a["n"]:.4f} | 3-5: {obs_all["3-5"][key]/obs_all["3-5"]["n"]:.4f} ratio {row["3-5"]["ratio"]:.2f} p={row["3-5"]["p"]:.3f} | 1-2: {obs_all["1-2"][key]/obs_all["1-2"]["n"]:.4f} ratio {row["1-2"]["ratio"]:.2f} p={row["1-2"]["p"]:.3f}')
res['local_control'] = loc
# non-overlapping variant
res['bins_nonoverlap'] = {k: dict(n=v['n'], shape_only_rate=v['shape_only'] / max(v['n'], 1), transposed_rate=v['transposed'] / max(v['n'], 1)) for k, v in obs_no.items()}

# ---------------- 2. within-minute file null ----------------
print(f'\n== within-minute onset permutation ({NDRAW} draws)')
tic = time.time(); nulls = []
for _ in range(NDRAW):
    t0p, t1p = P.permuted(rng, P.file_groups)
    lp, gp, op = P.geometry(t0p, t1p)
    nulls.append(dict(all=rates(lp, op), ps=per_session(lp), fine=fine(lp)))
print(f'  {time.time()-tic:.1f}s')


def null_stat(name, key, which='all'):
    v = np.array([d[which][name][key] for d in nulls], float); n = np.array([d[which][name]['n'] for d in nulls], float)
    return v.mean(), v.std(), (v / n).mean(), (v / n).std()


def ratio_ci(o, m, sd):
    if m <= 0 or o <= 0: return [np.nan, np.nan]
    rel = np.sqrt(1 / o + (sd / m) ** 2); r = o / m
    return [float(r * np.exp(-1.96 * rel)), float(r * np.exp(1.96 * rel))]


fn = {}
for name in BINS:
    fn[name] = {}
    for key in ['exact', 'shape', 'shape_only', 'transposed']:
        o = obs_all[name][key]; m, sd, rm, rsd = null_stat(name, key)
        # rate ratio (obs rate / null rate) is the right comparison since permutation changes n per bin
        orate = o / obs_all[name]['n']
        fn[name][key] = dict(obs=o, obs_rate=orate, null_count=m, null_count_sd=sd, null_rate=rm, null_rate_sd=rsd,
                             rate_ratio=orate / rm, z_rate=(orate - rm) / rsd if rsd > 0 else np.nan,
                             rate_ratio_ci=[c / rm for c in wilson(o, obs_all[name]['n'])])
    r = fn[name]
    print(f'  {name:5s}: exact {r["exact"]["rate_ratio"]:.2f} (z {r["exact"]["z_rate"]:.1f}) | shape-only {r["shape_only"]["rate_ratio"]:.2f} [{r["shape_only"]["rate_ratio_ci"][0]:.2f}-{r["shape_only"]["rate_ratio_ci"][1]:.2f}] z {r["shape_only"]["z_rate"]:.1f}'
          f' | transposed {r["transposed"]["rate_ratio"]:.2f} [{r["transposed"]["rate_ratio_ci"][0]:.2f}-{r["transposed"]["rate_ratio_ci"][1]:.2f}] z {r["transposed"]["z_rate"]:.1f}')
res['file_null'] = fn

# ---------------- 3. per-session replication ----------------
print('\n== per-session: is the 2-3 s shape / transposed rate above the 3-5 s rate?')
ps = {}
for key, (k23, k35) in {'shape': ('shape23', 'shape35'), 'transposed': ('trans23', 'trans35'), 'shape_only': ('so23', 'so35')}.items():
    tested = [s for s in sess_list if obs_ps[s]['n23'] >= 30 and obs_ps[s]['n35'] >= 30 and obs_ps[s][k35] > 0]
    gt = [s for s in tested if obs_ps[s][k23] / obs_ps[s]['n23'] > obs_ps[s][k35] / obs_ps[s]['n35']]
    # against file null: sessions where observed 2-3 s rate > mean null 2-3 s rate
    gt_null = []
    for s in tested:
        nr = np.mean([d['ps'][s][k23] / max(d['ps'][s]['n23'], 1) for d in nulls])
        if obs_ps[s][k23] / obs_ps[s]['n23'] > nr: gt_null.append(s)
    pv = float(binomtest(len(gt), len(tested), 0.5).pvalue) if tested else np.nan
    pv2 = float(binomtest(len(gt_null), len(tested), 0.5).pvalue) if tested else np.nan
    ps[key] = dict(tested=len(tested), gt_35=len(gt), p_sign_35=pv, gt_null=len(gt_null), p_sign_null=pv2,
                   sessions_gt_35=[int(s) for s in gt], sessions_gt_null=[int(s) for s in gt_null])
    print(f'  {key:10s}: 2-3 > 3-5 in {len(gt)}/{len(tested)} sessions (sign p={pv:.2f}); 2-3 obs > file-null in {len(gt_null)}/{len(tested)} (p={pv2:.2f})')
res['per_session'] = ps
res['per_session_table'] = {int(s): obs_ps[s] for s in sess_list}

# ---------------- 4. session concentration ----------------
print('\n== session concentration of 2-3 s shape matches')
m23 = (lag >= 2) & (lag < 3); m35 = (lag >= 3) & (lag < 5)
conc = []
for s in sess_list:
    p = P.sess_pos[s]
    conc.append(dict(session=int(s), n23=int(m23[p].sum()), shape23=int((shape & m23)[p].sum()), trans23=int((transposed & m23)[p].sum()),
                     share_pairs=float(m23[p].sum() / m23.sum()), share_shape=float((shape & m23)[p].sum() / (shape & m23).sum()),
                     share_trans=float((transposed & m23)[p].sum() / max((transposed & m23).sum(), 1))))
cd = pd.DataFrame(conc).sort_values('share_shape', ascending=False)
cd['enrich'] = cd.share_shape / cd.share_pairs
print(cd.head(8).to_string(index=False, float_format=lambda x: f'{x:.3f}'))
# is the 2-3 s shape excess (vs 3-5) driven by one session? leave-one-out ratio
loo = {}
for s in sess_list:
    keep = P.S != s
    r23 = shape[m23 & keep].mean(); r35 = shape[m35 & keep].mean()
    loo[int(s)] = float(r23 / r35)
res['concentration'] = dict(table=cd.to_dict('records'), loo_shape_ratio_23_vs_35=loo,
                            loo_min=float(min(loo.values())), loo_max=float(max(loo.values())),
                            full_shape_ratio_23_vs_35=float(shape[m23].mean() / shape[m35].mean()))
print(f'  leave-one-session-out shape ratio (2-3 / 3-5): full {res["concentration"]["full_shape_ratio_23_vs_35"]:.3f}, range {res["concentration"]["loo_min"]:.3f}-{res["concentration"]["loo_max"]:.3f}')

# ---------------- 5. class-pair composition ----------------
print('\n== class-pair composition of shape-matched pairs, 2-3 s vs 3-5 s')
def cp(sel):
    a = cls[P.I[sel]]; b = cls[P.J[sel]]
    key = np.where(a < b, np.char.add(np.char.add(a, '-'), b), np.char.add(np.char.add(b, '-'), a))
    return pd.Series(key).value_counts()
c23 = cp(shape & m23); c35 = cp(shape & m35); ct = pd.concat([c23, c35], axis=1).fillna(0); ct.columns = ['2-3', '3-5']
ct['frac_23'] = ct['2-3'] / ct['2-3'].sum(); ct['frac_35'] = ct['3-5'] / ct['3-5'].sum()
print(ct.sort_values('2-3', ascending=False).to_string(float_format=lambda x: f'{x:.3f}'))
chi, pchi, dof, _ = chi2_contingency(ct[['2-3', '3-5']].values + 0.5)
print(f'  chi-square {chi:.1f} dof {dof} p={pchi:.3f}')
# transposed-only composition
t23 = cp(transposed & m23); t35 = cp(transposed & m35); tt = pd.concat([t23, t35], axis=1).fillna(0); tt.columns = ['2-3', '3-5']
res['class_pairs'] = dict(shape=ct.to_dict(), chi2=float(chi), p=float(pchi), transposed=tt.to_dict())

# ---------------- 6. fine lag grid ----------------
print('\n== fine lag grid 1-5 s (0.25 s), shape-only and transposed rate ratio vs file null')
fg = []
for k, r in enumerate(obs_fine):
    n = r['n']; row = dict(lo=r['lo'], hi=r['hi'], n=n)
    for key in ['shape_only', 'transposed', 'exact']:
        v = np.array([d['fine'][k][key] / max(d['fine'][k]['n'], 1) for d in nulls]); orate = r[key] / n
        row[key + '_ratio'] = float(orate / v.mean()); row[key + '_z'] = float((orate - v.mean()) / v.std()) if v.std() > 0 else np.nan
    fg.append(row)
    print(f'  {r["lo"]:.2f}-{r["hi"]:.2f} n={n:5d}: exact {row["exact_ratio"]:.2f} | shape-only {row["shape_only_ratio"]:.2f} (z {row["shape_only_z"]:+.1f}) | transposed {row["transposed_ratio"]:.2f} (z {row["transposed_z"]:+.1f})')
res['fine_grid'] = fg

# ---------------- verdict ----------------
so = fn['2-3']['shape_only']; tr = fn['2-3']['transposed']
verdict = dict(shape_only_ratio_file=so['rate_ratio'], shape_only_z=so['z_rate'], transposed_ratio_file=tr['rate_ratio'], transposed_z=tr['z_rate'],
               local_ratio_shape_only=loc['shape_only']['3-5']['ratio'], local_p=loc['shape_only']['3-5']['p'],
               sessions_gt=f"{ps['transposed']['gt_35']}/{ps['transposed']['tested']}", loo_range=[res['concentration']['loo_min'], res['concentration']['loo_max']])
signal = (so['rate_ratio_ci'][0] > 1) and (tr['rate_ratio_ci'][0] > 1) and ps['transposed']['p_sign_35'] < 0.05
verdict['call'] = 'signal' if signal else 'noise'
res['verdict'] = verdict
print(f'\nVERDICT: {verdict["call"].upper()} — shape-only vs file null {so["rate_ratio"]:.2f} (z {so["z_rate"]:.1f}), transposed {tr["rate_ratio"]:.2f} (z {tr["z_rate"]:.1f}), local 2-3/3-5 p={loc["shape_only"]["3-5"]["p"]:.3f}, sessions {verdict["sessions_gt"]}')
json.dump(res, open(os.path.join(OUT, 'd05_lag23.json'), 'w'), indent=1, default=float)
print('saved d05_lag23.json')
