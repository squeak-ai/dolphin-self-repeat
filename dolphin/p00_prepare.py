"""p00 — build every derived input the d01-d05 scripts read, straight from the public DOLPHINFREE deposit.

Input  (DOLPHINFREE_DIR = extracted Zenodo record, doi:10.5281/zenodo.14637674):
    Single_hydrophone/Whistle_annotations/*-contours.json    whistle contours (time s, frequency Hz)
    Single_hydrophone/Visual_observation/*.xlsx              per-minute visual observation tables
Output (DOLPHIN_DATA_DIR, default ../data/dolphin):
    whistles_obs.csv          one row per whistle: timing, frequency summary, session id, observation metadata
    features.pkl              dict(X, Xs, Xk, C): 32-pt resampled contours (Hz), z-scored shape, kHz, raw contours
    dtw_abs.npy               pairwise DTW on absolute kHz contours       (float32, N x N)
    dtw_shape.npy             pairwise DTW on z-scored contour shape      (float32, N x N)
    v02_spectral_labels.csv   idx, spec4: four shape classes from spectral clustering on the DTW-shape affinity

Conventions
    * Contours are resampled to 32 points over normalised time; DTW uses a Sakoe-Chiba window of 8 samples
      (25 % of 32) — `dtaidistance.dtw.distance_matrix_fast`, symmetrised, stored as float32.
    * A *session* is a run of consecutive 1-min recordings of one device on one date (start-to-start gap <= 90 s).
    * spec4 codes follow d01_lib.SPEC4_NAMES (0 arch, 1 rise, 2 fall, 3 U). Spectral-clustering cluster numbers are
      arbitrary, so the four clusters are re-labelled by the shape of their mean z-scored contour.
Runtime: a few minutes on 8+ cores (the DTW matrices dominate). Row order is deterministic; the spectral step is
seeded (random_state=0).
"""
import os, re, glob, json, pickle, time, sys
import numpy as np, pandas as pd
from sklearn.cluster import SpectralClustering
from dtaidistance import dtw

ROOT = os.environ.get('DOLPHINFREE_DIR')
if not ROOT:
    sys.exit('Set DOLPHINFREE_DIR to the extracted Zenodo deposit (the folder containing Single_hydrophone/).')
CONTOURS = os.path.join(ROOT, 'Single_hydrophone', 'Whistle_annotations')
OBS = os.path.join(ROOT, 'Single_hydrophone', 'Visual_observation')
OUT = os.environ.get('DOLPHIN_DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'dolphin'))
os.makedirs(OUT, exist_ok=True)
N_RESAMPLE, DTW_WINDOW = 32, 8


def count_inflections(t, f, min_df=300.0):
    """Slope sign changes, ignoring wiggles below min_df Hz (tracing noise)."""
    if len(f) < 3:
        return 0
    ext, direction, ref = [], 0, f[0]
    for x in f[1:]:
        if direction >= 0 and x < ref - min_df:
            if direction == 1: ext.append(ref)
            direction = -1; ref = x
        elif direction <= 0 and x > ref + min_df:
            if direction == -1: ext.append(ref)
            direction = 1; ref = x
        else:
            if direction == 1 and x > ref: ref = x
            elif direction == -1 and x < ref: ref = x
            elif direction == 0: ref = x if abs(x - ref) > 0 else ref
    return len(ext)


def load_all():
    rows, contours = [], []
    for fn in sorted(glob.glob(os.path.join(CONTOURS, '*.json'))):
        d = json.load(open(fn))
        if not d:
            continue
        dev, date, tstr = re.match(r'(SCW\d+)_(\d{8})_(\d{6})-contours\.json', os.path.basename(fn)).groups()
        hh, mm, ss = int(tstr[:2]), int(tstr[2:4]), int(tstr[4:6])
        file_start = hh * 3600 + mm * 60 + ss
        for wid, pts in d.items():
            a = np.asarray(pts, dtype=float)
            if len(a) < 2:
                continue
            a = a[np.argsort(a[:, 0])]
            t, f = a[:, 0], a[:, 1]
            dur = t[-1] - t[0]
            if dur <= 0:
                continue
            idx = len(contours); contours.append(a)
            dt = np.diff(t); slope = np.diff(f) / np.where(dt > 0, dt, np.nan); slope = slope[np.isfinite(slope)]
            rows.append(dict(
                idx=idx, file=os.path.basename(fn), device=dev, date=date, time=tstr,
                year=int(date[:4]), hour=hh, minute=mm, wid=wid, file_start=file_start,
                t0=t[0], t1=t[-1], abs_t0=file_start + t[0], abs_t1=file_start + t[-1],
                dur=dur, npts=len(a), f_start=f[0], f_end=f[-1], f_min=f.min(), f_max=f.max(),
                f_mean=f.mean(), bw=f.max() - f.min(), delta_f=f[-1] - f[0],
                slope_mean=slope.mean() if len(slope) else 0.0,
                slope_absmean=np.abs(slope).mean() if len(slope) else 0.0,
                slope_max=np.abs(slope).max() if len(slope) else 0.0,
                n_infl=count_inflections(t, f)))
    df = pd.DataFrame(rows).sort_values(['device', 'date', 'file_start', 't0'], kind='mergesort').reset_index(drop=True)
    sess, cur, last_key, last_start = [], 0, None, None
    for r in df.itertuples():
        key = (r.device, r.date)
        if key != last_key or (last_start is not None and r.file_start - last_start > 90):
            cur += 1
        sess.append(cur); last_key, last_start = key, r.file_start
    df['session'] = sess
    return df, [contours[i] for i in df.idx]


def resample(contours, n=N_RESAMPLE):
    X = np.zeros((len(contours), n)); grid = np.linspace(0, 1, n)
    for i, a in enumerate(contours):
        t, f = a[:, 0], a[:, 1]
        tn = (t - t[0]) / (t[-1] - t[0])
        tn, ui = np.unique(tn, return_index=True)
        X[i] = np.interp(grid, tn, f[ui])
    return X


def shape_normalize(X):
    mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True); sd[sd < 1e-6] = 1.0
    return (X - mu) / sd


def dtw_matrix(X):
    D = dtw.distance_matrix_fast(np.ascontiguousarray(X), window=DTW_WINDOW, use_pruning=False, parallel=True, compact=False)
    D = np.where(np.isinf(D), 0, D)
    return (D + D.T).astype(np.float32)


def join_observations(w):
    """Attach the per-minute visual-observation metadata (behaviour, group size, distance, playback ...)."""
    obs = pd.concat([pd.read_excel(f) for f in sorted(glob.glob(os.path.join(OBS, '*.xlsx')))], ignore_index=True)
    obs['file'] = obs.audio_file.str.replace('.wav', '-contours.json', regex=False)
    assert obs.file.is_unique
    beh_cols = ['percent_travelling', 'percent_foraging', 'percent_socialising', 'percent_milling', 'percent_attracted']
    P = obs[beh_cols].fillna(0).values
    dom = np.array([c.replace('percent_', '') for c in beh_cols])[P.argmax(1)]
    dom = np.where(P.max(1) >= 50, dom, 'mixed')          # 'mixed' if no state >= 50 %
    dom = np.where(obs[beh_cols].isna().all(1), 'unobserved', dom)
    obs['behaviour'] = dom

    def playback(r):   # what was actually in the water during this minute
        if r.activation_sequence == 'during': return f'during_{r.signal}'
        if r.activation_sequence == 'after': return f'after_{r.signal}'
        return 'silent'
    obs['playback'] = obs.apply(playback, axis=1)
    obs['group_bin'] = obs.group_size.apply(
        lambda g: 'unknown' if pd.isna(g) else ('small(<=5)' if g <= 5 else ('medium(6-12)' if g <= 12 else 'large(>=14)')))
    obs['dist_bin'] = pd.cut(obs.distance, [0, 10, 30, 1000], labels=['near(<=10m)', 'mid(11-30m)', 'far(>30m)']).astype(str)
    keep = ['file', 'datetime_utc', 'sonar_noise', 'signal', 'group_size', 'group_bin', 'distance', 'dist_bin',
            'fishing_net_type', 'fishing_net', 'ID_group', 'ID_sequence', 'activation_sequence', 'playback',
            'behaviour', 'group_clustering', 'direction', 'speed', 'diving_time', 'special_observations'] + beh_cols
    wo = w.merge(obs[keep], on='file', how='left')
    assert wo.activation_sequence.notna().all(), 'whistle files without a matching observation row'
    return wo


def spectral_shape_classes(Ds, Xs, k=4, seed=0):
    """Spectral clustering on a Gaussian affinity of the DTW-shape matrix; clusters re-labelled arch/rise/fall/U."""
    N = len(Ds)
    sigma = np.median(Ds[np.triu_indices(N, 1)])
    A = np.exp(-(Ds.astype(np.float64) / sigma) ** 2)
    raw = SpectralClustering(k, affinity='precomputed', random_state=seed, assign_labels='kmeans').fit(A).labels_
    mean_curve = {c: Xs[raw == c].mean(0) for c in range(k)}
    rise_fall = {c: mean_curve[c][-8:].mean() - mean_curve[c][:8].mean() for c in range(k)}
    rise = max(rise_fall, key=rise_fall.get); fall = min(rise_fall, key=rise_fall.get)
    rest = [c for c in range(k) if c not in (rise, fall)]
    mid = {c: mean_curve[c][8:24].mean() - 0.5 * (mean_curve[c][:8].mean() + mean_curve[c][-8:].mean()) for c in rest}
    arch = max(mid, key=mid.get); u = min(mid, key=mid.get)
    code = {arch: 0, rise: 1, fall: 2, u: 3}
    return np.array([code[c] for c in raw])


if __name__ == '__main__':
    tic = time.time()
    df, C = load_all()
    print(f'{len(df)} whistles, {df.session.nunique()} sessions ({time.time() - tic:.0f}s)')
    X = resample(C); Xs = shape_normalize(X); Xk = X / 1000.0
    pickle.dump(dict(X=X, Xs=Xs, Xk=Xk, C=C), open(os.path.join(OUT, 'features.pkl'), 'wb'))
    wo = join_observations(df)
    wo.to_csv(os.path.join(OUT, 'whistles_obs.csv'), index=False)
    print('whistles_obs.csv', wo.shape)
    Da = dtw_matrix(Xk); np.save(os.path.join(OUT, 'dtw_abs.npy'), Da); print(f'dtw_abs {Da.shape} ({time.time() - tic:.0f}s)')
    Ds = dtw_matrix(Xs); np.save(os.path.join(OUT, 'dtw_shape.npy'), Ds); print(f'dtw_shape {Ds.shape} ({time.time() - tic:.0f}s)')
    spec4 = spectral_shape_classes(Ds, Xs)
    pd.DataFrame(dict(idx=df.idx.values, spec4=spec4)).to_csv(os.path.join(OUT, 'v02_spectral_labels.csv'), index=False)
    print('spec4 class sizes (arch, rise, fall, U):', np.bincount(spec4).tolist())
    print(f'done in {time.time() - tic:.0f}s')
