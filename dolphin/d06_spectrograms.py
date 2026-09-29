"""d06 — supplementary spectrograms from the raw DOLPHINFREE single-hydrophone WAVs (not numbered figures).

spectrogram_repeat_bout   the Fig 1d sequence (session 34): eight arches and one U in 31 s, onsets marked
spectrogram_repeat_zoom   the first two arches of that bout (~3.4 s) — the repeat at listening scale
spectrogram_chorus        the densest 10-s window in the dataset (37 annotated whistles, up to 5 at once)

Audio: 512 kHz / 24-bit mono. Resampled to 64 kHz (whistles sit at 3–20 kHz), STFT, per-frequency median noise floor
subtracted (flattens the hydrophone's tonal self-noise), 30–32 dB display range above the floor. Onset ticks and the whistle count come
from the annotated contour table (whistles_obs.csv), so the audio and the analysis can be read against each other.

Env: DOLPHIN_DATA_DIR (analysis outputs), DOLPHIN_AUDIO_DIR (…/Single_hydrophone/Audio_data), DOLPHIN_FIG_DIR.
"""
import sys
import numpy as np, pandas as pd, soundfile as sf
from scipy.signal import resample_poly, stft
from matplotlib.colors import LinearSegmentedColormap
from d01_lib import OUT, FIG, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from figstyle import *

AUDIO = os.environ.get('DOLPHIN_AUDIO_DIR', os.path.join(os.path.dirname(__file__), '..', 'data', 'dolphin', 'audio'))
df = pd.read_csv(os.path.join(OUT, 'whistles_obs.csv'))
SR = 64000
# ink on paper, warmed toward the dolphin orange
CMAP = LinearSegmentedColormap.from_list('paper_ink', [SURF, '#f7d9c8', DOLPHIN_LIGHT, ORANGE, '#7a2f12', INK])


def load(wav, t0, t1, pad=1.0):
    info = sf.info(wav); a = max(0.0, t0 - pad); b = min(info.duration, t1 + pad)
    x, sr = sf.read(wav, start=int(a * info.samplerate), stop=int(b * info.samplerate), dtype='float32')
    x = resample_poly(x, SR, sr)                                  # 512k -> 64k (anti-aliased)
    return x, a


def spec(x, nper=1024, hop=128):
    f, t, Z = stft(x, SR, nperseg=nper, noverlap=nper - hop, window='hann', boundary=None, padded=False)
    S = 20 * np.log10(np.abs(Z) + 1e-12)
    S -= np.median(S, axis=1, keepdims=True)                      # per-frequency noise floor
    return f, t, S


def render(name, wav_stem, t0, t1, headline_text, sub, fmax=20, width=FULL_W, height=4.2, rng=30, mark=True):
    wav = os.path.join(AUDIO, wav_stem + '.wav')
    x, a = load(wav, t0, t1)
    f, t, S = spec(x)
    t = t + a; keep = (t >= t0) & (t <= t1); fk = (f >= 1000) & (f <= fmax * 1000)
    S = S[np.ix_(fk, keep)]; t = t[keep] - t0; f = f[fk] / 1000
    fig, ax = plt.subplots(figsize=(width, height))
    style(ax, xlabel='time (s)', ylabel='frequency (kHz)'); ax.grid(False)
    ax.pcolormesh(t, f, S, cmap=CMAP, vmin=2, vmax=2 + rng, shading='auto', rasterized=True)
    ax.set_ylim(1, fmax); ax.set_xlim(0, t1 - t0)
    if mark:
        g = df[(df.file == wav_stem + '-contours.json') & (df.t0 >= t0) & (df.t0 <= t1)]
        ax.plot(g.t0 - t0, np.full(len(g), fmax * 0.985), 'v', color=INK2, ms=5, clip_on=False, zorder=5)
    headline(fig, headline_text, y=1.03, sub=sub, sub_y=0.975)
    fig.subplots_adjust(top=0.82)
    save_both(fig, FIG, name)
    return fig


# --- the Fig 1d bout: session 34, whistles 3987–3995 ---
seq = df.iloc[3987:3996]; stem = seq.file.iloc[0].replace('-contours.json', ''); d = str(int(seq.date.iloc[0]))
s0, s1 = seq.t0.min() - 0.6, seq.t1.max() + 0.6
render('spectrogram_repeat_bout', stem, s0, s1,
       'A repeat bout in the raw recording: eight arches and one U in 31 seconds',
       f'Session {int(seq.session.iloc[0])} ({d[:4]}-{d[4:6]}-{d[6:]}), single hydrophone. ▼ = annotated whistle onsets. '
       'Same whistles as Fig 1d.')
a2 = seq.iloc[:2]
dt = a2.t0.iloc[1] - a2.t0.iloc[0]
render('spectrogram_repeat_zoom', stem, a2.t0.min() - 0.3, a2.t1.max() + 0.3,
       f'The same arch, twice, onsets {dt:.1f} seconds apart',
       f'First two renditions of the session {int(seq.session.iloc[0])} bout. Same contour, same absolute frequency. '
       'The upper trace is\neach whistle\'s second harmonic (twice the fundamental), not a second animal.', height=5.2)

# --- densest 10-s window in the dataset ---
best = None
for fname, g in df.groupby('file'):
    for s in np.arange(0, 50.5, 0.5):
        n = int(((g.t1 > s) & (g.t0 < s + 10)).sum())
        if best is None or n > best[0]:
            best = (n, fname, s)
n, fname, s = best; g = df[df.file == fname].iloc[0]; d = str(int(g.date))
grid = np.arange(s, s + 10, 0.01); w = df[df.file == fname]
sim = max(int(((w.t0 <= u) & (w.t1 >= u)).sum()) for u in grid)
render('spectrogram_chorus', fname.replace('-contours.json', ''), s, s + 10,
       f'The chorus: {n} whistles in 10 seconds, up to {sim} at once',
       f'Densest 10-s window in the dataset — session {int(g.session)} ({d[:4]}-{d[4:6]}-{d[6:]}), single hydrophone. '
       '▼ = annotated whistle onsets.', rng=32)
print('done')
