"""Gate: chunked output vs single-pass output.

- log-mel L1 (perceptual proxy) must be < 0.1 over the aligned overlap
- boundary click detector: max |sample diff| jump at chunk joints <= 3x the
  95th percentile intra-chunk jump

Run: server/.venv/bin/python native/tools/compare_chunked.py single.wav chunked.wav <block40k>
"""
import sys

import librosa
import numpy as np


def main() -> None:
    single_p, chunked_p, block = sys.argv[1], sys.argv[2], int(sys.argv[3])
    s, _ = librosa.load(single_p, sr=40000, mono=True)
    c, _ = librosa.load(chunked_p, sr=40000, mono=True)

    # SOLA legitimately time-warps within its search window, so a single global
    # lag under-measures fidelity. Align each 0.5 s segment independently
    # (bounded per-segment lag), then compare log-mel per segment.
    w = 20000
    l1s = []
    lags_used = []
    for pos in range(0, len(c) - w, w):
        seg = c[pos : pos + w]
        lo, hi = max(0, pos - 6000), pos + 12000
        best, bl = -1e18, pos
        for lag in range(lo, min(hi, len(s) - w), 4):
            v = float(np.dot(s[lag : lag + w], seg))
            if v > best:
                best, bl = v, lag
        sseg = s[bl : bl + w]
        ms = np.log(np.clip(librosa.feature.melspectrogram(y=sseg, sr=40000, n_mels=80), 1e-5, None))
        mc = np.log(np.clip(librosa.feature.melspectrogram(y=seg, sr=40000, n_mels=80), 1e-5, None))
        l1s.append(float(np.mean(np.abs(ms - mc))))
        lags_used.append(bl - pos)
    l1 = float(np.mean(l1s))
    best_lag = int(np.median(lags_used))

    jumps = np.abs(np.diff(c))
    intra = np.percentile(jumps, 95)
    joint_idx = np.arange(block, len(c) - 1, block)
    joint_max = float(np.max(jumps[joint_idx - 1])) if len(joint_idx) else 0.0
    clicky = joint_max > 3 * max(intra, 1e-4)

    print(
        {
            "align_lag": best_lag,
            "logmel_L1": round(l1, 4),
            "joint_max_jump": round(joint_max, 5),
            "intra_p95_jump": round(float(intra), 5),
            "click": bool(clicky),
            "pass": bool(l1 < 0.1 and not clicky),
        }
    )
    sys.exit(0 if (l1 < 0.1 and not clicky) else 1)


if __name__ == "__main__":
    main()
