"""T0c: put the broadcast recording on the tactical timeline.

The broadcast is the live TV feed: real time keeps running through replays and close-ups, so
inside each half   tactical frame = broadcast frame + a constant.   The two constants are found
here by aligning the ball track of the two trackings; nothing is typed by hand.

Output cache/t0_broadcast.npz (row = tactical video frame):
    covered[T]     the broadcast has a frame for this instant
    wide[T]        the broadcast shows the pitch (a homography exists); False = replay, close-up, crowd
    bc_ball[T]     the broadcast tracking has the ball (only possible in the wide shot)
    offsets[2]     tactical frame - broadcast frame, per half
    med_dist_m[2]  median distance between the two ball tracks after alignment, metres
"""
import os

import numpy as np

from . import io, paths

CHUNK = 7500        # broadcast frames per coarse estimate (5 min)
MIN_COMMON = 500    # frames with the ball in both trackings needed to trust an estimate
MAX_DIST_M = 1.5    # a correct alignment leaves the two balls well under this
MAX_DRIFT = 2       # frames the best offset of a 10-minute piece may differ from its half's


def _corr(a, fb, m):
    """c[off] = sum_i a[i] * b[i + off] for off = 0 .. m - 1; fb is the rfft of b (size m)."""
    return np.fft.irfft(np.conj(np.fft.rfft(a, m)) * fb, m)


def find_offset(ref_xy, ref_ok, qry_xy, qry_ok, min_common=MIN_COMMON):
    """(offset, rms, n_common): the offset >= 0, with  ref index = qry index + offset, that minimises
    the mean squared distance between the two tracks over the frames where both exist. Every offset
    is evaluated at once through FFT correlations; (-1, inf, 0) when none has min_common frames."""
    n, k = len(ref_xy), len(qry_xy)
    m = 1 << int(np.ceil(np.log2(n + k)))
    mr, mq = ref_ok.astype(np.float64), qry_ok.astype(np.float64)
    r = np.where(ref_ok[:, None], ref_xy, 0.0).astype(np.float64)
    q = np.where(qry_ok[:, None], qry_xy, 0.0).astype(np.float64)
    f_mr = np.fft.rfft(mr, m)
    cnt = _corr(mq, f_mr, m)
    sse = np.zeros(m)
    for c in (0, 1):
        sse += (_corr(q[:, c] ** 2, f_mr, m) + _corr(mq, np.fft.rfft(r[:, c] ** 2, m), m)
                - 2.0 * _corr(q[:, c], np.fft.rfft(r[:, c], m), m))
    ok = (cnt >= min_common - 0.5) & (np.arange(m) < n)
    if not ok.any():
        return -1, float("inf"), 0
    mse = np.where(ok, sse / np.maximum(cnt, 1.0), np.inf)
    off = int(np.argmin(mse))
    return off, float(np.sqrt(max(mse[off], 0.0))), int(round(cnt[off]))


def refine_offset(ref_xy, ref_ok, qry_xy, qry_ok, coarse, radius=15):
    """(offset, median distance) within coarse +- radius. The median ignores replays, where the
    broadcast ball belongs to another moment of the match."""
    idx = np.flatnonzero(qry_ok)
    best, best_off = np.inf, coarse
    for off in range(coarse - radius, coarse + radius + 1):
        j = idx + off
        k = (j >= 0) & (j < len(ref_xy))
        i2, j2 = idx[k], j[k]
        both = ref_ok[j2]
        if both.sum() < 50:
            continue
        d = np.hypot(*(qry_xy[i2[both]] - ref_xy[j2[both]]).T)
        med = float(np.median(d))
        if med < best:
            best, best_off = med, off
    return best_off, best


def build(verbose=True):
    if not (os.path.exists(paths.BC_ATD_TXT) and os.path.exists(paths.BC_HOMOGRAPHY_CSV)):
        raise SystemExit("broadcast files not found: %s, %s" % (paths.BC_ATD_TXT, paths.BC_HOMOGRAPHY_CSV))
    scale = np.array([paths.PITCH_L, paths.PITCH_W])
    b_xy, b_wide = io.load_broadcast_ball()
    b_xy = b_xy.astype(np.float64) * scale
    b_ok = np.isfinite(b_xy[:, 0]) & b_wide
    t_xy = io.load_atd_raw()[:, -1, :2].astype(np.float64) * scale
    t_ok = np.isfinite(t_xy[:, 0]) & np.load(paths.cache("t0_frames.npz"))["vis"]
    T, NB = len(t_xy), len(b_xy)

    # coarse offset of every 5-minute chunk of broadcast against the whole tactical track
    est = []
    for i0 in range(0, NB - CHUNK + 1, CHUNK):
        off, _, n = find_offset(t_xy, t_ok, b_xy[i0:i0 + CHUNK], b_ok[i0:i0 + CHUNK])
        if n:
            est.append(off - i0)
    est = np.sort(np.array(est))
    if len(est) < 4:
        raise SystemExit("broadcast sync failed: only %d chunks could be aligned" % len(est))
    cut = int(np.argmax(np.diff(est))) + 1                      # the two halves differ by the half-time break
    if np.diff(est).max() < 1000 or cut < 2 or len(est) - cut < 2:
        raise SystemExit("broadcast sync failed: the chunk offsets do not form two halves")
    o1, o2 = int(np.median(est[:cut])), int(np.median(est[cut:]))
    split = ((paths.GRADED[1][1] - o1) + (paths.KICKOFF_2 - o2)) // 2   # first broadcast frame of the 2nd half

    offsets, dist = [], []
    for a, b, o in ((0, split, o1), (split, NB, o2)):
        off, d = refine_offset(t_xy, t_ok, b_xy[a:b], b_ok[a:b], o + a)
        if not d < MAX_DIST_M:
            raise SystemExit("broadcast sync failed: the ball tracks are %.1f m apart after alignment" % d)
        for s in range(a, b, 2 * CHUNK):                         # the offset must hold along the half
            e = min(s + 2 * CHUNK, b)
            loc, dl = refine_offset(t_xy, t_ok, b_xy[s:e], b_ok[s:e], off - a + s, radius=10)
            if np.isfinite(dl) and abs((loc - s) - (off - a)) > MAX_DRIFT:
                raise SystemExit("broadcast sync failed: the offset drifts by %d frames" % ((loc - s) - (off - a)))
        offsets.append(off - a)
        dist.append(d)

    covered = np.zeros(T, bool)
    wide = np.zeros(T, bool)
    bc_ball = np.zeros(T, bool)
    for (a, b), o in zip(((0, split), (split, NB)), offsets):
        i = np.arange(a, b)
        j = i + o
        k = (j >= 0) & (j < T)
        covered[j[k]] = True
        wide[j[k]] = b_wide[i[k]]
        bc_ball[j[k]] = b_ok[i[k]]
    np.savez(paths.cache("t0_broadcast.npz"), covered=covered, wide=wide, bc_ball=bc_ball,
             offsets=np.array(offsets, np.int32), med_dist_m=np.array(dist, np.float32))
    if verbose:
        play = np.zeros(T, bool)
        play[paths.GRADED[1][0]:paths.GRADED[1][1] + 1] = True
        play[paths.KICKOFF_2:paths.GRADED[2][1] + 1] = True
        print("offsets %s (tactical = broadcast + offset) | med_dist_m %s"
              % (offsets, [round(float(x), 2) for x in dist]))
        print("play frames: covered %.4f | wide shot %.3f | broadcast ball %.3f"
              % (covered[play].mean(), wide[play].mean(), bc_ball[play].mean()))
    return dict(covered=covered, wide=wide, bc_ball=bc_ball, offsets=np.array(offsets, np.int32),
                med_dist_m=np.array(dist, np.float32))


def load():
    return dict(np.load(paths.cache("t0_broadcast.npz")))


if __name__ == "__main__":
    build()
