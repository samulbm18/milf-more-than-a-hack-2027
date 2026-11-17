"""T1a: deterministic dead-ball and restart evidence (M2).

Only `pre` is a hard fact. Everything else is soft evidence for the decoder:
    dead[T]    probability that the ball is dead (0 where no detector fires)
    restart[T] gaussian bump around each detected restart, scaled by its confidence
    rho_h[T]   probability that the home team restarts, held over each bump
plus the list of detected intervals.
"""
import numpy as np

from . import frames as fr
from . import io, paths

FPS = paths.FPS
TAG_WINDOW_S = 35
SEARCH_S = 120
# "out of play -> the opponents of the last toucher restart": agrees with the first touch after the
# restart only 50% of the time (26 restarts, 10-11-2026), i.e. no signal with these detectors.
OUT_LAW_W = 0.0


def _sustained(mask, n):
    """True on runs of `mask` that last at least n frames."""
    out = np.zeros_like(mask)
    for s, e in zip(*fr.runs(mask)):
        if e - s >= n:
            out[s:e] = True
    return out


def _smooth(x, n):
    k = np.ones(n) / n
    return np.convolve(np.nan_to_num(x, nan=np.nanmean(x)), k, "same")


def dist_to_line(p):
    return np.minimum(paths.HALF_L - np.abs(p[..., 0]), paths.HALF_W - np.abs(p[..., 1]))


def candidates(F, tags):
    """Dead-ball start candidates as (frame, confidence, source)."""
    vis, b, bv = F["vis"], F["ball"], F["ball_vel"]
    bs = np.hypot(bv[:, 0], bv[:, 1])
    ms = _smooth(np.nanmean(np.hypot(F["vel"][..., 0], F["vel"][..., 1]), 1), FPS)
    out = []

    # observed crossing: ball visibly beyond a line (+0.5 m) for >= 5 frames
    beyond = vis & (dist_to_line(np.nan_to_num(b)) < -0.5)
    for s, e in zip(*fr.runs(beyond)):
        if e - s >= 5:
            out.append((s, 1.0, "out_obs"))

    # inferred crossing: ball vanishes near a line moving outwards and stays hidden >= 2 s
    gs, ge = fr.runs(~vis)
    for s, e in zip(gs, ge):
        if s == 0 or e - s < 2 * FPS:
            continue
        p, v = b[s - 1], bv[s - 1]
        if not np.isfinite(p).all() or dist_to_line(p) > 1.5:
            continue
        outward = max(np.sign(p[0]) * v[0], np.sign(p[1]) * v[1])
        if outward > 2.0:
            frozen = ms[s:min(e, s + 3 * FPS)].min() < 1.5
            out.append((s, 0.85 if frozen else 0.6, "out_inf"))

    # still ball and frozen players for >= 3 s
    bs_s = _smooth(np.where(vis, bs, np.nan), FPS // 2)
    calm = _sustained(vis & (bs_s < 1.5) & (ms < 1.4), 3 * FPS)
    for s, e in zip(*fr.runs(calm)):
        out.append((s, 0.7, "calm"))

    # sudden collective slowdown (whistle): >2.2 m/s to <1.3 m/s within 3 s, held 2 s
    slow = _sustained(ms < 1.3, 2 * FPS)
    for s, e in zip(*fr.runs(slow)):
        if s > 3 * FPS and ms[s - 3 * FPS:s].max() > 2.2:
            out.append((s, 0.5, "slowdown"))

    # set-piece tags open a search window
    for t in tags:
        if t["code"] == "SET PIECES":
            out.append((t["f0"], 0.6, "tag"))
    return sorted(out), bs, ms


def find_restart(c, F, bs, ms, horizon):
    """First restart after candidate start c, or (None, 0).

    Around the public restarts the ball is slow or hidden while players walk (~1 m/s); the
    restart is the first kick (ball > 4 m/s for 3 frames) that follows >= 1 s of that calm.
    """
    vis = F["vis"]
    lo = c + FPS
    end = min(c + horizon, len(vis) - 1)
    if end <= lo:
        return None, 0.0
    seg = slice(lo, end)
    v, s, m = vis[seg], bs[seg], ms[seg]
    calm = _sustained((~v | (s < 1.5)) & (m < 1.6), FPS)
    calm_recent = np.convolve(calm.astype(float), np.ones(2 * FPS), "full")[:len(calm)] > 0
    kick = _sustained(v & (s > 4.0), 3)
    hit = np.flatnonzero(kick & calm_recent & ~calm)
    if hit.size:
        return lo + int(hit[0]), 0.9
    up = np.flatnonzero((m[2 * FPS:] > 2.2) & (np.minimum.accumulate(m)[:-2 * FPS] < 1.3)) if m.size > 2 * FPS else []
    if len(up):
        return lo + 2 * FPS + int(up[0]), 0.5
    return None, 0.0


def last_touch_team(a, tt, window_s=5.0):
    """Team (0/1) of the last touch in the window_s before frame a, or -1."""
    seg = tt[max(0, a - int(window_s * FPS)):a]
    hit = np.flatnonzero(seg >= 0)
    return int(seg[hit[-1]]) if hit.size else -1


def restart_team(t, F, tags, team_names, start=None, src=(), tt=None):
    """P(home restarts) from the nearest player, any covering set-piece tag and two laws of the game:
    a ball out of play is restarted by the opponents of the last team to touch it, and after a goal
    the team that conceded kicks off."""
    logit = 0.0
    if tt is not None and start is not None:
        if {"out_obs", "out_inf"} & set(src):
            k = last_touch_team(start, tt)
            if k >= 0:
                logit += -OUT_LAW_W if k == 0 else OUT_LAW_W
        for tg in tags:
            if tg["code"] == "GOALS" and tg["f0"] - 10 * FPS <= start <= tg["f1"] + 10 * FPS:
                k = io.tag_team_index(tg["team"], team_names)
                if k >= 0:
                    logit += -3.0 if k == 0 else 3.0
    vis, b = F["vis"], F["ball"]
    for u in range(t, min(t + FPS, len(vis))):
        if vis[u]:
            d = np.hypot(*(F["pos"][u] - b[u]).T)
            if np.isfinite(d).any() and np.nanmin(d) < 2.5:
                logit += 2.0 if F["slot_team"][int(np.nanargmin(d))] == 0 else -2.0
            break
    for tg in tags:
        if tg["code"] == "SET PIECES" and tg["f0"] - FPS <= t <= tg["f0"] + TAG_WINDOW_S * FPS:
            k = io.tag_team_index(tg["team"], team_names)
            if k >= 0:
                logit += 1.0 if k == 0 else -1.0
    return 1.0 / (1.0 + np.exp(-logit))


def build(verbose=True):
    F = fr.load()
    tags = io.load_tags()
    team_names = io.load_meta()["team_names"]
    T = len(F["vis"])
    cands, bs, ms = candidates(F, tags)

    # each candidate -> interval [start, restart); overlapping intervals are merged
    iv = []
    for c, conf, src in cands:
        if F["pre"][c]:
            continue
        r, rc = find_restart(c, F, bs, ms, SEARCH_S * FPS)
        if r is None:
            r, rc = min(c + 30 * FPS, T), 0.0
        iv.append([c, r, conf, rc, {src}])
    iv.sort()
    merged = []
    for a in iv:
        if merged and a[0] <= merged[-1][1]:
            m = merged[-1]
            # same stoppage: keep its restart unless it never found one
            if m[3] == 0.0 and a[1] > m[1]:
                m[1], m[3] = a[1], a[3]
            m[2] = max(m[2], a[2])
            m[4] |= a[4]
        else:
            merged.append(a)
    # second-half kickoff is known
    merged = [m for m in merged if not (paths.GRADED[2][0] <= m[0] < paths.KICKOFF_2)]
    merged.append([paths.GRADED[2][0], paths.KICKOFF_2, 1.0, 1.0, {"pre"}])
    merged.sort()

    dead = np.zeros(T, np.float32)
    restart = np.zeros(T, np.float32)
    rho_h = np.full(T, 0.5, np.float32)
    w = np.arange(-2 * FPS, 2 * FPS + 1)
    kern = np.exp(-0.5 * (w / (0.5 * FPS)) ** 2)
    from . import proxy  # local import: proxy imports this module
    tt = proxy.touches(F, dict(proxy.DEFAULT))[0]
    events = []
    for a, r, conf, rc, src in merged:
        if len(src) > 1:
            conf = max(conf, 0.9)
        dead[a:r] = np.maximum(dead[a:r], conf)
        ph = restart_team(r, F, tags, team_names, a, src, tt) if r < T else 0.5
        if rc > 0:
            lo, hi = max(r + w[0], 0), min(r + w[-1] + 1, T)
            restart[lo:hi] = np.maximum(restart[lo:hi], rc * kern[lo - r - w[0]:hi - r - w[0]])
            rho_h[lo:hi] = ph
        events.append((a, r, conf, rc, ph, "+".join(sorted(src))))
    dead[F["pre"]] = 1.0

    np.savez(paths.cache("t1_events.npz"), dead=dead, restart=restart, rho_h=rho_h,
             ev=np.array([e[:5] for e in events], np.float64),
             ev_src=np.array([e[5] for e in events]))
    if verbose:
        report(F, dead, events)
    return dead, restart, rho_h, events


def report(F, dead, events):
    gt = io.load_gt_public()
    g = F["graded"]
    w = gt != "X"
    minutes = g.sum() / FPS / 60
    srcs = io.counts(e[5] for e in events)
    print("dead intervals: %d (%.2f per graded minute); sources %s" % (len(events), len(events) / minutes, srcs))
    print("dead share of graded frames (>=0.5): %.3f" % (dead[g] >= 0.5).mean())
    for th in (0.5, 0.85):
        p = (dead >= th) & w
        prec = (gt[p] == "D").mean() if p.any() else float("nan")
        rec = p[gt == "D"].mean()
        print("public window, dead>=%.2f: precision %.3f recall %.3f" % (th, prec, rec))
    # restart timing against GT D -> alive transitions
    s, e = fr.runs(gt == "D")
    rs = np.array([ev[1] for ev in events if ev[3] > 0])
    for a, z in zip(s, e):
        err = (rs - z) / FPS
        j = np.argmin(np.abs(err)) if rs.size else None
        ph = [ev[4] for ev in events if ev[3] > 0][j] if j is not None else None
        print("GT restart -> %s: nearest detected %+.1f s, P(home)=%.2f" % (gt[z], err[j], ph))


def load():
    return dict(np.load(paths.cache("t1_events.npz")))


if __name__ == "__main__":
    build()
