"""T0a: raw ATD -> aligned per-frame arrays in metres, with H always attacking +x.

Output cache/t0_frames.npz (25 Hz, row = video frame):
    pos[T,S,2]  vel[T,S,2]  acc[T,S]     players, metres / m/s / m/s^2, NaN when absent
    trk_age[T,S]                         seconds since the slot's current tracklet started
    ball[T,2]  ball_vel[T,2]  ball_acc[T]
    vis[T]                               ball observed and not a suspected spare ball
    ball_suspect[T]
    graded[T]  half[T]  pre[T]
    flip[3]                              x sign applied per half (index 1, 2)
    slot_team[S]  slot_gk[S]
"""
import numpy as np
from scipy.signal import savgol_filter

from . import io, paths

FPS = paths.FPS


def runs(mask):
    """(starts, ends) of consecutive True runs, end exclusive."""
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def smooth_derivatives(p, present, window):
    """Savitzky-Golay velocity and acceleration magnitude of one track, run by run."""
    v = np.full(p.shape, np.nan, np.float32)
    acc = np.full(p.shape[0], np.nan, np.float32)
    for s, e in zip(*runs(present)):
        seg = p[s:e].astype(np.float64)
        n = e - s
        if n >= window:
            d1 = savgol_filter(seg, window, 2, deriv=1, delta=1 / FPS, axis=0)
            d2 = savgol_filter(seg, window, 2, deriv=2, delta=1 / FPS, axis=0)
        elif n >= 2:
            d1 = np.gradient(seg, 1 / FPS, axis=0)
            d2 = np.zeros_like(seg)
        else:
            d1 = np.zeros_like(seg)
            d2 = np.zeros_like(seg)
        v[s:e] = d1
        acc[s:e] = np.hypot(d2[:, 0], d2[:, 1])
    return v, acc


def build(verbose=True):
    meta = io.load_meta()
    raw = io.load_atd_raw()
    T = raw.shape[0]
    slot_team, slot_gk = meta["slot_team"], meta["slot_gk"]
    S = len(slot_team)

    xy = raw[:, :S, :2].astype(np.float32)
    xy[..., 0] = (xy[..., 0] - 0.5) * paths.PITCH_L
    xy[..., 1] = (xy[..., 1] - 0.5) * paths.PITCH_W

    graded = np.zeros(T, bool)
    half = np.zeros(T, np.int8)
    for h, (a, b) in paths.GRADED.items():
        graded[a:b + 1] = True
        half[a:b + 1] = h
    # extend half labels over the ungraded margins so every frame has a side
    half[:paths.GRADED[1][0]] = 1
    half[paths.GRADED[1][1] + 1:paths.GRADED[2][0]] = 2
    half[paths.GRADED[2][1] + 1:] = 2
    pre = np.zeros(T, bool)
    pre[paths.GRADED[2][0]:paths.KICKOFF_2] = True

    # attack direction: the home goalkeeper stands in the goal H defends
    flip = np.ones(3, np.float32)
    gk_h = np.flatnonzero(slot_gk & (slot_team == 0))
    report = {}
    for h in (1, 2):
        m = graded & (half == h)
        gx = np.nanmean(xy[m][:, gk_h, 0])
        flip[h] = 1.0 if gx < 0 else -1.0  # H defends -x  ->  H attacks +x already
        report["half%d_home_gk_mean_x" % h] = round(float(gx), 1)
    xml_dir = meta["team_params"][0].get("attack_direction_first_half", "")
    report["xml_home_first_half"] = xml_dir
    report["xml_consistent"] = bool((flip[1] == 1.0) == (xml_dir == "left_to_right"))
    assert flip[1] == -flip[2], "attack direction should swap at half-time"
    for h in (1, 2):
        xy[half == h, :, 0] *= flip[h]

    present = np.isfinite(xy[..., 0])
    vel = np.full((T, S, 2), np.nan, np.float32)
    acc = np.full((T, S), np.nan, np.float32)
    trk_age = np.full((T, S), np.nan, np.float32)
    n_trk = 0
    for j in range(S):
        vel[:, j], acc[:, j] = smooth_derivatives(xy[:, j], present[:, j], 15)
        for s, e in zip(*runs(present[:, j])):
            trk_age[s:e, j] = np.arange(e - s, dtype=np.float32) / FPS
            n_trk += 1

    ball = raw[:, S, :2].astype(np.float32)
    ball[:, 0] = (ball[:, 0] - 0.5) * paths.PITCH_L
    ball[:, 1] = (ball[:, 1] - 0.5) * paths.PITCH_W
    for h in (1, 2):
        ball[half == h, 0] *= flip[h]
    bpres = np.isfinite(ball[:, 0])
    bvel, bacc = smooth_derivatives(ball, bpres, 9)

    # spare balls: well outside the pitch and not moving for a second
    out_far = (np.abs(ball[:, 0]) > paths.HALF_L + 5) | (np.abs(ball[:, 1]) > paths.HALF_W + 5)
    still = np.hypot(bvel[:, 0], bvel[:, 1]) < 0.3
    cand = bpres & out_far & still
    suspect = np.zeros(T, bool)
    for s, e in zip(*runs(cand)):
        if e - s >= FPS:
            suspect[s:e] = True
    vis = bpres & ~suspect

    out = dict(pos=xy, vel=vel, acc=acc, trk_age=trk_age, ball=ball, ball_vel=bvel, ball_acc=bacc,
               vis=vis, ball_suspect=suspect, graded=graded, half=half, pre=pre, flip=flip,
               slot_team=slot_team, slot_gk=slot_gk)
    np.savez(paths.cache("t0_frames.npz"), **out)

    if verbose:
        gs, ge = runs(~vis & graded)
        sp = np.hypot(vel[..., 0], vel[..., 1])[graded][present[graded]]
        bsp = np.hypot(bvel[:, 0], bvel[:, 1])[graded & vis]
        report.update({
            "frames": T, "slots": S, "tracklets": n_trk,
            "ball_visible_graded": round(float(vis[graded].mean()), 4),
            "ball_gaps_graded": int(len(gs)),
            "gap_median_s": round(float(np.median(ge - gs) / FPS), 2),
            "ball_suspect_frames": int(suspect.sum()),
            "player_speed_p50_p99": [round(float(x), 2) for x in np.nanpercentile(sp, [50, 99])],
            "ball_speed_p50_p99": [round(float(x), 2) for x in np.nanpercentile(bsp, [50, 99])],
            "mean_objects_graded": round(float(present[graded].sum(1).mean()), 2),
        })
        for k, v in report.items():
            print("%-26s %s" % (k, v))
    return out


def load():
    return dict(np.load(paths.cache("t0_frames.npz")))


if __name__ == "__main__":
    build()
