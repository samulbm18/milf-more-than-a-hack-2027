"""T1b: rule-based proxy targets on frames where the ball is really visible (M3).

Default (touch) mode, last-touch possession: 0 OWN_H, 1 OWN_A, -1 ignore. The owner is the team of
the last touch, carried forward while the ball stays visible and live. Dead-ball frames are -1:
the network only learns live play; D comes from the rule events (events.py).
Legacy mode (build_legacy): 0 CTRL_H, 1 CTRL_A, 2 LOOSE, 3 DUEL, 4 DEAD.
Output cache/t1_proxy.npz: sym[T] int8, weight[T] float32, holder[T] int16 (slot or -1).
"""
import numpy as np

from . import events as ev
from . import frames as fr
from . import io, paths

SYMBOLS = ["OWN_H", "OWN_A"]
SYMBOLS_LEGACY = ["CTRL_H", "CTRL_A", "LOOSE", "DUEL", "DEAD"]
# hold / opp_free chosen on the public window (quality gate); see README
DEFAULT_LEGACY = dict(d_ctrl=1.0, d_duel=1.5, d_loose=2.0, dv_ctrl=3.0, opp_free=2.5, hold=10,
                      fast_ball=7.0, dead_conf=0.85)
# touch / switch thresholds: a flat region of the public-window grid (2 folds), not its maximum.
# Frame precision plateaus at ~0.88 for every setting, hence the 0.85 gate.
# v3 (09-11): goalkeeper touches ignored, owner precision 0.878 -> 0.922. Continuity rule (cont_s)
# tested and neutral, left off.
DEFAULT = dict(d_touch=1.5, excl=0.0, dv_ctrl=3.0, hold=2, a_touch=18.0, kick_win=2,
               sw_hold=12, sw_kick=True, fade_s=4.0, dead_conf=0.85, ignore_gk=True)


def ball_geometry(F):
    """Per frame: nearest slot, its distance, nearest distance per team, relative speed."""
    d = np.hypot(F["pos"][..., 0] - F["ball"][:, None, 0], F["pos"][..., 1] - F["ball"][:, None, 1])
    team = F["slot_team"]
    big = np.where(np.isfinite(d), d, np.inf)
    near = np.argmin(big, 1)
    d1 = big[np.arange(len(d)), near]
    dh = big[:, team == 0].min(1)
    da = big[:, team == 1].min(1)
    rel = np.hypot(*(F["ball_vel"] - F["vel"][np.arange(len(d)), near]).T)
    return near, d1, dh, da, rel


def reach_geometry(F, v_ref=6.0, v_base=3.0):
    """Like ball_geometry, but players are ranked by TIME to reach the ball (Link & Hoernig 2017:
    possession is about who can act on the ball within ~1 s). Time = distance / (v_base + speed
    towards the ball); returned as an equivalent distance (time x v_ref) so the metre thresholds of
    the touch detector keep their meaning."""
    dx = F["ball"][:, None, 0] - F["pos"][..., 0]
    dy = F["ball"][:, None, 1] - F["pos"][..., 1]
    d = np.hypot(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        toward = (F["vel"][..., 0] * dx + F["vel"][..., 1] * dy) / np.maximum(d, 1e-3)
    teq = d / (v_base + np.clip(np.nan_to_num(toward), 0, None)) * v_ref
    team = F["slot_team"]
    big = np.where(np.isfinite(teq), teq, np.inf)
    near = np.argmin(big, 1)
    d1 = big[np.arange(len(big)), near]
    dh = big[:, team == 0].min(1)
    da = big[:, team == 1].min(1)
    rel = np.hypot(*(F["ball_vel"] - F["vel"][np.arange(len(big)), near]).T)
    return near, d1, dh, da, rel


def touches(F, c, geom=None):
    """Per frame: team of an unambiguous touch (0/1) or -1, and the touching slot."""
    if geom is None and c.get("reach", False):
        geom = reach_geometry(F)
    near, d1, dh, da, rel = geom or ball_geometry(F)
    vis = F["vis"]
    team = F["slot_team"][near]
    d_opp = np.where(team == 0, da, dh)
    close = vis & (d1 <= c["d_touch"]) & (d_opp >= d1 + c["excl"])
    # contact evidence: held with low relative speed for >= hold frames, or a kick (ball acc peak)
    held = close & (np.nan_to_num(rel, nan=99) <= c["dv_ctrl"])
    same = np.r_[False, near[1:] == near[:-1]]
    keep = np.zeros(len(vis), bool)
    for s, e in zip(*fr.runs(held)):
        k = s
        for t in range(s + 1, e + 1):
            if t == e or not same[t]:
                if t - k >= c["hold"]:
                    keep[k:t] = True
                k = t
    acc = np.nan_to_num(F["ball_acc"])
    w = c["kick_win"]
    peak = np.convolve((acc >= c["a_touch"]).astype(float), np.ones(2 * w + 1), "same") > 0
    touch = close & (keep | peak)
    if c.get("ignore_gk", False):
        # a goalkeeper "touch" in 2D is often a ball passing over or by him (half disagree with GT)
        touch &= ~F["slot_gk"][near]
    if c.get("traj_val", False):
        touch = trajectory_validated(touch, near, F, c)
    tt = np.where(touch, team, -1).astype(np.int8)
    return tt, np.where(touch, near, -1).astype(np.int16), touch & peak


def trajectory_validated(touch, near, F, c):
    """Ball-control validation (Vidal-Codina et al. 2022): a touch episode counts only if the ball
    changes direction (> ba_deg) or speed (> bv_ms) across it, or the player keeps it for
    >= keep_frames. A ball running past a player in a straight line is not a touch."""
    v = F["ball_vel"]
    T = len(touch)
    out = touch.copy()
    s_all, e_all = fr.runs(touch)
    lag = int(c.get("traj_lag", 3))
    for s, e in zip(s_all, e_all):
        if e - s >= c.get("keep_frames", 12):
            continue
        a, b = s - lag, e - 1 + lag
        if a < 0 or b >= T:
            continue
        vin, vout = v[a], v[b]
        if not (np.isfinite(vin).all() and np.isfinite(vout).all()):
            continue  # cannot check: keep it
        sin, sout = np.hypot(*vin), np.hypot(*vout)
        dspeed = abs(sout - sin)
        cosang = np.dot(vin, vout) / max(sin * sout, 1e-6)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1))) if min(sin, sout) > 0.5 else 180.0
        if ang <= c.get("ba_deg", 10.0) and dspeed <= c.get("bv_ms", 3.0):
            out[s:e] = False
    return out


def control_evidence(F, c, geom=None, hold_s=1.0, n_pass=1, max_gap_s=4.0):
    """Frames where a team shows CONTROLLED possession (CHALLENGE.md, possession rules):
    a player keeps the ball for >= hold_s, or the team completes >= n_pass consecutive passes
    (touches by different players of the same team with no opponent touch in between, at most
    max_gap_s apart). Simple touches, deflections and goalkeeper touches are not evidence.
    Returns ev[T] int8: team at the start of the controlling sequence, -1 elsewhere."""
    geom = geom or ball_geometry(F)
    tt, holder, _ = touches(F, dict(c, ignore_gk=True), geom)
    T = len(tt)
    ev = np.full(T, -1, np.int8)
    # touch episodes: (start, end, team, slot)
    eps = []
    t = 0
    while t < T:
        if tt[t] >= 0:
            u = t
            while u < T and tt[u] == tt[t] and holder[u] == holder[t]:
                u += 1
            eps.append((t, u, int(tt[t]), int(holder[t])))
            t = u
        else:
            t += 1
    hold = int(hold_s * paths.FPS)
    gap = int(max_gap_s * paths.FPS)
    chain_start, chain_team, passes, prev = None, -1, 0, None
    for s, e, k, slot in eps:
        if e - s >= hold:
            ev[s] = k  # a player keeps the ball
        if prev is not None and k == chain_team and slot != prev[3] and s - prev[1] <= gap:
            passes += 1
        elif prev is not None and k == chain_team and slot == prev[3] and s - prev[1] <= gap:
            pass  # same player again: dribble, chain continues
        else:
            chain_start, chain_team, passes = s, k, 0
        if passes >= n_pass:
            ev[chain_start] = k
        prev = (s, e, k, slot)
    return ev


def continues(tt, u, e, k, cont, if_none):
    """After a touch run of team k ending at u: is the next touch (within cont frames, same
    segment) also by k? A deflection or a touch without control is followed by the other team."""
    nxt = np.flatnonzero(tt[u:min(e, u + cont)] >= 0)
    if len(nxt) == 0:
        return if_none
    return tt[u + nxt[0]] == k


def carry(tt, kick, segs, c):
    """Last-touch owner with hysteresis: any touch confirms the owner; switching team needs a
    touch run of >= sw_hold frames, or (if sw_kick) a run containing a kick. With cont_s, a switch
    also needs continuity: the next touch must be by the same new team (CHALLENGE.md: a simple
    touch without control or a deflection is not a change of possession)."""
    T = len(tt)
    sym = np.full(T, -1, np.int8)
    last = np.full(T, -1, np.int64)
    cont = int(c.get("cont_s", 0) * paths.FPS)
    for s, e in zip(*segs):
        cur, lt = -1, -1
        t = s
        while t < e:
            k = tt[t]
            if k >= 0:
                u = t
                while u < e and tt[u] == k:
                    u += 1
                strong = (u - t >= c["sw_hold"]) or (c["sw_kick"] and kick[t:u].any())
                if strong and cont and cur >= 0 and k != cur and u - t < c["sw_hold"]:
                    strong = continues(tt, u, e, k, cont, c.get("cont_none", True))
                if cur < 0 or k == cur or strong:
                    cur = k
                    sym[t:u], last[t:u] = cur, np.arange(t, u)
                    lt = u - 1
                elif cur >= 0:
                    sym[t:u], last[t:u] = cur, lt
                t = u
                continue
            if cur >= 0:
                sym[t], last[t] = cur, lt
            t += 1
    return sym, last


def build(cfg=None, verbose=True):
    """Last-touch owner on visible live frames."""
    c = dict(DEFAULT, **(cfg or {}))
    F = fr.load()
    E = ev.load()
    T = len(F["vis"])
    tt, holder, kick = touches(F, c)
    dead = (E["dead"] >= c["dead_conf"]) | F["pre"]
    # carry the owner forward inside each visible live segment; reset at gaps and dead ball
    sym, last = carry(tt, kick, fr.runs(F["vis"] & ~dead), c)
    age = np.arange(T) - last
    weight = np.where(sym >= 0, np.maximum(0.3, 1.0 - age / (c["fade_s"] * paths.FPS)), 0).astype(np.float32)
    holder[(sym < 0) | (tt != sym)] = -1

    np.savez(paths.cache("t1_proxy.npz"), sym=sym, weight=weight, holder=holder)
    if verbose:
        report(F, sym, dead)
    return sym, weight, holder


def proxy_labels(sym, dead):
    """Proxy as a submission: owner where known, D on rule dead, X elsewhere."""
    lab = np.full(len(sym), "X", dtype="<U1")
    lab[sym == 0] = "H"
    lab[sym == 1] = "A"
    lab[dead] = "D"
    return lab


def report(F, sym, dead):
    gt = io.load_gt_public()
    g = F["graded"]
    w = gt != "X"
    vis = F["vis"]
    live = w & (gt != "D")
    lv = g & vis & ~dead
    print("coverage on visible live graded frames: %.3f" % (sym[lv] >= 0).mean())
    print("public live frames labelled: %.3f" % (sym[live] >= 0).mean())
    ph = (gt[live & (sym == 0)] == "H").mean()
    pa = (gt[live & (sym == 1)] == "A").mean()
    lab_on_d = (sym[w & (gt == "D")] >= 0).mean()
    print("owner precision on live public frames: H %.3f  A %.3f | labelled share of GT D %.3f"
          % (ph, pa, lab_on_d))
    # possession changes inside labelled runs vs GT
    s, e = fr.runs(w & (sym >= 0))
    ch = sum(int((np.diff(sym[a:b].astype(int)) != 0).sum()) for a, b in zip(s, e))
    gs, ge = fr.runs(live)
    gch = sum(int((gt[a + 1:b] != gt[a:b - 1]).sum()) for a, b in zip(gs, ge))
    print("owner changes in public window: proxy %d  GT %d" % (ch, gch))
    gate = dict(own_h=ph >= 0.85, own_a=pa >= 0.85, coverage=(sym[lv] >= 0).mean() >= 0.80)
    print("quality gate:", gate, "PASS" if all(gate.values()) else "FAIL")
    return gate


def build_legacy(cfg=None, verbose=True):
    c = dict(DEFAULT_LEGACY, **(cfg or {}))
    F = fr.load()
    E = ev.load()
    T = len(F["vis"])
    vis = F["vis"]
    near, d1, dh, da, rel = ball_geometry(F)
    near_team = F["slot_team"][near]
    d_opp = np.where(near_team == 0, da, dh)
    bs = np.hypot(F["ball_vel"][:, 0], F["ball_vel"][:, 1])

    sym = np.full(T, -1, np.int8)
    weight = np.zeros(T, np.float32)
    holder = np.full(T, -1, np.int16)

    ctrl = vis & (d1 <= c["d_ctrl"]) & (np.nan_to_num(rel, nan=99) <= c["dv_ctrl"]) & (d_opp > c["opp_free"])
    # require the same holder for >= hold consecutive frames
    same = np.r_[False, near[1:] == near[:-1]]
    keep = np.zeros(T, bool)
    for s, e in zip(*fr.runs(ctrl)):
        k = s
        for t in range(s + 1, e + 1):
            if t == e or not same[t]:
                if t - k >= c["hold"]:
                    keep[k:t] = True
                k = t
    duel = vis & (dh <= c["d_duel"]) & (da <= c["d_duel"])
    loose = vis & ((d1 > c["d_loose"]) | ((bs > c["fast_ball"]) & (d1 > c["d_ctrl"])))
    dead = (E["dead"] >= c["dead_conf"]) | F["pre"]

    sym[loose] = 2
    sym[duel] = 3
    sym[keep] = near_team[keep]
    holder[keep] = near[keep]
    sym[dead] = 4

    # confidence: lower near the thresholds
    margin = np.minimum(np.abs(d1 - c["d_ctrl"]), np.abs(d1 - c["d_loose"]))
    weight[sym >= 0] = 1.0
    weight[(sym >= 0) & (sym <= 3) & (margin < 0.2)] = 0.5

    np.savez(paths.cache("t1_proxy.npz"), sym=sym, weight=weight, holder=holder)
    if verbose:
        report_legacy(F, sym)
    return sym, weight, holder


def report_legacy(F, sym):
    gt = io.load_gt_public()
    g = F["graded"]
    w = gt != "X"
    vis = F["vis"]
    print("coverage on visible graded frames: %.3f" % (sym[g & vis] >= 0).mean())
    share = {SYMBOLS_LEGACY[k]: round(float((sym[g] == k).mean()), 3) for k in range(5)}
    print("symbol share (graded):", share)
    rows = []
    for k, name in enumerate(SYMBOLS_LEGACY):
        m = w & (sym == k)
        if m.any():
            dist = {lab: round(float((gt[m] == lab).mean()), 3) for lab in "HAD"}
            rows.append((name, int(m.sum()), dist))
    for r in rows:
        print("public  %-7s n=%5d  GT %s" % r)
    # CTRL while the ball is dead (a player holding it for a restart) is expected and handled by
    # the decoder, so control precision is measured on live frames
    alive = w & (gt != "D")
    ph = (gt[alive & (sym == 0)] == "H").mean()
    pa = (gt[alive & (sym == 1)] == "A").mean()
    print("CTRL precision on live public frames: H %.3f  A %.3f" % (ph, pa))
    pd_ = (gt[w & (sym == 4)] == "D").mean()
    gate = dict(ctrl_h=ph >= 0.92, ctrl_a=pa >= 0.92, dead=pd_ >= 0.90,
                coverage=(sym[g & vis] >= 0).mean() >= 0.60)
    print("quality gate:", gate, "PASS" if all(gate.values()) else "FAIL")
    return gate


def load():
    return dict(np.load(paths.cache("t1_proxy.npz")))


if __name__ == "__main__":
    build()
