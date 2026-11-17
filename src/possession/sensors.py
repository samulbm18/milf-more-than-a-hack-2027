"""Detectors as noisy sensors of the hidden state (H / A / D) for the M5 machine.

The network and the rules only DETECT; the machine decides. Each sensor emits a discrete symbol
on the 5 Hz step frames (so correlated 25 Hz output is counted once) or -1 when it has nothing to
say. Its reliability is a confusion matrix E[label, symbol] = P(symbol | true label), bounded and
learnt by EM on the whole match (never on the ground truth). H/A rows are tied by a symbol swap
(team symmetry).
"""
import numpy as np

from . import paths, proxy

STEP = 5
LAB_H, LAB_A, LAB_D = 0, 1, 2


class Sensor:
    def __init__(self, name, obs, E0, swap, alpha, cap=None):
        self.name, self.obs, self.swap, self.alpha, self.cap = name, obs, np.asarray(swap), alpha, cap
        self.E0 = np.asarray(E0, float)
        self.E0 /= self.E0.sum(1, keepdims=True)
        self.E = self.E0.copy()

    def loglik(self, sl):
        """[T_sl, 3] log P(obs | label); 0 where the sensor is silent."""
        o = self.obs[sl]
        out = np.zeros((len(o), 3))
        m = o >= 0
        out[m] = np.log(self.E[:, o[m]].T)
        return out

    def counts(self, gamma_lab, sl):
        o = self.obs[sl]
        n = np.zeros_like(self.E)
        m = o >= 0
        for j in range(self.E.shape[1]):
            n[:, j] = gamma_lab[m & (o == j)].sum(0)
        return n

    def update(self, n):
        E = n + self.alpha * self.E0
        E /= E.sum(1, keepdims=True)
        sym = 0.5 * (E[LAB_H] + E[LAB_A][self.swap])
        E[LAB_H], E[LAB_A] = sym, sym[self.swap]
        if self.cap:
            # bounded sensor: P(o|H) / P(o|A) within [1/cap, cap] for every symbol, so that EM cannot
            # turn a detector into the decision maker. Shrink H/A rows towards their mean.
            for _ in range(50):
                r = E[LAB_H] / E[LAB_A]
                if r.max() <= self.cap * 1.0001 and r.min() >= 1 / self.cap / 1.0001:
                    break
                m = 0.5 * (E[LAB_H] + E[LAB_A])
                E[LAB_H], E[LAB_A] = 0.9 * E[LAB_H] + 0.1 * m, 0.9 * E[LAB_A] + 0.1 * m
        self.E = np.maximum(E, 1e-4)
        self.E /= self.E.sum(1, keepdims=True)


def on_steps(x):
    """Keep a per-frame symbol only on step frames."""
    out = np.full(len(x), -1, np.int64)
    out[::STEP] = x[::STEP]
    return out


def net_symbols(q, edges=(0.1, 0.35, 0.65, 0.9)):
    """q_H -> 0 'A sure' .. 4 'H sure'."""
    return np.digitize(q[:, 0], edges).astype(np.int64)


# the middle dead-rule level (0.3-0.85, precision ~0.5) starts neutral: only clear rule evidence
# moves the machine towards D (10-11-2026: LR 4 -> 1 on that level, spellF1 v3 0.579 -> 0.613 with EM)
# Smart Tagging codes that describe a team's ACTION WITH THE BALL (its team tag names the team in
# possession). Zone descriptors (BALL IN FINAL THIRD, BALL IN THE BOX, PLAYERS IN THE BOX) are left
# out by meaning; their team tag agrees with the public owner only ~45-54% of the time.
ACTION_TAGS = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "LONG BALL", "ATTACKING TRANSITION", "SET PIECES")


def tag_sensors(inp, alpha, cap, codes=ACTION_TAGS, every=STEP):
    """One sensor per code: 0 no tag, 1 tag names H, 2 tag names A (N/A team -> no tag)."""
    from . import io
    out = []
    T = len(inp.vis)
    for code in codes:
        sym = np.zeros(T, np.int64)
        for t in inp.tags:
            if t["code"] != code:
                continue
            k = io.tag_team_index(t["team"], inp.team_names)
            if k >= 0:
                sym[t["f0"]:t["f1"]] = 1 + k
        obs = np.full(T, -1, np.int64)
        obs[::every] = sym[::every]  # a phase lasts seconds: read it sparsely
        out.append(Sensor("tag:" + code, obs,
                          [[.90, .075, .025], [.90, .025, .075], [.90, .05, .05]], [0, 2, 1], alpha, cap=cap))
    return out


DEAD_RULE_E0 =[[.88, .10, .02], [.88, .10, .02], [.50, .10, .40]]
MACHINE_USE = ("net_vis", "net_hid", "touch", "dead_rule")

# Broadcast picture as a dead-ball sensor (experiment 16). When the TV director leaves the wide shot
# (replay, close-up) the ball is almost always dead. He reacts ~2-4 s after play stops and comes back
# ~1-3 s after it restarts, so the cut is read BC_LEAD frames early (2.5 s: the middle of the flat
# region 1.5-3.5 s measured on the public window, not its maximum).
# Symbols: 0 wide shot with the ball in the broadcast tracking, 1 neither, 2 cut-away.
# Rows H, A, D. On the public window with this lead the rates are [.905 .089 .006] live and
# [.016 .118 .867] dead; the matrix is deliberately softer (likelihood ratio 13 for D on a cut, 6.5 for
# live play on symbol 0; the top level of the dead-rule sensor has 20) and is re-estimated by the EM.
BC_LEAD = 62
BC_CUT_E0 = [[.78, .17, .05], [.78, .17, .05], [.12, .23, .65]]
MACHINE_USE_BC = MACHINE_USE + ("bc_cut",)


def bc_symbols(wide, bc_ball, covered, lead=BC_LEAD):
    """Per-frame symbol of the broadcast picture (see BC_CUT_E0); -1 where the broadcast has no frame
    for that instant. The rules' restart detector is deliberately not allowed to silence a cut: with
    the ball in view a player moving it during the stoppage looks like a kick (it silenced 6.5 min of
    cut-aways, among them most of a 20 s public stoppage)."""
    n = len(wide)
    cut = np.zeros(n, bool)
    cut[:n - lead] = covered[lead:] & ~wide[lead:]
    sym = np.where(cut, 2, np.where(bc_ball & wide, 0, 1)).astype(np.int64)
    sym[~covered] = -1
    return sym


# ---- experiment 17 (not part of the submitted configuration): stoppage edges and quick restarts ----
# A SET PIECES tag starts while the ball is still dead (3-12 s before the restart on the two public
# ones). Symbol 1 inside a window around the tag start; rows H, A, D.
SP_BEFORE, SP_AFTER = 50, 50
SET_PIECE_E0 = [[.985, .015], [.985, .015], [.90, .10]]


def quiet_after_cut(sym, frames):
    """Broadcast symbols with the sensor silent for `frames` frames after each cut-away: the director
    may be back on the wide shot before the restart, so there the picture says nothing about play."""
    out = sym.copy()
    d = np.diff(np.concatenate([[0], (sym == 2).astype(np.int8), [0]]))
    for b in np.flatnonzero(d == -1):
        seg = out[b:b + frames]
        seg[seg != 2] = -1
    return out


def set_piece_symbols(n, tag_starts, before=SP_BEFORE, after=SP_AFTER):
    """[n] int64: 1 inside [start - before, start + after) of each tagged set piece, else 0."""
    s = np.zeros(n, np.int64)
    for f0 in tag_starts:
        s[max(f0 - before, 0):f0 + after] = 1
    return s


def build(inp, F, alpha=2000.0, use=("net_vis", "net_hid", "touch", "dead_rule", "dead_model"), cap=None,
          dead_rule_E0=None, tag_every=STEP, tag_cap=None, touch_cfg=None, alpha_bc=None, bc_wait=0):
    vis = inp.vis
    sens = []
    sp = None
    if "set_piece" in use:
        sp = set_piece_symbols(len(vis), [t["f0"] for t in inp.tags if t["code"] == "SET PIECES"])

    def mk(*a):
        return Sensor(*a, cap=cap)
    ns = net_symbols(inp.q)
    rev5 = [4, 3, 2, 1, 0]
    if "net_vis" in use:
        sens.append(mk("net_vis", on_steps(np.where(vis, ns, -1)),
                           [[.03, .04, .10, .23, .60], [.60, .23, .10, .04, .03], [.2, .2, .2, .2, .2]], rev5, alpha))
    if "net_hid" in use:
        sens.append(mk("net_hid", on_steps(np.where(~vis, ns, -1)),
                           [[.05, .10, .25, .30, .30], [.30, .30, .25, .10, .05], [.2, .2, .2, .2, .2]], rev5, alpha))
    if "touch" in use:
        tt = proxy.touches(F, dict(proxy.DEFAULT, **(touch_cfg or {})))[0]
        sym = np.where(tt == 0, 1, np.where(tt == 1, 2, 0))
        sym = np.where(vis, sym, -1)  # no ball, no touch information
        sens.append(mk("touch", on_steps(sym),
                           [[.80, .15, .05], [.80, .05, .15], [.97, .015, .015]], [0, 2, 1], alpha))
    if "dead_rule" in use:
        d = np.digitize(inp.dead_rule, (0.3, 0.85))
        sens.append(mk("dead_rule", on_steps(d), dead_rule_E0 or DEAD_RULE_E0, [0, 1, 2], alpha))
    if "bc_cut" in use:
        bc = getattr(inp, "bc", None)
        if bc is None:
            raise SystemExit("the bc_cut sensor needs cache/t0_broadcast.npz: run  python -m possession.sync")
        sym = bc_symbols(bc["wide"], bc["bc_ball"], bc["covered"])
        if bc_wait:
            sym = quiet_after_cut(sym, bc_wait)
        if sp is not None:
            sym[sp == 1] = -1  # next to a tagged set piece the wide shot says nothing about play
        sens.append(mk("bc_cut", on_steps(sym), BC_CUT_E0, [0, 1, 2], alpha_bc or alpha))
    if sp is not None:
        sens.append(mk("set_piece", on_steps(sp), SET_PIECE_E0, [0, 1], alpha))
    if "tags" in use:
        sens += tag_sensors(inp, alpha, tag_cap or cap, every=tag_every)
    if "dead_model" in use and inp.p_dead is not None:
        d = np.digitize(inp.p_dead, (0.5, 0.8))
        sens.append(mk("dead_model", on_steps(d), [[.80, .12, .08], [.80, .12, .08], [.30, .20, .50]], [0, 1, 2], alpha))
    return sens
