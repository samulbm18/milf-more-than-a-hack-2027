"""M5: automaton x explicit durations (Coxian phase-type HSMM expanded into an HMM).

States per half: PRE, E_H^1..E_H^{k-1}, P_H^1..P_H^r, E_A^..., P_A^..., DS^1..DS^rs, DL^1..DL^rl.
Transitions are a fixed list of edges whose probability may depend on the frame through the
deterministic evidence (dead O_t, restart R_t, rho_h). Forward-backward and Viterbi run in log
space over that edge list (numba).
"""
import numpy as np
from numba import njit

from . import paths

LAB_H, LAB_A, LAB_D = 0, 1, 2
CLS = ["EH", "PH", "EA", "PA", "D", "PRE"]
C_INIT = np.array([  # rows: EH, PH, EA, PA, D, PRE ; cols: CTRL_H, CTRL_A, LOOSE, DUEL, DEAD
    [0.80, 0.01, 0.10, 0.08, 0.01],
    [0.55, 0.05, 0.25, 0.13, 0.02],
    [0.01, 0.80, 0.10, 0.08, 0.01],
    [0.05, 0.55, 0.25, 0.13, 0.02],
    [0.08, 0.08, 0.14, 0.02, 0.68],
    [0.08, 0.08, 0.14, 0.02, 0.68],
])

DEFAULT = dict(k=6, r=4, rs=3, rl=3, mean_pos_s=10.0, mean_ds_s=15.0, mean_dl_s=60.0, early_exit=0.25,
               delta0=0.15, kappa=0.8, eps_r=0.05, beta_vis=0.3, beta_hid=0.5, eta_out=0.05,
               eta_in=0.05, lam_vis=0.5, tag_n0=2000.0, out_conf=0.85, gate_eps=1.0, gate_win=12, lam_b=0.0,
               chg=0, tov0=1.0, w_b=0.0, w_t=0.0, w_n=0.0, chg_win=12, swing_s=1.0)


def c_init(prior):
    """Compatibility matrix for the symbol alphabet of the source. With 2 symbols (live-play
    owner) the D / PRE rows equal the prior, so the network says nothing about D."""
    if len(prior) == 5:
        return C_INIT.copy()
    return np.array([[0.90, 0.10], [0.80, 0.20], [0.10, 0.90], [0.20, 0.80], prior, prior])

# edge kinds (K_TURN goes to the away entry, K_TURN_H to the home entry)
K_STATIC, K_TURN, K_TO_DS, K_TO_DL, K_RS_H, K_RS_A, K_DSTAY, K_TURN_H = range(8)


class Topology:
    def __init__(self, k, r, rs, rl):
        self.k, self.r, self.rs, self.rl = k, r, rs, rl
        names, cls, lab = ["PRE"], [5], [LAB_D]
        self.E, self.P = {}, {}
        for team, (ce, cp, lb) in ((0, (0, 1, LAB_H)), (1, (2, 3, LAB_A))):
            self.E[team] = list(range(len(names), len(names) + k - 1))
            names += ["E%s%d" % ("HA"[team], i + 1) for i in range(k - 1)]
            cls += [ce] * (k - 1)
            lab += [lb] * (k - 1)
            self.P[team] = list(range(len(names), len(names) + r))
            names += ["P%s%d" % ("HA"[team], i + 1) for i in range(r)]
            cls += [cp] * r
            lab += [lb] * r
        self.DS = list(range(len(names), len(names) + rs))
        names += ["DS%d" % (i + 1) for i in range(rs)]
        self.DL = list(range(len(names), len(names) + rl))
        names += ["DL%d" % (i + 1) for i in range(rl)]
        cls += [4] * (rs + rl)
        lab += [LAB_D] * (rs + rl)
        self.names, self.cls, self.lab = names, np.array(cls), np.array(lab)
        self.n = len(names)

    def entry(self, team):
        return self.E[team][0] if self.E[team] else self.P[team][0]


def coxian(mean_frames, r, early):
    """Per-phase (stay, advance, exit) with equal phase means and some early exit."""
    m = mean_frames / r
    leave = 1.0 / m
    stay = np.full(r, 1 - leave)
    adv = np.full(r, leave * (1 - early))
    ext = np.full(r, leave * early)
    adv[-1], ext[-1] = 0.0, leave
    return np.stack([stay, adv, ext], 1)


def init_params(p, C0=None):
    f = paths.FPS
    return dict(
        P=[coxian(p["mean_pos_s"] * f, p["r"], p["early_exit"]) for _ in (0, 1)],
        DS=coxian(p["mean_ds_s"] * f, p["rs"], p["early_exit"]),
        DL=coxian(p["mean_dl_s"] * f, p["rl"], p["early_exit"]),
        C=(C_INIT if C0 is None else C0).copy(),
    )


def build_edges(top, prm, p):
    """List of (src, dst, kind, base_prob)."""
    E = []
    for team in (0, 1):
        ch = top.E[team]
        for a, b in zip(ch[:-1], ch[1:]):
            E.append((a, b, K_STATIC, 1.0))
        if ch:
            E.append((ch[-1], top.P[team][0], K_STATIC, 1.0))
        ph = top.P[team]
        for j, s in enumerate(ph):
            st, ad, ex = prm["P"][team][j]
            E.append((s, s, K_STATIC, st))
            if j + 1 < len(ph):
                E.append((s, ph[j + 1], K_STATIC, ad))
            E.append((s, top.entry(1 - team), K_TURN if team == 0 else K_TURN_H, ex))
            E.append((s, top.DS[0], K_TO_DS, ex))
            E.append((s, top.DL[0], K_TO_DL, ex))
    for phases, q in ((top.DS, prm["DS"]), (top.DL, prm["DL"])):
        for j, s in enumerate(phases):
            st, ad, ex = q[j]
            E.append((s, s, K_DSTAY, ex))       # base = exit prob; stay = 1 - adv - exit*u_t
            if j + 1 < len(phases):
                E.append((s, phases[j + 1], K_STATIC, ad))
            E.append((s, top.entry(0), K_RS_H, ex))
            E.append((s, top.entry(1), K_RS_A, ex))
    E.append((0, 0, K_STATIC, 1.0 - 1e-3))
    E.append((0, top.entry(0), K_RS_H, 1e-3))
    E.append((0, top.entry(1), K_RS_A, 1e-3))
    # the advance probability of D phases is needed for the stay edge
    adv = np.zeros(top.n)
    for phases, q in ((top.DS, prm["DS"]), (top.DL, prm["DL"])):
        for j, s in enumerate(phases):
            adv[s] = q[j][1]
    return np.array([(a, b) for a, b, _, _ in E], np.int64), np.array([k for _, _, k, _ in E], np.int64), \
        np.array([w for _, _, _, w in E]), adv


def edge_logprob(src, kind, base, adv, dead, restart, rho_h, p, gate=None):
    """[T, n_edges] log probabilities. gate [T, 2]: multiplier of the turnover towards H / towards A
    (touch gating with the ball visible; 1 = no information)."""
    T = len(dead)
    delta = p["delta0"] + (1 - p["delta0"]) * dead
    u = p["eps_r"] + (1 - p["eps_r"]) * restart
    if gate is None:
        gate = np.ones((T, 2))
    P = np.empty((T, len(kind)))
    for e, (k, b) in enumerate(zip(kind, base)):
        if k == K_STATIC:
            P[:, e] = b
        elif k == K_TURN:
            P[:, e] = b * (1 - delta) * gate[:, 1]
        elif k == K_TURN_H:
            P[:, e] = b * (1 - delta) * gate[:, 0]
        elif k == K_TO_DS:
            P[:, e] = b * delta * p["kappa"]
        elif k == K_TO_DL:
            P[:, e] = b * delta * (1 - p["kappa"])
        elif k == K_RS_H:
            P[:, e] = b * u * rho_h
        elif k == K_RS_A:
            P[:, e] = b * u * (1 - rho_h)
        elif k == K_DSTAY:
            P[:, e] = 1 - adv[src[e]] - b * u
    with np.errstate(divide="ignore"):
        return np.log(np.clip(P, 0, 1))


@njit(cache=True)
def _lae(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    m = a if a > b else b
    return m + np.log1p(np.exp(-abs(a - b)))


@njit(cache=True)
def _forward(log_init, src, dst, logA, logB):
    T, n = logB.shape
    la = np.full((T, n), -np.inf)
    la[0] = log_init + logB[0]
    for t in range(1, T):
        acc = np.full(n, -np.inf)
        for e in range(src.shape[0]):
            acc[dst[e]] = _lae(acc[dst[e]], la[t - 1, src[e]] + logA[t, e])
        la[t] = acc + logB[t]
    return la


@njit(cache=True)
def _backward(src, dst, logA, logB):
    T, n = logB.shape
    lb = np.full((T, n), -np.inf)
    lb[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        acc = np.full(n, -np.inf)
        for e in range(src.shape[0]):
            acc[src[e]] = _lae(acc[src[e]], logA[t + 1, e] + logB[t + 1, dst[e]] + lb[t + 1, dst[e]])
        lb[t] = acc
    return lb


@njit(cache=True)
def _viterbi(log_init, src, dst, logA, logB):
    T, n = logB.shape
    delta = log_init + logB[0]
    back = np.zeros((T, n), np.int64)
    for t in range(1, T):
        new = np.full(n, -np.inf)
        arg = np.zeros(n, np.int64)
        for e in range(src.shape[0]):
            v = delta[src[e]] + logA[t, e]
            if v > new[dst[e]]:
                new[dst[e]] = v
                arg[dst[e]] = src[e]
        delta = new + logB[t]
        back[t] = arg
    path = np.zeros(T, np.int64)
    path[T - 1] = np.argmax(delta)
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


@njit(cache=True)
def _xi_counts(src, dst, logA, logB, la, lb, logZ):
    """Expected count of each edge."""
    T = logB.shape[0]
    c = np.zeros(src.shape[0])
    for t in range(1, T):
        for e in range(src.shape[0]):
            v = la[t - 1, src[e]] + logA[t, e] + logB[t, dst[e]] + lb[t, dst[e]] - logZ
            if v > -50:
                c[e] += np.exp(v)
    return c
