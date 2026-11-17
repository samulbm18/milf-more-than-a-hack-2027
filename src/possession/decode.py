"""Run the M5 decoder on a symbol source, fit durations by EM, score on the public GT.

    python -m possession.decode --source proxy              # decoder v0 (no network)
    python -m possession.decode --source runs/<run>/oof.npz # with PossNet symbols
"""
import argparse
import json
import os
import sys

import numpy as np

from . import decoder as dc
from . import events as ev
from . import frames as fr
from . import io, paths, proxy, sync

sys.path.insert(0, os.path.join(paths.MTAH_ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

LABELS = np.array(["H", "A", "D"])
# the machine's configuration: one entry phase (k=1), 3 s mean possession
# prior, turnover hazard driven by the network's border head (chg=1, w_b=2)
MACHINE_P = dict(k=1, mean_pos_s=3.0, chg=1, tov0=1.0, w_b=2.0)


# ------------------------------------------------------------------------------- inputs
def q_from_proxy(sym, conf=0.9, n=2):
    q = np.full((len(sym), n), 1 / n)
    m = sym >= 0
    q[m] = (1 - conf) / (n - 1)
    q[m, sym[m]] = conf
    return q


def q_from_oof(path):
    """5 Hz log q -> 25 Hz probabilities by linear interpolation."""
    z = np.load(path)
    lq = z["logq"]
    steps = np.arange(0, paths.N_FRAMES, 5)
    t = np.arange(paths.N_FRAMES)
    q = np.stack([np.interp(t, steps, np.exp(lq[:, j])) for j in range(lq.shape[1])], 1)
    return q / q.sum(1, keepdims=True)


def border_from_oof(path):
    """5 Hz border probability (a possession change within +-2 steps) -> 25 Hz."""
    b = np.load(path)["border"]
    return np.interp(np.arange(paths.N_FRAMES), np.arange(0, paths.N_FRAMES, 5), b)


def vis_loglik(gt, vis, n_eff_div=250, prior_n=20):
    """log P(visible | label) and log P(hidden | label), shrunk toward the match-wide rate."""
    w = gt != "X"
    pbar = vis[w].mean()
    out = np.zeros((2, 3))
    for j, lab in enumerate("HAD"):
        m = gt == lab
        n = m.sum() / n_eff_div
        p = (vis[m].mean() * n + prior_n * pbar) / (n + prior_n)
        out[0, j], out[1, j] = np.log(1 - p), np.log(p)  # row 0 hidden, row 1 visible
    # cap the H/A ratio
    for r in (0, 1):
        out[r, :2] = np.clip(out[r, :2], out[r, :2].max() - np.log(5), None)
    return out


def tag_bias(tags, team_names, gt, n0):
    """[T, 2] additive log bias for home-label / away-label states from team tags."""
    T = paths.N_FRAMES
    bias = np.zeros((T, 2))
    weights = {}
    for code in sorted({t["code"] for t in tags}):
        if code == "SET PIECES":
            continue
        agree = n = 0
        for t in tags:
            k = io.tag_team_index(t["team"], team_names)
            if t["code"] != code or k < 0:
                continue
            g = gt[t["f0"]:t["f1"]]
            live = (g == "H") | (g == "A")
            n += live.sum()
            agree += (g[live] == "HA"[k]).sum()
        if n == 0:
            weights[code] = 0.0
            continue
        p = np.clip(agree / n, 0.02, 0.98)
        weights[code] = float(n / (n + n0) * np.log(p / (1 - p)))
    for t in tags:
        k = io.tag_team_index(t["team"], team_names)
        if k >= 0 and t["code"] in weights:
            bias[t["f0"]:t["f1"], k] += weights[t["code"]]
    return bias, weights


class Inputs:
    def __init__(self, q, prior, dead_model=None, dm_w=0.0):
        F = fr.load()
        E = ev.load()
        self.q, self.prior = q, prior
        self.vis = F["vis"]
        self.pre = F["pre"]
        self.dead_rule, self.restart, self.rho_h = E["dead"], E["restart"], E["rho_h"]
        self.p_dead = np.load(dead_model)["p_dead"] if dead_model else None
        self.set_dead_weight(dm_w)
        self.touch_team = proxy.touches(F, dict(proxy.DEFAULT))[0]
        self.control_ev = None  # set to proxy.control_evidence(...) to gate turnovers on control
        self.border = None      # set to border_from_oof(...) to modulate the turnover hazard
        self.sensors = None     # set to sensors.build(...) to decode from noisy sensors
        # the broadcast picture on the tactical timeline (sync.py), for the bc_cut sensor; None if not built
        self.bc = sync.load() if os.path.exists(paths.cache("t0_broadcast.npz")) else None
        self.gt = io.load_gt_public()
        self.tags = io.load_tags()
        self.team_names = io.load_meta()["team_names"]

    def set_dead_weight(self, w):
        """Dead-ball evidence: rules, raised by the dead model (deadmodel.py) scaled by w."""
        self.dead = self.dead_rule if self.p_dead is None else np.maximum(self.dead_rule, w * self.p_dead)


def change_gate(inp, p):
    """[T, 2] turnover multipliers towards H / A from coinciding change evidence:
        tov0 * exp(w_b log(b_t / mean b) + w_t [touch by k within +-chg_win] + w_n (q_k(t+L) - q_k(t-L)))
    border = the network's change detector, touch = geometry, the q swing = the network turning
    towards k. With tov0 < 1 a change without supporting evidence becomes expensive."""
    T = len(inp.vis)
    z = np.zeros((T, 2))
    if inp.border is not None and p["w_b"]:
        b = np.clip(inp.border, 1e-3, 1)
        z += p["w_b"] * np.log(b / b.mean())[:, None]
    if p["w_t"]:
        k = np.ones(2 * int(p["chg_win"]) + 1)
        for j in (0, 1):
            z[:, j] += p["w_t"] * (np.convolve((inp.touch_team == j).astype(float), k, "same") > 0)
    if p["w_n"]:
        L = int(p["swing_s"] * paths.FPS)
        for j in (0, 1):
            qj = inp.q[:, j]
            fut = np.r_[qj[L:], np.repeat(qj[-1], L)]
            past = np.r_[np.repeat(qj[0], L), qj[:-L]]
            z[:, j] += p["w_n"] * (fut - past)
    return p["tov0"] * np.exp(np.clip(z, -6, 6))


def touch_gate(inp, p):
    """[T, 2] turnover multipliers towards H / A. With the ball visible a turnover towards team k
    needs a touch by k within +-gate_win frames; otherwise it is scaled by gate_eps. Where the
    ball is hidden (or near a gap) there is no information: 1."""
    T = len(inp.vis)
    g = np.ones((T, 2))
    if inp.border is not None and p["lam_b"] > 0:
        # network "a change happens here" head: turnover hazard x (b_t / mean b)^lam_b
        b = np.clip(inp.border, 1e-3, 1)
        g *= np.clip((b / b.mean()) ** p["lam_b"], 0.05, 20.0)[:, None]
    if p["gate_eps"] >= 1.0:
        return g
    w = int(p["gate_win"])
    k = np.ones(2 * w + 1)
    src = inp.touch_team if inp.control_ev is None else inp.control_ev
    near = np.stack([np.convolve((src == j).astype(float), k, "same") > 0 for j in (0, 1)], 1)
    seen = np.convolve((~inp.vis).astype(float), k, "same") == 0
    gc = np.where(near, 1.0, p["gate_eps"])
    gc[~seen] = 1.0
    return g * gc


def emissions(inp, top, prm, p, sl):
    if inp.sensors is not None:
        # detectors as noisy sensors: the machine decides (sensors.py)
        lab_ll = sum(s.loglik(sl) for s in inp.sensors)
        logB = lab_ll[:, top.lab].copy()
        pre = inp.pre[sl]
        logB[np.ix_(pre, np.arange(top.n) != 0)] = -np.inf
        logB[~pre, 0] = -np.inf
        return logB
    q = inp.q[sl]
    ratio = q / inp.prior
    C = prm["C"]
    lik = ratio @ C.T                                   # [T, n_classes]
    loglik = np.log(np.maximum(lik, 1e-12))[:, top.cls]
    beta = np.where(inp.vis[sl], p["beta_vis"], p["beta_hid"])[:, None]
    logB = beta * loglik
    vl = vis_loglik(inp.gt, inp.vis)
    logB += p["lam_vis"] * vl[inp.vis[sl].astype(int)][:, top.lab]
    tb, _ = tag_bias(inp.tags, inp.team_names, inp.gt, p["tag_n0"])
    tb = tb[sl]
    logB[:, top.lab == dc.LAB_H] += tb[:, :1]
    logB[:, top.lab == dc.LAB_A] += tb[:, 1:]
    out = inp.dead[sl] >= p["out_conf"]
    logB[np.ix_(out, top.lab != dc.LAB_D)] += np.log(p["eta_out"])
    if C.shape[1] == 2:
        # D only from the rules: penalise D states unless the dead-ball evidence supports them
        lack = np.clip(1 - inp.dead[sl] / p["out_conf"], 0, 1)
        logB[:, top.lab == dc.LAB_D] += (lack * np.log(p["eta_in"]))[:, None]
    pre = inp.pre[sl]
    logB[np.ix_(pre, np.arange(top.n) != 0)] = -np.inf
    logB[~pre, 0] = -np.inf
    return logB


def decode(inp, p, prm=None, em_iters=0, verbose=False):
    top = dc.Topology(p["k"], p["r"], p["rs"], p["rl"])
    prm = prm or dc.init_params(p, dc.c_init(inp.prior))
    labels = np.full(paths.N_FRAMES, "X", dtype="<U1")
    post = np.zeros((paths.N_FRAMES, 3))
    for it in range(em_iters + 1):
        src, kind, base, adv = dc.build_edges(top, prm, p)
        dst = src[:, 1].copy()
        s0 = src[:, 0].copy()
        counts = np.zeros(len(kind))
        Csuff = np.zeros_like(prm["C"])
        Ssuff = [np.zeros_like(s.E) for s in inp.sensors] if inp.sensors is not None else []
        ll = 0.0
        gate = change_gate(inp, p) if p.get("chg", 0) else touch_gate(inp, p)
        for h, (a, b) in paths.GRADED.items():
            sl = slice(a, b + 1)
            logB = emissions(inp, top, prm, p, sl)
            logA = dc.edge_logprob(s0, kind, base, adv, inp.dead[sl], inp.restart[sl], inp.rho_h[sl], p, gate[sl])
            init = np.full(top.n, -np.inf)
            if h == 1:
                for st in (top.entry(0), top.entry(1), top.DS[0]):
                    init[st] = np.log(1 / 3)
            else:
                init[0] = 0.0
            la = dc._forward(init, s0, dst, logA, logB)
            lb = dc._backward(s0, dst, logA, logB)
            logZ = np.logaddexp.reduce(la[-1])
            ll += logZ
            gamma = np.exp(la + lb - logZ)
            if it == em_iters:
                path = dc._viterbi(init, s0, dst, logA, logB)
                labels[sl] = LABELS[top.lab[path]]
                for j in range(3):
                    post[sl, j] = gamma[:, top.lab == j].sum(1)
            elif inp.sensors is not None:
                glab = np.stack([gamma[:, top.lab == j].sum(1) for j in range(3)], 1)
                for k, s in enumerate(inp.sensors):
                    Ssuff[k] += s.counts(glab, sl)
                if p.get("em_dur", 0):
                    counts += dc._xi_counts(s0, dst, logA, logB, la, lb, logZ)
            else:
                counts += dc._xi_counts(s0, dst, logA, logB, la, lb, logZ)
                r = inp.q[sl] / inp.prior
                for c in range(prm["C"].shape[0]):
                    g = gamma[:, top.cls == c].sum(1)
                    resp = r * prm["C"][c]
                    resp /= np.maximum(resp.sum(1, keepdims=True), 1e-12)
                    Csuff[c] += (g[:, None] * resp).sum(0)
        if verbose:
            print("iter %d  loglik %.1f" % (it, ll))
        if it < em_iters and inp.sensors is not None:
            for s, n in zip(inp.sensors, Ssuff):
                s.update(n)
            if p.get("em_dur", 0):
                prm = m_step(top, prm, src, kind, counts, Csuff)
        elif it < em_iters:
            prm = m_step(top, prm, src, kind, counts, Csuff)
    return labels, post, prm


def m_step(top, prm, src, kind, counts, Csuff, pseudo=50.0):
    new = {k: (v.copy() if isinstance(v, np.ndarray) else [x.copy() for x in v]) for k, v in prm.items()}
    by_src = {}
    for e, (a, b) in enumerate(src):
        by_src.setdefault(a, []).append((e, b))

    def phase_update(state, old):
        stay = adv = ext = 0.0
        for e, b in by_src[state]:
            if b == state:
                stay += counts[e]
            elif kind[e] == dc.K_STATIC:
                adv += counts[e]
            else:
                ext += counts[e]
        c = np.array([stay, adv, ext]) + pseudo * old
        if old[1] == 0:
            c[1] = 0
        return c / c.sum()

    for team in (0, 1):
        for j, s in enumerate(top.P[team]):
            new["P"][team][j] = phase_update(s, prm["P"][team][j])
    # tie home and away possession durations (symmetry)
    avg = 0.5 * (new["P"][0] + new["P"][1])
    new["P"] = [avg.copy(), avg.copy()]
    for key, phases in (("DS", top.DS), ("DL", top.DL)):
        for j, s in enumerate(phases):
            new[key][j] = phase_update(s, prm[key][j])
    C0 = dc.C_INIT if Csuff.shape[1] == 5 else prm["C"]
    C = Csuff + 200 * C0
    C = np.maximum(C / C.sum(1, keepdims=True), 0.005)
    # H/A symmetry of the compatibility matrix
    perm = [1, 0] + list(range(2, C.shape[1]))
    for a, b in ((0, 2), (1, 3)):
        m = 0.5 * (C[a] + C[b][perm])
        C[a], C[b] = m, m[perm]
    C = C / C.sum(1, keepdims=True)
    if C.shape[1] == 2:
        C[4:] = prm["C"][4:]  # D / PRE stay uninformative: D comes from the rules only
    new["C"] = C
    return new


# ------------------------------------------------------------------------------- scoring
def score(labels, gt, lo=paths.PUBLIC[0], hi=paths.PUBLIC[1]):
    g = np.full(paths.N_FRAMES, "X", dtype="<U1")
    g[lo:hi + 1] = gt[lo:hi + 1]
    gc = pm.label_codes(g)
    pc = pm.label_codes(labels)
    runs = pm.valid_runs(gc)
    sp = pm.spell_scores(gc, pm.fill_abstentions(pc, runs), runs)
    cm = pm.scored_confusion(pm.confusion(gc, pc))
    return dict(spell_f1=round(sp["mean_f1"], 4), macro_f1=round(float(pm.macro_f1(cm)), 4),
                n_gt=sp["n_gt"], n_pred=sp["n_pred"])


def score_folds(labels, gt):
    a, b = paths.PUBLIC
    f1 = score(labels, gt, a, paths.PUBLIC_SPLIT - 1)
    f2 = score(labels, gt, paths.PUBLIC_SPLIT, b)
    full = score(labels, gt)
    return dict(full=full, fold1=f1, fold2=f2,
                mean_fold_spell=round((f1["spell_f1"] + f2["spell_f1"]) / 2, 4))


def match_stats(labels):
    """Match-wide summary as the scorer sees the file: an abstention repeats the previous answer,
    it does not end a possession (counting X as a break used to inflate the possession count)."""
    from . import audit
    st = audit.structure(labels)
    codes = audit.fill_like_scorer(labels)
    g = np.zeros(len(codes), bool)
    for a, b in audit.graded_runs():
        g[a:b] = True
    share = {l: round(float((codes[g] == k).mean()), 3) for k, l in enumerate("HAD")}
    return dict(share=share, possessions=st["possessions"], possessions_per_min=st["poss_per_min"],
                median_possession_s=st["median_possession_s"], stoppages=st["stoppages"])


def delay_turnovers(labels, k):
    """Move every direct H<->A change k frames later. CHALLENGE.md: while the ball is contested or
    loose, possession stays with the last team - it changes when the new team controls the ball,
    not when the ball leaves the opponent (experiment 12: the machine was ~0.43 s early)."""
    x = labels.copy()
    t = np.flatnonzero(((labels[1:] == "H") & (labels[:-1] == "A")) | ((labels[1:] == "A") & (labels[:-1] == "H"))) + 1
    for c in t:
        seg = x[c:c + k]
        x[c:c + k] = np.where(seg == labels[c], labels[c - 1], seg)
    return x


def drop_short_stoppages(labels, min_frames):
    """No restart takes under two seconds: a D run shorter than min_frames between two live labels is
    not a stoppage. Its frames stay with the team that had the ball (CHALLENGE.md: a contested or
    loose ball stays with the last team). Runs at the edge of a half are left alone. (Experiment 17.)"""
    x = labels.copy()
    for a, b in paths.GRADED.values():
        seg = x[a:b + 1]
        s, e = fr.runs(seg == "D")
        for i, j in zip(s, e):
            if j - i < min_frames and i > 0 and j < len(seg) and seg[i - 1] in ("H", "A"):
                seg[i:j] = seg[i - 1]
    return x


def extend_to_tags(labels, tag_starts, max_gap, after):
    """A tagged set piece starts while the ball is still dead. A stoppage that ends at most max_gap
    frames before a tag start is the same stoppage: it runs on until `after` frames into the tag.
    (Experiment 17.)"""
    x = labels.copy()
    for f0 in sorted(tag_starts):
        lo = max(f0 - max_gap, 0)
        d = np.flatnonzero(x[lo:f0] == "D")
        if d.size:
            x[lo + d[-1] + 1:f0 + after] = "D"
    return x


def write_submission(labels, name):
    import pandas as pd
    lab = labels.copy()
    path = os.path.join(paths.OUTPUTS, name + ".csv")
    pd.DataFrame({"frame": np.arange(paths.N_FRAMES), "label": lab}).to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="proxy")
    ap.add_argument("--em", type=int, default=5)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--machine", action="store_true",
                    help="the network only detects, the machine decides: detectors as bounded sensors "
                         "(sensors.py), border head as turnover hazard; use with --em 3")
    ap.add_argument("--dead_model", help="runs/<stamp>_deadgbm/oof.npz")
    ap.add_argument("--dm_w", type=float, default=0.9)
    ap.add_argument("--delay", type=int, default=0,
                    help="frames to delay every H<->A change (experiment 12: 10)")
    ap.add_argument("--abstain", type=float, default=0.0,
                    help="write X where the machine's posterior is below this (experiments 08/10: 0.75)")
    ap.add_argument("--out")
    a = ap.parse_args()
    p = dict(dc.DEFAULT)
    for kv in a.set:
        k, v = kv.split("=")
        p[k] = type(p[k])(float(v)) if not isinstance(p[k], int) else int(v)
    if a.source == "proxy":
        q = q_from_proxy(proxy.load()["sym"])
        prior = np.full(2, 0.5)
    else:
        q = q_from_oof(a.source)
        S = np.load(os.path.join(paths.CACHE, "t0_steps.npz"))
        sym = S["sym"]
        prior = np.bincount(sym[sym >= 0], minlength=q.shape[1]) / (sym >= 0).sum()
    inp = Inputs(q, prior, a.dead_model, a.dm_w)
    if a.machine:
        from . import sensors
        inp.border = border_from_oof(a.source)
        inp.sensors = sensors.build(inp, fr.load(), alpha=500.0, use=sensors.MACHINE_USE, cap=4.0)
        p.update({k: v for k, v in MACHINE_P.items()
                  if not any(kv.startswith(k + "=") for kv in a.set)})
    labels, post, prm = decode(inp, p, em_iters=a.em, verbose=True)
    if a.delay > 0:
        labels = delay_turnovers(labels, a.delay)
    if a.abstain > 0:
        labels[(labels != "X") & (post.max(1) < a.abstain)] = "X"
    res = dict(params=p, score=score_folds(labels, inp.gt), match=match_stats(labels))
    print(json.dumps(res["score"]))
    print(json.dumps(res["match"]))
    if a.out:
        path = write_submission(labels, a.out)
        np.savez(os.path.join(paths.OUTPUTS, a.out + "_decoder.npz"), post=post,
                 P=np.array(prm["P"]), DS=prm["DS"], DL=prm["DL"], C=prm["C"])
        print("wrote", path)


if __name__ == "__main__":
    main()
