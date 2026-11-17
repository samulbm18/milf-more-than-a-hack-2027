"""Figures and generated tables for the technical document (docs/paper/MILF.tex).

Everything written here is an aggregate: per-block shares, per-second label strips of OUR OWN
predictions, sensor matrices learnt by the EM, and the documented experiment results. No raw
tracking row, no video frame.

    python docs/paper/make_figures.py            # from the repository root; needs matplotlib, numpy, pandas, pyyaml

Inputs: cache/t0_frames.npz, cache/t0_broadcast.npz, the Smart Tagging XML (start frames of SET
PIECES), outputs/{a_check,final_check}.csv, outputs/final_check_sensors.json and the public ground
truth (published 5 minutes only).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
from possession import io, paths  # noqa: E402

OUT = os.path.join(ROOT, "docs", "paper", "fig")
os.makedirs(OUT, exist_ok=True)
FPS = paths.FPS

# ---- style: light print, recessive axes, text in ink tokens --------------------------------------
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
C_H, C_A, C_D, C_X = "#2a78d6", "#eb6834", "#c9c8c2", "#ffffff"
C_ACC = "#1baf7a"  # accent for "chosen / adopted"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "legend.frameon": False, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def labels(csv):
    s = pd.read_csv(csv)["label"].to_numpy()
    return s


def fill_x(lab):
    """Scorer's fill rule for abstentions: X takes the previous label."""
    out = lab.copy()
    for i in range(1, len(out)):
        if out[i] == "X" and out[i - 1] != "X":
            out[i] = out[i - 1]
    return out


# ---- F3: ball visibility per 5-minute block --------------------------------------------------------
def fig_visibility():
    z = np.load(paths.cache("t0_frames.npz"))
    vis = z["vis"]
    blk = 5 * 60 * FPS
    rows = []
    for half, (a, b) in paths.GRADED.items():
        start = paths.KICKOFF_2 if half == 2 else a
        for s in range(start, b + 1, blk):
            e = min(s + blk, b + 1)
            if e - s < blk // 2:
                continue
            rows.append((half, (s - start) / FPS / 60, vis[s:e].mean(), s == paths.PUBLIC[0]))
    df = pd.DataFrame(rows, columns=["half", "min", "vis", "public"])
    med = df.loc[~df.public, "vis"].median()
    fig, ax = plt.subplots(figsize=(6.3, 1.9))
    x = np.arange(len(df))
    ax.bar(x, df.vis * 100, width=0.8, color=[C_ACC if p else C_H for p in df.public])
    ax.axhline(med * 100, color=INK2, lw=0.8, ls="--")
    ax.text(len(df) - 0.4, med * 100 + 2, "median outside public window %.0f%%" % (med * 100),
            ha="right", va="bottom", color=INK2, fontsize=7)
    pv = df.loc[df.public, "vis"].iloc[0]
    ax.annotate("public window %.0f%%" % (pv * 100), (0, pv * 100), (1.2, 97), color=INK,
                fontsize=7, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    n1 = int((df.half == 1).sum())
    ax.axvline(n1 - 0.5, color=MUTED, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["%d'" % m for m in df["min"]], fontsize=6)
    ax.text(n1 / 2 - 0.5, -24, "first half", ha="center", color=INK2, fontsize=7)
    ax.text(n1 + (len(df) - n1) / 2 - 0.5, -24, "second half", ha="center", color=INK2, fontsize=7)
    ax.set_ylabel("ball tracked (% frames)")
    ax.set_ylim(0, 105)
    save(fig, "visibility")
    return dict(public=pv, median=med, min=df.loc[~df.public, "vis"].min(), n=len(df))


# ---- F7: whole-match label strips -------------------------------------------------------------------
def strip_rgb(lab, step):
    col = {"H": C_H, "A": C_A, "D": C_D, "X": C_X}
    idx = np.arange(0, len(lab), step)
    return [col.get(lab[i], C_X) for i in idx]


def fig_match_strips():
    a = fill_x(labels("outputs/a_check.csv"))
    f = fill_x(labels("outputs/final_check.csv"))
    bc = np.load(paths.cache("t0_broadcast.npz"))
    cut = bc["covered"] & ~bc["wide"]
    tags = [t["f0"] for t in io.load_tags() if t["code"] == "SET PIECES"]
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 1.9), gridspec_kw=dict(hspace=0.9))
    step = FPS  # one cell per second
    for ax, half in zip(axes, (1, 2)):
        a0, b0 = paths.GRADED[half]
        if half == 2:
            a0 = paths.KICKOFF_2
        sl = slice(a0, b0 + 1)
        n = (b0 + 1 - a0) // step
        t = np.arange(n) / 60
        rows = [("before broadcast", a[sl]), ("submitted", f[sl])]
        for k, (name, lab) in enumerate(rows):
            cols = strip_rgb(lab, step)[:n]
            ax.bar(t, 1, width=1 / 60, bottom=1.2 * (2 - k), color=cols, align="edge", lw=0)
            ax.text(-0.6, 1.2 * (2 - k) + 0.5, name, ha="right", va="center", fontsize=7, color=INK)
        c = cut[sl][::step][:n]
        ax.bar(t[c], 0.5, width=1 / 60, bottom=0.45, color=INK2, align="edge", lw=0)
        ax.text(-0.6, 0.7, "broadcast cut-away", ha="right", va="center", fontsize=7, color=INK)
        tt = [(x - a0) / FPS / 60 for x in tags if a0 <= x <= b0]
        ax.scatter(tt, np.full(len(tt), 0.1), marker="|", s=18, color=C_ACC, lw=1)
        ax.text(-0.6, 0.1, "SET PIECES tag", ha="right", va="center", fontsize=7, color=INK)
        ax.set_xlim(0, t[-1] + 1 / 60)
        ax.set_ylim(-0.15, 3.65)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title("%s half (minutes from kick-off)" % ("first" if half == 1 else "second"),
                     fontsize=7, color=INK2, loc="left", pad=2)
        ax.tick_params(axis="x", labelsize=6)
    axes[0].legend(handles=[Patch(color=C_H, label="H (FC Barcelona)"), Patch(color=C_A, label="A (Athletic)"),
                            Patch(color=C_D, label="D (dead ball)")], ncol=3, fontsize=7,
                   loc="lower right", bbox_to_anchor=(1, 1.15))
    save(fig, "match_strips")


# ---- F8: public window, ground truth vs submitted -------------------------------------------------------
def fig_public_strip():
    gt = pd.read_csv(paths.GT_PUBLIC_CSV)
    gt = gt.set_index("frame")["label"]
    f = fill_x(labels("outputs/final_check.csv"))
    a = fill_x(labels("outputs/a_check.csv"))
    a0, b0 = paths.PUBLIC
    fr = np.arange(a0, b0 + 1)
    g = gt.reindex(fr).fillna("X").to_numpy()
    rows = [("ground truth", g), ("before broadcast", a[a0:b0 + 1]), ("submitted", f[a0:b0 + 1])]
    step = 5  # 0.2 s
    fig, ax = plt.subplots(figsize=(6.3, 1.0))
    n = len(fr) // step
    t = np.arange(n) * step / FPS
    for k, (name, lab) in enumerate(rows):
        ax.bar(t, 0.8, width=step / FPS, bottom=len(rows) - 1 - k, color=strip_rgb(lab, step)[:n],
               align="edge", lw=0)
        ax.text(-3, len(rows) - 1 - k + 0.4, name, ha="right", va="center", fontsize=7, color=INK)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("seconds into the public window", fontsize=7)
    ax.tick_params(axis="x", labelsize=6)
    save(fig, "public_strip")


# ---- F6: ablation ladder (official evaluator, public window; docs/TECHNICAL.md section 4) ----------------
LADDER = [
    ("Official baseline", 0.243, None),
    ("Detector + HSMM (single model, mean)", 0.557, 0.791),
    ("Detector ensemble + HSMM", 0.613, 0.788),
    ("+ self-training (single model, mean)", 0.625, 0.802),
    ("+ self-training, ensemble, abstention", 0.640, 0.801),
    ("+ turnover shift (0.4 s)", 0.667, 0.803),
    ("+ broadcast cut-away sensor", 0.667, 0.931),
    ("Final: + stoppage boundary rules", 0.667, 0.931),
]


def fig_ladder():
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    y = np.arange(len(LADDER))[::-1]
    for ax, j, title in ((axes[0], 1, "spellF1 (ranking metric)"), (axes[1], 2, "macroF1 (tie-break)")):
        v = [r[j] for r in LADDER]
        for yi, vi, r in zip(y, v, LADDER):
            if vi is None:
                ax.text(0.02, yi, "n/a", va="center", fontsize=6, color=MUTED)
                continue
            last = r is LADDER[-1]
            ax.plot([0, vi], [yi, yi], color=GRID if not last else C_ACC, lw=2, solid_capstyle="round")
            ax.plot(vi, yi, "o", ms=4.5, color=C_ACC if last else C_H, mec="white", mew=1)
            ax.text(vi + 0.015, yi, "%.3f" % vi, va="center", fontsize=6.5, color=INK)
        if j == 1:
            ax.errorbar(0.667, y[-1] - 0.6, xerr=[[0.667 - 0.45], [0.87 - 0.667]], fmt="none",
                        ecolor=INK2, elinewidth=0.7, capsize=2)
            ax.text(0.40, y[-1] - 1.05, "bootstrap 95% interval 0.45-0.87", fontsize=6, color=INK2)
        ax.set_xlim(0, 1.08)
        ax.set_title(title, fontsize=7.5, color=INK, loc="left")
        ax.tick_params(axis="x", labelsize=6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in LADDER], fontsize=6.8)
    axes[0].set_ylim(-1.35, len(LADDER) - 0.5)
    axes[1].tick_params(axis="y", length=0)
    save(fig, "ladder")


# ---- F5: experiments, paired Δ spellF1 vs the reference (experiments/README.md) -------------------------
# (id, short name, low, high, wins/ties/losses, verdict)  low==high when a single number was reported
EXPERIMENTS = [
    ("00", "machine decides (network = sensor)", 0.025, 0.025, "6/1/3", "adopted"),
    ("01", "Smart Tagging actions as sensors", -0.115, 0.0, "0/0/10..0/9/1", "dropped"),
    ("02", "trajectory-validated touches", 0.0, 0.0, "0/10/0", "dropped"),
    ("04", "self-training on machine labels", 0.071, 0.071, "4/0/0", "adopted"),
    ("05", "reach-based control", 0.002, 0.002, "2/8/0", "dropped"),
    ("06", "dead-ball TCN", -0.040, -0.040, "1/0/9", "dropped"),
    ("07", "restart team from geometry", 0.0, 0.0, "0/10/0", "dropped"),
    ("08", "abstention X < 0.75", 0.014, 0.020, "7/2/1", "adopted"),
    ("09", "forward label-noise correction", -0.065, -0.065, "3/0/2", "dropped"),
    ("11", "second self-training round", -0.026, -0.026, "3/0/1 vs v3", "dropped"),
    ("12", "delay turnovers 10 frames", 0.020, 0.020, "8/4/0", "adopted"),
    ("13", "pass-receiver / interception", -0.003, -0.003, "0/9/3", "dropped"),
    ("14", "sustained-control sensor", -0.003, 0.0, "0/9/3", "dropped"),
    ("15", "dead-ball LightGBM / extend D", -0.111, -0.030, "0/0/12", "dropped"),
]


def fig_experiments():
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    y = np.arange(len(EXPERIMENTS))[::-1]
    ax.axvline(0, color=MUTED, lw=0.7)
    for yi, (eid, name, lo, hi, wtl, verdict) in zip(y, EXPERIMENTS):
        col = C_ACC if verdict == "adopted" else MUTED
        if hi - lo > 1e-9:
            ax.plot([lo, hi], [yi, yi], color=col, lw=2.2, solid_capstyle="round")
        else:
            ax.plot(lo, yi, "o", ms=5, color=col, mec="white", mew=1)
        ax.text(0.105, yi, wtl, va="center", fontsize=6.3, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(["%s  %s" % (e[0], e[1]) for e in EXPERIMENTS], fontsize=6.8)
    ax.set_xlim(-0.125, 0.14)
    ax.text(0.105, len(EXPERIMENTS) - 0.15, "win/tie/loss", fontsize=6.3, color=INK2)
    ax.set_xlabel(r"paired $\Delta$ spellF1 vs. reference (public window)", fontsize=7)
    ax.tick_params(axis="x", labelsize=6)
    ax.legend(handles=[Patch(color=C_ACC, label="adopted"), Patch(color=MUTED, label="dropped")],
              loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, fontsize=6.5)
    save(fig, "experiments")


# ---- T1: what the EM learnt (LaTeX table body) -----------------------------------------------------------
SYMBOLS = {
    "net_vis": ["A sure", "A", "?", "H", "H sure"],
    "net_hid": ["A sure", "A", "?", "H", "H sure"],
    "touch": ["none", "H touch", "A touch"],
    "dead_rule": ["none", "weak", "strong"],
    "bc_cut": ["wide+ball", "neither", "cut-away"],
}


def table_em():
    E = json.load(open("outputs/final_check_sensors.json"))
    lines = []
    for k in ("touch", "dead_rule", "bc_cut"):
        m = np.array(E[k])
        for r, st in ((0, "H"), (2, "D")):
            cells = " & ".join("%.3f" % v for v in m[r])
            name = {"touch": "touches", "dead_rule": "dead-ball rules", "bc_cut": "broadcast picture"}[k]
            sym = ", ".join(SYMBOLS[k])
            lines.append(("%s (%s)" % (name, sym) if r == 0 else "") + " & %s & %s \\\\" % (st, cells))
        lines.append("\\addlinespace")
    with open(os.path.join(OUT, "em_table.tex"), "w") as fh:
        fh.write("\n".join(lines[:-1]) + "\n")
    print("wrote em_table.tex")


if __name__ == "__main__":
    v = fig_visibility()
    print("visibility", {k: round(float(x), 3) if not isinstance(x, int) else x for k, x in v.items()})
    fig_match_strips()
    fig_public_strip()
    fig_ladder()
    fig_experiments()
    table_em()
