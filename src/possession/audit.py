"""Acceptance report of a submission: what the official scorer says on the public window (with its
bootstrap interval) and whether the labels make sense over the whole match, where there is no
ground truth. The thresholds in GATE are fixed before a variant is tried, never tuned to pass one.

    python scripts/check_match.py --csv outputs/<name>.csv [--sensors outputs/<name>_sensors.json]

Aggregates only; nothing is written.
"""
import os
import sys

import numpy as np

from . import paths

sys.path.insert(0, os.path.join(paths.MTAH_ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

FPS = paths.FPS

# public_spell_min is 0.60 and not the current 0.667: with perfect stoppages the public window gives
# 0.645 and one possession is worth 0.02 there - the gate stops a real regression, it does not reward
# the public number. set_pieces_with_d_min is 48 of 58: the tags are automatic, a clear improvement
# over the 38 of the first submission is asked for, not perfection.
GATE = dict(public_spell_min=0.60, public_macro_min=0.88, public_d_precision_min=0.92,
            public_d_recall_min=0.80, set_pieces_with_d_min=48, cut_inside_one_max=5,
            stoppages=(85, 130), d_share=(0.42, 0.52), short_stoppages_max=3,
            bc_cut_given_live_max=0.15, bc_cut_given_dead_min=0.40)


def graded_runs():
    """Half-open [start, stop) spans of the graded halves."""
    return [(a, b + 1) for a, b in paths.GRADED.values()]


def play_mask():
    """Graded frames from each kickoff on (the dead block before the 2nd-half kickoff is left out)."""
    m = np.zeros(paths.N_FRAMES, bool)
    m[paths.GRADED[1][0]:paths.GRADED[1][1] + 1] = True
    m[paths.KICKOFF_2:paths.GRADED[2][1] + 1] = True
    return m


def fill_like_scorer(labels):
    """Label codes (official numbering) with every X filled as the scorer does before it counts
    possessions: with the last answer of the same half."""
    return pm.fill_abstentions(pm.label_codes(np.asarray(labels)), graded_runs())


def structure(labels):
    """How many possessions and stoppages the file claims, as the scorer sees them."""
    labels = np.asarray(labels)
    codes = fill_like_scorer(labels)
    runs = graded_runs()
    graded = np.zeros(len(codes), bool)
    for a, b in runs:
        graded[a:b] = True
    segs = pm.segments(codes, runs)
    poss = np.array([b - a for a, b, k in segs if k in (pm.H, pm.A)])
    stops = np.array([b - a for a, b, k in segs if k == pm.D and a != paths.GRADED[2][0]])
    minutes = play_mask().sum() / FPS / 60
    return dict(possessions=int(len(poss)), poss_per_min=round(len(poss) / minutes, 2),
                median_possession_s=round(float(np.median(poss)) / FPS, 1) if len(poss) else None,
                stoppages=int(len(stops)), short_stoppages=int((stops < 2 * FPS).sum()),
                d_share=round(float((codes[graded] == pm.D).mean()), 4),
                x_share=round(float((labels[graded] == "X").mean()), 4))


def cut_runs(wide, covered):
    """[start, stop) spans, during play, where the broadcast is away from the wide shot."""
    m = np.concatenate([[0], (play_mask() & covered & ~wide).astype(np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def cuts_vs_labels(codes, runs):
    """Where the cut-aways fall: over a predicted stoppage, wholly inside one predicted possession
    (a stoppage the labels do not have), or across a predicted change of team."""
    out = dict(runs=len(runs), inside_one=0, straddle=0, touches_d=0)
    for a, b in runs:
        w = codes[a:b]
        if (w == pm.D).any():
            out["touches_d"] += 1
        elif (w == w[0]).all():
            out["inside_one"] += 1
        else:
            out["straddle"] += 1
    return out


def tag_checks(codes, tags):
    """Tagged set pieces start while the ball is still dead: count those the labels agree with
    (D on more than half of the 2 s before the tag)."""
    n = k = 0
    for t in tags:
        f0 = t["f0"]
        if t["code"] != "SET PIECES" or not any(a <= f0 < b for a, b in graded_runs()):
            continue
        n += 1
        k += bool((codes[f0 - 2 * FPS:f0] == pm.D).mean() > 0.5)
    return dict(set_pieces=n, with_d=k)


def public_metrics(labels, gt, bootstrap=2000):
    res = pm.evaluate(pm.label_codes(np.asarray(gt)), pm.label_codes(np.asarray(labels)), bootstrap_n=bootstrap)
    d = res["per_class"]["D"]
    return dict(spell=round(res["spell_f1_mean"], 4), spell_lo=round(res["spell_bootstrap_low"], 3),
                spell_hi=round(res["spell_bootstrap_high"], 3), macro=round(res["macro_f1"], 4),
                d_precision=round(d["precision"], 3), d_recall=round(d["recall"], 3),
                n_pred=res["n_pred_spells"], n_gt=res["n_gt_spells"])


def report(labels, sensors=None, bootstrap=2000):
    from . import io, sync
    if not os.path.exists(paths.cache("t0_broadcast.npz")):
        raise SystemExit("the report needs cache/t0_broadcast.npz: run  python -m possession.sync")
    bc = sync.load()
    codes = fill_like_scorer(labels)
    rep = dict(public=public_metrics(labels, io.load_gt_public(), bootstrap), match=structure(labels),
               tags=tag_checks(codes, io.load_tags()),
               cuts=cuts_vs_labels(codes, cut_runs(bc["wide"], bc["covered"])))
    if sensors:
        rep["sensors"] = sensors
    return rep


def verdict(rep):
    """One boolean per criterion of GATE."""
    g, pub, mt = GATE, rep["public"], rep["match"]
    bc = (rep.get("sensors") or {}).get("bc_cut")
    return dict(
        public_spell=bool(pub["spell"] >= g["public_spell_min"]),
        public_macro=bool(pub["macro"] >= g["public_macro_min"]),
        public_d=bool(pub["d_precision"] >= g["public_d_precision_min"] and pub["d_recall"] >= g["public_d_recall_min"]),
        set_pieces=bool(rep["tags"]["with_d"] >= g["set_pieces_with_d_min"]),
        cut_inside_one=bool(rep["cuts"]["inside_one"] <= g["cut_inside_one_max"]),
        stoppages=bool(g["stoppages"][0] <= mt["stoppages"] <= g["stoppages"][1]),
        d_share=bool(g["d_share"][0] <= mt["d_share"] <= g["d_share"][1]),
        short_stoppages=bool(mt["short_stoppages"] <= g["short_stoppages_max"]),
        # the cut-away sensor must stay a sensor: rare in live play, common in stoppages
        bc_sensor=bool(bc is None or (bc[0][2] <= g["bc_cut_given_live_max"] and bc[2][2] >= g["bc_cut_given_dead_min"])),
    )


def score_stoppages(labelled, is_dead, window):
    """Hand-labelled stoppages against a dead-ball mask, inside window = (first, last) frame.

    labelled: [(start, end)] in video frames, end = first frame with the ball back in play.
    A labelled stoppage is detected when the mask covers at least half of it; a stoppage of the mask
    is right when at least half of it lies inside a labelled one. Edge errors (mask minus label, in
    seconds; positive = late) are taken on the detected ones, from the mask run that overlaps most.
    """
    lo, hi = window
    m = np.zeros(len(is_dead), bool)
    m[lo:hi + 1] = is_dead[lo:hi + 1]
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    runs = list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))
    lab = np.zeros(len(is_dead), bool)
    for a, b in labelled:
        lab[a:b] = True
    detected, start_err, end_err = 0, [], []
    for a, b in labelled:
        if m[a:b].mean() < 0.5:
            continue
        detected += 1
        x, y = max(runs, key=lambda r: min(r[1], b) - max(r[0], a))
        start_err.append((x - a) / FPS)
        end_err.append((y - b) / FPS)
    right = sum(1 for x, y in runs if lab[x:y].mean() >= 0.5)
    return dict(labelled=len(labelled), detected=detected,
                recall=round(detected / len(labelled), 3) if labelled else None,
                predicted=len(runs), precision=round(right / len(runs), 3) if runs else None,
                median_start_error_s=round(float(np.median(start_err)), 2) if start_err else None,
                median_end_error_s=round(float(np.median(end_err)), 2) if end_err else None)
