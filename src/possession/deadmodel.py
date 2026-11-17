"""T1c: dead-ball model (LightGBM) on collective player behaviour only - no ball variables.

Weak supervision from the rules (positive / unlabelled):
  positive  = rule dead >= pos_conf (precision 0.996 on the public window), outside PRE
  negative  = confident live play: a last-touch owner with a touch in the last neg_age_s,
              rule dead < 0.3
  the rest is unlabelled and not used for fitting (it holds the stoppages the rules miss).
The public window is never used for fitting; it is only reported.

    python -m possession.deadmodel            # 4-fold OOF + final model, writes runs/<stamp>_deadgbm/

Outputs: fold_<k>.txt and all.txt (LightGBM boosters), oof.npz (p_dead per frame, 25 Hz, out of fold
inside the graded halves, final-model elsewhere), metrics.json.
"""
import json
import os
import time
import warnings

import lightgbm as lgb
import numpy as np

from . import events as ev
from . import frames as fr
from . import paths, proxy

CFG = dict(step=5, windows_s=(1, 3, 8), pos_conf=0.85, neg_age_s=2.0, folds=4, buffer_s=30,
           lgb=dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200,
                    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                    verbose=-1, seed=0),
           rounds=300)


def frame_features(F):
    """Per-frame collective features, [T, n]. Uses players only."""
    pos, vel, acc = F["pos"], F["vel"], F["acc"]
    sp = np.hypot(vel[..., 0], vel[..., 1])
    ok = np.isfinite(sp)
    team = F["slot_team"]
    HL, HW = paths.PITCH_L / 2, paths.PITCH_W / 2
    x, y = pos[..., 0], pos[..., 1]
    near_side = ok & (np.abs(y) > HW - 2.5)
    near_end = ok & (np.abs(x) > HL - 2.5)
    outside = ok & ((np.abs(y) > HW + 0.5) | (np.abs(x) > HL + 0.5))
    feats = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feats["n_players"] = ok.sum(1)
        feats["spd_med"] = np.nanmedian(sp, 1)
        feats["spd_p90"] = np.nanpercentile(sp, 90, 1)
        feats["spd_max"] = np.nanmax(sp, 1)
        feats["slow_frac"] = np.nanmean(np.where(ok, sp < 1.5, np.nan), 1)
        feats["fast_n"] = (np.nan_to_num(sp) > 4.0).sum(1)
        feats["acc_mean"] = np.nanmean(np.abs(acc), 1)
        for k, nm in ((0, "h"), (1, "a")):
            m = ok & (team[None] == k)
            xs, ys = np.where(m, x, np.nan), np.where(m, y, np.nan)
            feats["cx_" + nm] = np.nanmean(xs, 1)
            feats["sx_" + nm] = np.nanstd(xs, 1)
            feats["sy_" + nm] = np.nanstd(ys, 1)
            feats["spd_" + nm] = np.nanmean(np.where(m, sp, np.nan), 1)
            gk = m & F["slot_gk"][None]
            feats["gk_x_" + nm] = np.nanmean(np.where(gk, x, np.nan), 1) * (1 if k == 0 else -1)
            feats["gk_spd_" + nm] = np.nanmean(np.where(gk, sp, np.nan), 1)
        feats["n_near_side"] = near_side.sum(1)
        feats["n_near_end"] = near_end.sum(1)
        feats["n_outside"] = outside.sum(1)
        feats["cx_gap"] = feats["cx_h"] - feats["cx_a"]
    names = list(feats)
    X = np.stack([np.asarray(feats[n], np.float32) for n in names], 1)
    return X, names


def windowed(X, names, steps, windows_s, fps):
    """Centred rolling mean / min / max at several scales, plus past-vs-future change, at steps."""
    T = len(X)
    Xf = np.where(np.isfinite(X), X, 0.0)
    cnt = np.isfinite(X).astype(np.float64)
    cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(Xf, 0)])
    cc = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(cnt, 0)])
    cols, cn = [X[steps]], [n + "@0" for n in names]
    for w in windows_s:
        h = int(w * fps / 2)
        lo, hi = np.clip(steps - h, 0, T), np.clip(steps + h + 1, 0, T)
        mean = (cs[hi] - cs[lo]) / np.maximum(cc[hi] - cc[lo], 1)
        pl, ph = np.clip(steps - 2 * h, 0, T), steps
        fl, fh = steps + 1, np.clip(steps + 2 * h + 1, 0, T)
        past = (cs[ph] - cs[pl]) / np.maximum(cc[ph] - cc[pl], 1)
        fut = (cs[fh] - cs[fl]) / np.maximum(cc[fh] - cc[fl], 1)
        cols += [mean, fut - past]
        cn += [n + "@m%d" % w for n in names] + [n + "@d%d" % w for n in names]
    # rolling min / max of the speed features over the longest window
    w = windows_s[-1]
    h = int(w * fps / 2)
    for n in ("spd_p90", "spd_max", "slow_frac"):
        j = names.index(n)
        col = np.nan_to_num(X[:, j])
        mx = np.array([col[max(0, s - h):s + h + 1].max() for s in steps])
        mn = np.array([col[max(0, s - h):s + h + 1].min() for s in steps])
        cols += [mx[:, None], mn[:, None]]
        cn += [n + "@max%d" % w, n + "@min%d" % w]
    return np.hstack(cols).astype(np.float32), cn


def labels(F, E, P, c):
    T = len(F["vis"])
    pos = (E["dead"] >= c["pos_conf"]) & ~F["pre"]
    tt, _, _ = proxy.touches(F, dict(proxy.DEFAULT))
    last = np.full(T, -10 ** 9)
    idx = np.flatnonzero(tt >= 0)
    last[idx] = idx
    last = np.maximum.accumulate(last)
    age = np.arange(T) - last
    neg = (P["sym"] >= 0) & (age <= c["neg_age_s"] * paths.FPS) & (E["dead"] < 0.3) & ~F["pre"]
    y = np.full(T, -1, np.int8)
    y[neg] = 0
    y[pos] = 1
    return y


def folds_25(F, c):
    """Per fold: (train, val) masks over frames; public window never in train."""
    T = len(F["vis"])
    g = F["graded"]
    fold_of = np.full(T, -1)
    for h in (1, 2):
        idx = np.flatnonzero(g & (F["half"] == h))
        for k, part in enumerate(np.array_split(idx, c["folds"])):
            fold_of[part] = k
    pub = np.zeros(T, bool)
    pub[paths.PUBLIC[0]:paths.PUBLIC[1] + 1] = True
    buf = c["buffer_s"] * paths.FPS
    out = []
    for k in range(c["folds"]):
        val = fold_of == k
        near = np.convolve(val.astype(float), np.ones(2 * buf + 1), "same") > 0
        out.append((g & ~near & ~pub, val))
    return out, pub


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def main(c=CFG):
    t0 = time.time()
    F = fr.load()
    E = ev.load()
    P = proxy.load()
    T = len(F["vis"])
    X0, names = frame_features(F)
    steps = np.arange(0, T, c["step"])
    X, cn = windowed(X0, names, steps, c["windows_s"], paths.FPS)
    y25 = labels(F, E, P, c)
    y = y25[steps]
    regions, pub = folds_25(F, c)
    out_dir = os.path.join(paths.RUNS, time.strftime("%Y%m%d_%H%M_") + "deadgbm")
    os.makedirs(out_dir, exist_ok=True)
    print("features %d | steps %d | pos %d neg %d unlabelled %d"
          % (X.shape[1], len(steps), (y == 1).sum(), (y == 0).sum(), (y < 0).sum()))

    oof = np.full(len(steps), np.nan)
    met = dict(folds=[])
    for k, (tr25, va25) in enumerate(regions):
        tr, va = tr25[steps] & (y >= 0), va25[steps]
        ds = lgb.Dataset(X[tr], y[tr], feature_name=cn)
        b = lgb.train(c["lgb"], ds, c["rounds"])
        b.save_model(os.path.join(out_dir, "fold_%d.txt" % k))
        oof[va] = b.predict(X[va])
        lab = va & (y >= 0)
        met["folds"].append(dict(fold=k, auc=round(auc(y[lab], oof[lab]), 4), n_val=int(lab.sum())))
        print(met["folds"][-1], flush=True)
    b = lgb.train(c["lgb"], lgb.Dataset(X[(y >= 0) & ~pub[steps]], y[(y >= 0) & ~pub[steps]], feature_name=cn),
                  c["rounds"])
    b.save_model(os.path.join(out_dir, "all.txt"))
    full = b.predict(X)
    p = np.where(np.isnan(oof), full, oof)  # public window and non-graded: final model (never fit on public)

    g = F["graded"][steps]
    lab = g & (y >= 0) & ~np.isnan(oof)
    met["oof_auc"] = round(auc(y[lab], oof[lab]), 4)
    for th in (0.5, 0.7, 0.9):
        met["oof_th%.1f" % th] = dict(recall_rule_dead=round(float((oof[lab & (y == 1)] >= th).mean()), 3),
                                      fpr_live=round(float((oof[lab & (y == 0)] >= th).mean()), 3),
                                      unlabelled_pos_share=round(float((oof[g & (y < 0) & ~np.isnan(oof)] >= th).mean()), 3))
    imp = sorted(zip(b.feature_importance("gain"), cn), reverse=True)[:12]
    met["top_features"] = [n for _, n in imp]
    t = np.arange(T)
    p25 = np.interp(t, steps, p)
    np.savez(os.path.join(out_dir, "oof.npz"), p_dead=p25.astype(np.float32))
    json.dump(dict(cfg=c, metrics=met), open(os.path.join(out_dir, "metrics.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in met.items() if k != "folds"}, indent=1))
    print("%.0fs | %s" % (time.time() - t0, out_dir))
    return out_dir


if __name__ == "__main__":
    main()
