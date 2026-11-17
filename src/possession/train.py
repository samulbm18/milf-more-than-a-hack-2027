"""Train PossNet on proxy symbols with block-masked ball (T3).

    python -m possession.train --fold 0 --seed 0
    python -m possession.train --overfit          # sanity: one window must be learnt

Writes runs/<name>/ with config.json, best.pt, metrics.jsonl and oof.npz (log q on every step,
computed with the real visibility mask).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as Fn
import yaml

from . import features, paths
from .model import PossNet, featurize, mdn_nll, swap_symbols

DEV, DTYPE = "cpu", torch.float32


def set_device(device=None, dtype=None):
    """Choose where the network runs and in which precision; returns (DEV, DTYPE).

    Defaults: env POSS_DEVICE, else cuda when present; env POSS_DTYPE, else float64 on CPU and
    float32 on GPU. CPU in double precision is the canonical path of the submission: the decisions
    taken downstream (symbol bins, abstention threshold) are then insensitive to the rounding
    differences between machines, which single precision and GPU kernels are not.
    """
    global DEV, DTYPE
    DEV = device or os.environ.get("POSS_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    name = dtype or os.environ.get("POSS_DTYPE") or ("float64" if DEV == "cpu" else "float32")
    DTYPE = torch.float64 if name == "float64" else torch.float32
    return DEV, DTYPE


set_device()


def build_model(cfg):
    return PossNet(**cfg["model"]).to(DEV).to(DTYPE)


RAW_KEYS = ["p_pos", "p_vel", "p_acc", "p_team", "p_gk", "p_age", "b_pos", "b_vel", "b_acc", "b_vis", "cam"]
GAP_KEYS = {"ts": "g_ts", "tu": "g_tu", "rel": "g_rel", "last_pos": "g_last", "next_pos": "g_next",
            "tl": "g_tl", "tn": "g_tn", "dl": "g_dl", "dn": "g_dn", "in_img": "g_img"}


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


# ------------------------------------------------------------------------------- data
class Match:
    def __init__(self):
        S = features.load()
        self.S = S
        self.T5 = len(S["step_frame"])
        self.raw = {k: torch.as_tensor(S[k]).to(DEV) for k in RAW_KEYS}
        self.sym = torch.as_tensor(S["sym"].astype(np.int64)).to(DEV)
        self.w = torch.as_tensor(S["w"]).to(DEV)
        self.holder = torch.as_tensor(S["holder_tok"].astype(np.int64)).to(DEV)
        self.gc = features.GapContext()
        self.gap_len = features.real_gap_lengths()
        self.border = torch.as_tensor(border_target(S["sym"])).to(DEV)
        self.real_vis25 = self.gc.vis.copy()
        self.graded25 = np.zeros(len(self.real_vis25), bool)
        for a, b in paths.GRADED.values():
            self.graded25[a:b + 1] = True

    def block_mask(self, rng, frac, short_extra):
        """Synthetic hidden mask at 25 Hz over really visible graded frames."""
        T = len(self.real_vis25)
        cand = np.flatnonzero(self.real_vis25 & self.graded25)
        target = frac * len(cand)
        m = np.zeros(T, bool)
        hidden = 0
        while hidden < target:
            n = 256
            starts = rng.choice(cand, n)
            lens = rng.choice(self.gap_len, n)
            short = rng.random(n) < short_extra
            lens[short] = rng.integers(5, 50, short.sum())
            for s, l in zip(starts, lens):
                e = min(s + l, T)
                hidden += (~m[s:e] & self.real_vis25[s:e]).sum()
                m[s:e] = True
                if hidden >= target:
                    break
        return m & self.real_vis25

    def set_mask(self, mask25):
        steps = self.S["step_frame"]
        g = self.gc.compute(self.real_vis25 & ~mask25, steps)
        self.gap = {GAP_KEYS[k]: torch.as_tensor(v).to(DEV) for k, v in g.items()}
        self.masked = torch.as_tensor(mask25[steps]).to(DEV)

    def batch(self, starts, L):
        idx = torch.as_tensor(starts, device=DEV)[:, None] + torch.arange(L, device=DEV)[None]
        b = {k: v[idx] for k, v in self.raw.items()}
        b.update({k: v[idx] for k, v in self.gap.items()})
        return b, idx


def border_target(sym, radius=2):
    """1 where the (forward-filled) controlling team or the dead state changes, within +-radius."""
    state = np.full(len(sym), -1, np.int64)
    cur = -1
    for i, s in enumerate(sym):
        if s in (0, 1):
            cur = s
        elif s == 4:
            cur = 2
        state[i] = cur
    ch = np.zeros(len(sym), bool)
    ch[1:] = (state[1:] != state[:-1]) & (state[:-1] >= 0)
    out = np.convolve(ch.astype(float), np.ones(2 * radius + 1), "same") > 0
    return out.astype(np.float32)


def fold_regions(M, folds, buffer_steps, exclude_public):
    """Per fold: boolean train / val masks over steps."""
    S = M.S
    g = S["graded"]
    fold_of = np.full(M.T5, -1)
    for h in (1, 2):
        idx = np.flatnonzero(g & (S["half"] == h))
        for k, part in enumerate(np.array_split(idx, folds)):
            fold_of[part] = k
    pub = (S["step_frame"] >= paths.PUBLIC[0]) & (S["step_frame"] <= paths.PUBLIC[1])
    out = []
    for k in range(folds):
        val = fold_of == k
        near = np.convolve(val.astype(float), np.ones(2 * buffer_steps + 1), "same") > 0
        train = g & ~near
        if exclude_public:
            train &= ~pub
        out.append((train, val))
    return out, pub


# ------------------------------------------------------------------------------- forward
def run_model(model, M, b, idx, cfg, train, rng=None):
    B = idx.shape[0]
    masked = M.masked[idx]
    # features are derived in the working precision, not only the network's weights
    b = {k: (v.to(DTYPE) if v.is_floating_point() else v) for k, v in b.items()}
    if train:
        flip_y = torch.as_tensor(rng.random(B) < cfg["aug"]["flip_y"], device=DEV)
        swap = torch.as_tensor(rng.random(B) < cfg["aug"]["swap_teams"], device=DEV)
        npl, nb = cfg["aug"]["pos_noise_m"]
        b["p_pos"] = b["p_pos"] + npl * torch.randn_like(b["p_pos"])
        b["b_pos"] = b["b_pos"] + nb * torch.randn_like(b["b_pos"])
        drop = (torch.rand(b["p_team"].shape, device=DEV) < cfg["mask"]["player_dropout"])
        b["p_team"] = torch.where(drop, torch.full_like(b["p_team"], -1), b["p_team"])
    else:
        flip_y = torch.zeros(B, dtype=torch.bool, device=DEV)
        swap = torch.zeros(B, dtype=torch.bool, device=DEV)
    x = featurize(b, masked, flip_y, swap)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda" and cfg["optim"]["amp"] == "bf16")):
        out = model(*x)
    out = {k: v.to(DTYPE) for k, v in out.items()}
    return out, masked, swap, x[1]


def losses(out, M, idx, masked, swap, pad, cfg, class_w):
    lw = cfg["loss"]
    sym = swap_symbols(M.sym[idx], swap)
    w = M.w[idx] * torch.where(masked, lw["masked_weight"], 1.0)
    valid = sym >= 0
    eps = lw.get("fc_eps", 0.0)
    if eps > 0:
        # forward correction: the targets are the true owner flipped with probability eps
        pn = Fn.softmax(out["sym"], -1) * (1 - eps) + Fn.softmax(out["sym"], -1).flip(-1) * eps
        ce = Fn.nll_loss(torch.log(pn.clamp(min=1e-6)).transpose(1, 2), sym.clamp(min=0), weight=class_w,
                         reduction="none")
    else:
        ce = Fn.cross_entropy(out["sym"].transpose(1, 2), sym.clamp(min=0), weight=class_w, reduction="none")
    l_sym = (ce * w * valid).sum() / (w * valid).sum().clamp(min=1)

    hold = M.holder[idx]
    ctrl = ((sym == 0) | (sym == 1)) & (hold >= 0)
    if pad is not None:
        ok = ~pad.gather(2, hold.clamp(min=0)[..., None]).squeeze(-1)  # holder dropped -> skip
        ctrl = ctrl & ok
    tgt = torch.where(ctrl, hold, torch.full_like(hold, out["ptr"].shape[-1] - 1))
    ptr_valid = ctrl | (valid & ~((sym == 0) | (sym == 1)))
    lp = Fn.cross_entropy(out["ptr"].transpose(1, 2), tgt, reduction="none")
    l_ptr = (lp * ptr_valid).sum() / ptr_valid.sum().clamp(min=1)

    mm = masked
    l_mdn = (mdn_nll(out["mdn"], M.raw["b_pos"][idx] * torch.stack(
        [torch.where(swap, -1.0, 1.0)[:, None].expand_as(masked), torch.ones_like(masked, dtype=torch.float)], -1)
    ) * mm).sum() / mm.sum().clamp(min=1)

    bt = M.border[idx]
    p = torch.sigmoid(out["border"])
    bce = Fn.binary_cross_entropy_with_logits(out["border"], bt, reduction="none")
    focal = bce * ((1 - p) * bt + p * (1 - bt)) ** 2
    l_border = (focal * valid).sum() / valid.sum().clamp(min=1)

    total = lw["sym"] * l_sym + lw["ptr"] * l_ptr + lw["mdn"] * l_mdn + lw["border"] * l_border
    return total, dict(sym=l_sym.item(), ptr=l_ptr.item(), mdn=l_mdn.item(), border=l_border.item())


@torch.no_grad()
def predict(model, M, cfg, L):
    """log q for every step with the currently set mask (sliding windows, centre crop)."""
    model.eval()
    T5 = M.T5
    stride = L // 2
    starts = list(range(0, T5 - L, stride)) + [T5 - L]
    logq = torch.zeros(T5, cfg["model"].get("n_sym", 5), device=DEV, dtype=DTYPE)
    border = torch.zeros(T5, device=DEV, dtype=DTYPE)
    filled = torch.zeros(T5, dtype=torch.bool, device=DEV)
    for i in range(0, len(starts), 32):
        st = starts[i:i + 32]
        b, idx = M.batch(st, L)
        out, _, _, _ = run_model(model, M, b, idx, cfg, train=False)
        lq = Fn.log_softmax(out["sym"], -1)
        for j, s in enumerate(st):
            lo = 0 if s == 0 else L // 4
            hi = L if s == T5 - L else L - L // 4
            sel = idx[j, lo:hi]
            write = ~filled[sel] | (lo > 0)
            logq[sel[write]] = lq[j, lo:hi][write]
            border[sel[write]] = torch.sigmoid(out["border"][j, lo:hi][write])
            filled[sel] = True
    model.train()
    return logq, border


def evaluate(model, M, cfg, val, L):
    logq, _ = predict(model, M, cfg, L)
    sym = M.sym
    v = torch.as_tensor(val, device=DEV) & (sym >= 0)
    nll = -logq.gather(1, sym.clamp(min=0)[:, None]).squeeze(1)
    res = {}
    for name, m in (("masked", v & M.masked), ("visible", v & ~M.masked & M.raw["b_vis"])):
        res["nll_" + name] = nll[m].mean().item() if m.any() else float("nan")
        ctrl = m & ((sym == 0) | (sym == 1))
        if ctrl.any():
            pred = (logq[:, 1] > logq[:, 0]).long()
            res["ctrl_team_acc_" + name] = (pred[ctrl] == sym[ctrl]).float().mean().item()
        res["acc_" + name] = (logq.argmax(1)[m] == sym[m]).float().mean().item() if m.any() else float("nan")
    return res


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(paths.REPO, "configs", "possnet.yaml"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--name")
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--all", action="store_true", help="final model: train on every graded step")
    ap.add_argument("--steps", help="alternative steps cache (label variant), e.g. cache/selftrain_steps.npz")
    ap.add_argument("--fc_eps", type=float, default=0.0,
                    help="forward noise correction: symmetric label-flip rate assumed in the targets")
    a = ap.parse_args()
    if a.steps:
        os.environ["POSS_STEPS"] = os.path.abspath(a.steps)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    cfg["loss"]["fc_eps"] = a.fc_eps
    if a.epochs:
        cfg["optim"]["epochs"] = a.epochs
    set_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    name = a.name or ("overfit" if a.overfit else ("possnet_all_s%d" % a.seed if a.all else
                                                    "possnet_f%d_s%d" % (a.fold, a.seed)))
    out_dir = os.path.join(paths.RUNS, time.strftime("%Y%m%d_%H%M_") + name)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(dict(cfg=cfg, args=vars(a)), open(os.path.join(out_dir, "config.json"), "w"), indent=1)

    M = Match()
    L = cfg["data"]["window_steps"]
    regions, pub = fold_regions(M, cfg["data"]["folds"], cfg["data"]["buffer_s"] * 5,
                                cfg["data"]["exclude_public_from_train"] and not a.all)
    train, val = regions[a.fold]
    if a.all:
        train = M.S["graded"].copy()
    elig = np.convolve(train.astype(float), np.ones(L), "valid") >= 0.5 * L
    starts_ok = np.flatnonzero(elig)
    train_t = torch.as_tensor(train, device=DEV)

    freq = np.bincount(M.S["sym"][train & (M.S["sym"] >= 0)], minlength=cfg["model"].get("n_sym", 5)).astype(float)
    class_w = torch.as_tensor((freq.sum() / np.maximum(freq, 1)) ** 0.5, dtype=torch.float32, device=DEV)
    class_w = class_w / class_w.mean()

    model = build_model(cfg)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["wd"])
    E = cfg["optim"]["epochs"]
    steps_per_epoch = math.ceil(cfg["optim"]["windows_per_epoch"] / cfg["optim"]["batch"])
    total, warm = E * steps_per_epoch, cfg["optim"]["warmup_epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(s / total, 1))))

    # fixed validation mask so epochs are comparable
    val_mask = M.block_mask(np.random.default_rng(1234), cfg["mask"]["frac"], cfg["mask"]["short_extra"])
    print("device %s | params %d | train steps %d | val steps %d | class freq %s"
          % (DEV, n_par, train.sum(), val.sum(), (freq / freq.sum()).round(3).tolist()))
    log = open(os.path.join(out_dir, "metrics.jsonl"), "w")
    best = float("inf")
    fixed = rng.choice(starts_ok, cfg["optim"]["batch"]) if a.overfit else None
    if a.overfit:
        M.set_mask(M.block_mask(rng, cfg["mask"]["frac"], cfg["mask"]["short_extra"]))
    t0 = time.time()
    for ep in range(E):
        if not a.overfit:
            M.set_mask(M.block_mask(rng, cfg["mask"]["frac"], cfg["mask"]["short_extra"]))
        agg = {}
        for it in range(steps_per_epoch):
            st = fixed if a.overfit else rng.choice(starts_ok, cfg["optim"]["batch"])
            b, idx = M.batch(st, L)
            out, masked, swap, pad = run_model(model, M, b, idx, cfg, train=not a.overfit, rng=rng)
            inside = train_t[idx]
            lt, parts = losses(out, M, idx, masked, swap, pad, cfg, class_w)
            opt.zero_grad(set_to_none=True)
            lt.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0) + v / steps_per_epoch
            del inside
        rec = dict(epoch=ep + 1, time_s=round(time.time() - t0, 1), **{"train_" + k: round(v, 4) for k, v in agg.items()})
        if not a.overfit and ((ep + 1) % cfg["optim"]["eval_every"] == 0 or ep + 1 == E):
            M.set_mask(val_mask)
            ev = evaluate(model, M, cfg, val if not a.all else M.S["graded"], L)
            rec.update({"val_" + k: round(v, 4) for k, v in ev.items()})
            if ev["nll_masked"] < best:
                best = ev["nll_masked"]
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec))
    if a.overfit:
        return
    torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
    if a.all:
        torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
    # out-of-fold predictions with the REAL visibility (what the decoder will see)
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=DEV))
    M.set_mask(np.zeros_like(M.real_vis25))
    logq, border = predict(model, M, cfg, L)
    np.savez(os.path.join(out_dir, "oof.npz"), logq=logq.cpu().numpy(), border=border.cpu().numpy(),
             val=val, train=train)
    print("best val nll_masked %.4f | peak GPU MB %.0f | %s" %
          (best, torch.cuda.max_memory_allocated() / 2 ** 20 if DEV == "cuda" else 0, out_dir))


if __name__ == "__main__":
    main()
