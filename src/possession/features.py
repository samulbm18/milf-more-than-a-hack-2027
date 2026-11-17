"""T0b: per-step raw quantities for PossNet at 5 Hz, plus gap variables for any visibility mask.

Features are not stored finished: the model's `featurize` builds them on the GPU from these raw
quantities, so flips and team swaps are applied once, to geometry, before anything is derived.

cache/t0_steps.npz
    step_frame[T5]            video frame of each step (every 5th frame)
    p_pos[T5,N,2] p_vel p_acc p_team p_gk p_age p_slot     N = 30 tokens, team -1 = padding
    b_pos[T5,2] b_vel b_acc b_vis                           real ball (b_vis = really visible)
    cam[T5,2]                 image centre projected to the pitch, metres, direction-normalised
    sym[T5] w[T5] holder_tok[T5]                            proxy targets at the step frame
    graded[T5] half[T5] pre[T5]
"""
import os

import numpy as np

from . import frames as fr
from . import io, paths, proxy

STRIDE = 5
N_TOK = 30


def camera_centre(F):
    H, valid = io.load_homography()
    Hinv = np.full_like(H, np.nan)
    Hinv[valid] = np.linalg.inv(H[valid])
    c = Hinv @ np.array([paths.IMG_W / 2, paths.IMG_H / 2, 1.0])
    xy = c[:, :2] / c[:, 2:3]
    out = np.stack([(xy[:, 0] - 0.5) * paths.PITCH_L, (xy[:, 1] - 0.5) * paths.PITCH_W], 1)
    for h in (1, 2):
        out[F["half"] == h, 0] *= F["flip"][h]
    return out.astype(np.float32), H


def build(verbose=True):
    F = fr.load()
    P = proxy.load()
    steps = np.arange(0, len(F["vis"]), STRIDE)
    T5 = len(steps)
    cam, _ = camera_centre(F)

    pos = F["pos"][steps]
    present = np.isfinite(pos[..., 0])
    ball = F["ball"][steps]
    # order tokens by distance to the ball (or to the centroid when it is hidden); keep N_TOK
    ref = np.where(np.isfinite(ball), ball, np.nanmean(np.where(present[..., None], pos, np.nan), 1))
    d = np.hypot(*(pos - ref[:, None]).transpose(2, 0, 1))
    d = np.where(present, d, np.inf)
    order = np.argsort(d, 1)[:, :N_TOK]
    take = lambda a: np.take_along_axis(a, order.reshape(order.shape + (1,) * (a.ndim - 2)), 1)  # noqa: E731
    p_slot = np.where(np.take_along_axis(present, order, 1), order, -1).astype(np.int16)
    ok = p_slot >= 0
    team = np.where(ok, F["slot_team"][np.clip(p_slot, 0, None)], -1).astype(np.int8)

    holder = P["holder"][steps]
    holder_tok = np.full(T5, -1, np.int16)
    hit = (p_slot == holder[:, None]) & (holder[:, None] >= 0)
    has = hit.any(1)
    holder_tok[has] = hit[has].argmax(1)

    out = dict(
        step_frame=steps.astype(np.int32),
        p_pos=np.where(ok[..., None], take(pos), 0).astype(np.float32),
        p_vel=np.nan_to_num(np.where(ok[..., None], take(F["vel"][steps]), 0)).astype(np.float32),
        p_acc=np.nan_to_num(np.where(ok, take(F["acc"][steps]), 0)).astype(np.float32),
        p_team=team,
        p_gk=np.where(ok, F["slot_gk"][np.clip(p_slot, 0, None)], False),
        p_age=np.nan_to_num(np.where(ok, take(F["trk_age"][steps]), 0)).astype(np.float32),
        p_slot=p_slot,
        b_pos=np.nan_to_num(ball).astype(np.float32),
        b_vel=np.nan_to_num(F["ball_vel"][steps]).astype(np.float32),
        b_acc=np.nan_to_num(F["ball_acc"][steps]).astype(np.float32),
        b_vis=F["vis"][steps],
        cam=np.nan_to_num(cam[steps]).astype(np.float32),
        sym=P["sym"][steps], w=P["weight"][steps], holder_tok=holder_tok,
        graded=F["graded"][steps], half=F["half"][steps], pre=F["pre"][steps],
    )
    np.savez(paths.cache("t0_steps.npz"), **out)
    if verbose:
        print("steps %d, tokens/step mean %.1f, max present %d, holder found on %.3f of CTRL steps"
              % (T5, ok.sum(1).mean(), present.sum(1).max(),
                 (holder_tok[(out["sym"] == 0) | (out["sym"] == 1)] >= 0).mean()))
    return out


class GapContext:
    """Everything needed to recompute gap variables for an arbitrary visibility mask, at 25 Hz."""

    def __init__(self):
        F = fr.load()
        self.ball = F["ball"]
        self.vis = F["vis"]
        self.half = F["half"]
        self.flip = F["flip"]
        near, d1, _, _, _ = proxy.ball_geometry(F)
        self.near_team = F["slot_team"][near].astype(np.int8)
        self.d1 = np.nan_to_num(d1, posinf=99).astype(np.float32)
        _, self.H = camera_centre(F)
        # unflipped normalised coordinates for projecting into the image
        bx = self.ball[:, 0].copy()
        for h in (1, 2):
            bx[self.half == h] *= self.flip[h]
        self.ball_norm = np.stack([bx / paths.PITCH_L + 0.5, self.ball[:, 1] / paths.PITCH_W + 0.5], 1)

    def compute(self, vis_eff, steps):
        """Gap variables at `steps` for the effective mask (real visibility AND NOT synthetic mask)."""
        T = len(vis_eff)
        idx = np.arange(T)
        last = np.where(vis_eff, idx, -1)
        np.maximum.accumulate(last, out=last)
        nxt = np.where(vis_eff, idx, T)
        nxt = np.minimum.accumulate(nxt[::-1])[::-1]
        # never look across half-time
        same_l = (last >= 0) & (self.half[np.clip(last, 0, None)] == self.half)
        same_n = (nxt < T) & (self.half[np.clip(nxt, None, T - 1)] == self.half)
        last = np.where(same_l, last, -1)
        nxt = np.where(same_n, nxt, T)

        s = steps
        L, N = last[s], nxt[s]
        hl, hn = L >= 0, N < T
        Lc, Nc = np.clip(L, 0, T - 1), np.clip(N, 0, T - 1)
        ts = np.where(hl, (s - L) / paths.FPS, 60.0).clip(0, 60)
        tu = np.where(hn, (N - s) / paths.FPS, 60.0).clip(0, 60)
        gap = ts + tu
        rel = np.where(gap > 0, ts / np.maximum(gap, 1e-6), 0.0)
        last_pos = np.where(hl[:, None], self.ball[Lc], 0)
        next_pos = np.where(hn[:, None], self.ball[Nc], 0)
        tl = np.where(hl, self.near_team[Lc], -1)
        tn = np.where(hn, self.near_team[Nc], -1)
        dl = np.where(hl, self.d1[Lc], 9.9).clip(0, 9.9)
        dn = np.where(hn, self.d1[Nc], 9.9).clip(0, 9.9)
        # is the last known position inside the current image?
        q = np.c_[self.ball_norm[Lc], np.ones(len(s))]
        Hs = self.H[s]
        pix = np.einsum("nij,nj->ni", Hs, q)
        with np.errstate(invalid="ignore", divide="ignore"):
            px, py = pix[:, 0] / pix[:, 2], pix[:, 1] / pix[:, 2]
        in_img = hl & (px >= 0) & (px <= paths.IMG_W) & (py >= 0) & (py <= paths.IMG_H)
        return dict(ts=ts.astype(np.float32), tu=tu.astype(np.float32), rel=rel.astype(np.float32),
                    last_pos=np.nan_to_num(last_pos).astype(np.float32),
                    next_pos=np.nan_to_num(next_pos).astype(np.float32),
                    tl=tl.astype(np.int8), tn=tn.astype(np.int8), dl=dl.astype(np.float32),
                    dn=dn.astype(np.float32), in_img=in_img)


def real_gap_lengths():
    """Lengths (frames) of real ball gaps inside graded frames, for the block-mask sampler."""
    F = fr.load()
    s, e = fr.runs(~F["vis"] & F["graded"])
    return (e - s).astype(np.int32)


def load(path=None):
    """Steps cache; `path` (or env POSS_STEPS) selects an alternative label variant."""
    return dict(np.load(path or os.environ.get("POSS_STEPS") or paths.cache("t0_steps.npz")))


if __name__ == "__main__":
    build()
    g = GapContext()
    st = load()["step_frame"]
    out = g.compute(g.vis, st)
    print("gap vars on real mask: median tau_since on hidden steps %.1f s, last-in-image %.3f"
          % (np.median(out["ts"][out["ts"] > 0]), out["in_img"][~g.vis[st]].mean()))
