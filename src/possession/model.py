"""PossNet (M4): set encoder per step -> bidirectional temporal transformer -> four heads.

`featurize` turns raw per-step geometry into model inputs on the GPU, after applying the
augmentation flips to the geometry itself.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

HL, HW = 52.65, 34.0
D_PLAYER, D_BALL, D_GAP, D_CTX = 21, 8, 15, 18

# The temporal transformer adds an ALiBi bias (a float mask) to its attention. PyTorch's inference
# "fast path" (eval mode, no gradients, no autocast) only understands boolean masks: it would turn
# the bias into "every step sees only itself". It is never taken while training nor under the GPU
# autocast the networks were trained and first run with, so switching it off changes nothing there
# and makes every other device (CPU, GPU without autocast) compute the network that was trained.
torch.backends.mha.set_fastpath_enabled(False)


# ------------------------------------------------------------------------------- featurize
def featurize(b, masked, flip_y, swap):
    """b: dict of [B,L,...] raw tensors. masked: [B,L] bool (ball hidden for the model).
    flip_y, swap: [B] bool. Returns player tokens, padding mask, ball token, gap and context."""
    sy = torch.where(flip_y, -1.0, 1.0)[:, None, None]
    sx = torch.where(swap, -1.0, 1.0)[:, None, None]
    team = b["p_team"].long()
    pad = team < 0
    team = torch.where(swap[:, None, None] & ~pad, 1 - team, team)

    pp = b["p_pos"] * torch.stack([sx, sy], -1)
    pv = b["p_vel"] * torch.stack([sx, sy], -1)
    bp = b["b_pos"] * torch.cat([sx, sy], -1)
    bv = b["b_vel"] * torch.cat([sx, sy], -1)
    hidden = masked | ~b["b_vis"]
    hid = hidden[..., None]

    is_h = (team == 0).float()
    is_a = (team == 1).float()
    spd = pv.norm(dim=-1)
    # pairwise distances inside the step for pressure features
    dd = torch.cdist(pp.flatten(0, 1), pp.flatten(0, 1)).view(*pp.shape[:3], pp.shape[2])
    opp = (team[..., :, None] != team[..., None, :]) & ~pad[..., None, :] & ~pad[..., :, None]
    dd_opp = torch.where(opp, dd, torch.full_like(dd, 99.0))
    d_opp = dd_opp.min(-1).values.clamp(max=30)
    n_opp3 = (dd_opp < 3).sum(-1).float()
    own_goal_x = torch.where(team == 0, -HL, HL)
    d_goal_opp = torch.hypot(pp[..., 0] + own_goal_x, pp[..., 1])
    d_goal_own = torch.hypot(pp[..., 0] - own_goal_x, pp[..., 1])
    age = b["p_age"]

    rel = bp[..., None, :] - pp
    dball = rel.norm(dim=-1)
    ang = torch.atan2(rel[..., 1], rel[..., 0])
    relv = (bv[..., None, :] - pv).norm(dim=-1)
    big = torch.where(pad, torch.full_like(dball, 1e3), dball)
    rank = torch.zeros_like(dball)
    for t in (0, 1):
        m = team == t
        dt = torch.where(m, big, torch.full_like(big, 1e3))
        rank = torch.where(m, dt.argsort(-1).argsort(-1).float(), rank)
    ball_rel = torch.stack([(dball / 10).clamp(max=5), ang.cos(), ang.sin(), (relv / 8).clamp(max=5),
                            rank / 11], -1)
    ball_rel = torch.where(hid[..., None], torch.zeros_like(ball_rel), ball_rel)

    player = torch.cat([
        pp[..., :1] / HL, pp[..., 1:] / HW, pv / 8, spd[..., None] / 8, (b["p_acc"] / 5).clamp(max=5)[..., None],
        is_h[..., None], is_a[..., None], b["p_gk"].float()[..., None],
        (d_opp / 10)[..., None], (n_opp3 / 3)[..., None], (d_goal_opp / 105)[..., None],
        (d_goal_own / 105)[..., None], (torch.log1p(age) / 3)[..., None], (age < 1).float()[..., None],
        ball_rel, hidden[:, :, None, None].expand(-1, -1, pp.shape[2], 1).float()], -1)
    player = torch.where(pad[..., None], torch.zeros_like(player), player)

    inside = ((bp[..., 0].abs() <= HL) & (bp[..., 1].abs() <= HW)).float()
    ball = torch.cat([bp[..., :1] / HL, bp[..., 1:] / HW, bv / 10, bv.norm(dim=-1, keepdim=True) / 10,
                      (b["b_acc"][..., None] / 20).clamp(max=5), inside[..., None]], -1)
    ball = torch.where(hid, torch.zeros_like(ball), ball)
    ball = torch.cat([ball, hid.float()], -1)

    # gap variables (raw ends flipped like everything else)
    lp = b["g_last"] * torch.cat([sx, sy], -1)
    npos = b["g_next"] * torch.cat([sx, sy], -1)
    tl, tn = b["g_tl"].long(), b["g_tn"].long()
    tl = torch.where(swap[:, None] & (tl >= 0), 1 - tl, tl)
    tn = torch.where(swap[:, None] & (tn >= 0), 1 - tn, tn)
    oh = lambda t: torch.stack([(t == 0).float(), (t == 1).float()], -1)  # noqa: E731
    gap = torch.cat([
        (torch.log1p(b["g_ts"]) / 4)[..., None], (torch.log1p(b["g_tu"]) / 4)[..., None], b["g_rel"][..., None],
        lp[..., :1] / HL, lp[..., 1:] / HW, npos[..., :1] / HL, npos[..., 1:] / HW, oh(tl), oh(tn),
        ((tl == tn) & (tl >= 0)).float()[..., None], (b["g_dl"] / 5)[..., None], (b["g_dn"] / 5)[..., None],
        b["g_img"].float()[..., None]], -1)

    ctx = []
    for t in (0, 1):
        m = (team == t)
        mf = m.float()
        n = mf.sum(-1).clamp(min=1)
        cx = (pp[..., 0] * mf).sum(-1) / n
        cy = (pp[..., 1] * mf).sum(-1) / n
        xmax = torch.where(m, pp[..., 0], torch.full_like(mf, -99)).max(-1).values
        xmin = torch.where(m, pp[..., 0], torch.full_like(mf, 99)).min(-1).values
        ymax = torch.where(m, pp[..., 1], torch.full_like(mf, -99)).max(-1).values
        ymin = torch.where(m, pp[..., 1], torch.full_like(mf, 99)).min(-1).values
        has = mf.sum(-1) > 0
        length = torch.where(has, xmax - xmin, torch.zeros_like(cx))
        width = torch.where(has, ymax - ymin, torch.zeros_like(cx))
        mvx = (pv[..., 0] * mf).sum(-1) / n
        msp = (spd * mf).sum(-1) / n
        # second-to-last defender: H defends -x, A defends +x
        key = pp[..., 0] if t == 0 else -pp[..., 0]
        key = torch.where(m, key, torch.full_like(key, 99))
        line = key.sort(-1).values[..., 1].clamp(-HL, HL)
        line = line if t == 0 else -line
        ctx += [cx / HL, cy / HW, length / 50, width / 50, mvx / 8, msp / 8, line / HL, n / 11]
    cam = b["cam"] * torch.cat([sx, sy], -1)
    ctx += [cam[..., 0] / HL, cam[..., 1] / HW]
    ctx = torch.stack(ctx, -1)
    return player, pad, ball, gap, ctx


def swap_symbols(sym, swap):
    s = sym.clone()
    s[swap[:, None].expand_as(s) & (sym == 0)] = 1
    s[swap[:, None].expand_as(s) & (sym == 1)] = 0
    return s


# ------------------------------------------------------------------------------- modules
class MAB(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, q, k, pad=None):
        h = q + self.att(self.n1(q), self.n1(k), self.n1(k), key_padding_mask=pad, need_weights=False)[0]
        return h + self.ff(self.n2(h))


class SetEncoder(nn.Module):
    def __init__(self, d=64, heads=4, layers=2, seeds=4, d_out=128):
        super().__init__()
        self.phi_p = nn.Sequential(nn.Linear(D_PLAYER, d), nn.GELU(), nn.Linear(d, d))
        self.phi_b = nn.Sequential(nn.Linear(D_BALL, d), nn.GELU(), nn.Linear(d, d))
        self.sab = nn.ModuleList([MAB(d, heads) for _ in range(layers)])
        self.seeds = nn.Parameter(torch.randn(1, seeds, d) * 0.1)
        self.pma = MAB(d, heads)
        self.out = nn.Linear(seeds * d, d_out)

    def forward(self, player, pad, ball):
        B, L, N, _ = player.shape
        x = torch.cat([self.phi_b(ball)[..., None, :], self.phi_p(player)], 2).flatten(0, 1)
        m = torch.cat([torch.zeros_like(pad[..., :1]), pad], 2).flatten(0, 1)
        for blk in self.sab:
            x = blk(x, x, m)
        s = self.pma(self.seeds.expand(x.shape[0], -1, -1), x, m)
        f = self.out(s.flatten(1)).view(B, L, -1)
        return f, x[:, 1:].view(B, L, N, -1)


def alibi_bias(L, heads, device):
    slopes = torch.tensor([2 ** (-8 * (i + 1) / heads) for i in range(heads)], device=device)
    pos = torch.arange(L, device=device)
    dist = (pos[None, :] - pos[:, None]).abs().float()
    return -slopes[:, None, None] * dist[None]


class PossNet(nn.Module):
    def __init__(self, d_set=64, set_layers=2, seeds=4, d=128, layers=4, heads=4, ff=256, dropout=0.1,
                 n_mdn=3, n_sym=5):
        super().__init__()
        self.heads = heads
        self.set = SetEncoder(d_set, 4, set_layers, seeds, d)
        self.side = nn.Linear(D_GAP + D_CTX, 32)
        self.inp = nn.Linear(d + 32, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.temporal = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.h_sym = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_sym))
        self.h_ptr = nn.Sequential(nn.Linear(d + d_set, d), nn.GELU(), nn.Linear(d, 1))
        self.h_none = nn.Linear(d, 1)
        self.h_mdn = nn.Linear(d, n_mdn * 6)
        self.h_border = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.n_mdn = n_mdn

    def forward(self, player, pad, ball, gap, ctx):
        f, e = self.set(player, pad, ball)
        z = self.inp(torch.cat([f, Fn.gelu(self.side(torch.cat([gap, ctx], -1)))], -1))
        B, L, _ = z.shape
        bias = alibi_bias(L, self.heads, z.device).repeat(B, 1, 1).to(z.dtype)
        h = self.norm(self.temporal(z, mask=bias))
        sym = self.h_sym(h)
        ptr = self.h_ptr(torch.cat([h[..., None, :].expand(-1, -1, e.shape[2], -1), e], -1)).squeeze(-1)
        ptr = ptr.masked_fill(pad, -1e4)
        ptr = torch.cat([ptr, self.h_none(h)], -1)  # last index = nobody
        mdn = self.h_mdn(h).view(B, L, self.n_mdn, 6)
        border = self.h_border(h).squeeze(-1)
        return dict(sym=sym, ptr=ptr, mdn=mdn, border=border)


def mdn_nll(mdn, target):
    """Mixture of axis-aligned 2D gaussians in metres / 10."""
    logit, mu, logs = mdn[..., 0], mdn[..., 1:3], mdn[..., 3:5].clamp(-4, 3)
    x = (target / 10)[..., None, :]
    lp = -0.5 * (((x - mu) / logs.exp()) ** 2).sum(-1) - logs.sum(-1) - math.log(2 * math.pi)
    return -(torch.logsumexp(lp + Fn.log_softmax(logit, -1), -1))
