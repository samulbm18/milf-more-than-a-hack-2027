"""Unit tests of the state machine and its sensors. No match data needed."""
import os

import numpy as np
import pytest

from possession import decoder as dc
from possession import paths
from possession.sensors import Sensor, net_symbols

P = dict(dc.DEFAULT, k=1, mean_pos_s=3.0, chg=1, tov0=1.0, w_b=2.0)


def test_topology_of_the_submitted_machine():
    top = dc.Topology(P["k"], P["r"], P["rs"], P["rl"])
    assert top.names == ["PRE", "PH1", "PH2", "PH3", "PH4", "PA1", "PA2", "PA3", "PA4",
                         "DS1", "DS2", "DS3", "DL1", "DL2", "DL3"]
    assert top.entry(0) == top.names.index("PH1") and top.entry(1) == top.names.index("PA1")
    assert list(top.lab) == [2] + [0] * 4 + [1] * 4 + [2] * 6


def test_coxian_phases_are_distributions_with_the_right_mean():
    m, r = 3.0 * paths.FPS, 4
    cx = dc.coxian(m, r, 0.25)
    assert np.allclose(cx.sum(1), 1)
    assert cx[-1, 1] == 0  # the last phase cannot advance
    # without early exit the sojourn is a sum of r geometric phases of mean m / r each
    assert np.isclose(r / (1 - dc.coxian(m, r, 0.0)[0, 0]), m)


@pytest.mark.parametrize("dead,restart,rho", [(0.0, 0.0, 0.5), (0.9, 0.0, 0.5), (0.3, 1.0, 0.8)])
def test_transitions_leave_each_live_and_dead_state_with_probability_one(dead, restart, rho):
    top = dc.Topology(P["k"], P["r"], P["rs"], P["rl"])
    prm = dc.init_params(P, dc.c_init(np.array([0.5, 0.5])))
    src, kind, base, adv = dc.build_edges(top, prm, P)
    T = 3
    lp = dc.edge_logprob(src[:, 0], kind, base, adv, np.full(T, dead), np.full(T, restart),
                         np.full(T, rho), P, gate=np.ones((T, 2)))
    out = np.zeros((T, top.n))
    for e, s in enumerate(src[:, 0]):
        out[:, s] += np.exp(lp[:, e])
    states = top.P[0] + top.P[1] + top.DS + top.DL
    assert np.allclose(out[:, states], 1.0)


def test_viterbi_follows_clear_evidence():
    # two states that stick, observations favour state 0 then state 1
    src = np.array([0, 0, 1, 1]); dst = np.array([0, 1, 1, 0])
    T = 20
    logA = np.log(np.tile([0.9, 0.1, 0.9, 0.1], (T, 1)))
    logB = np.log(np.where(np.arange(T)[:, None] < 8, [[0.8, 0.2]], [[0.2, 0.8]]))
    path = dc._viterbi(np.log([0.5, 0.5]), src, dst, logA, logB)
    assert list(path) == [0] * 8 + [1] * 12


def test_sensor_stays_bounded_after_em():
    s = Sensor("x", np.zeros(10, np.int64), [[.6, .4], [.4, .6], [.5, .5]], [1, 0], alpha=1.0, cap=4.0)
    s.update(np.array([[1000.0, 0.0], [0.0, 1000.0], [10.0, 10.0]]))  # EM would like a perfect sensor
    r = s.E[0] / s.E[1]
    assert r.max() <= 4.0 * 1.001 and r.min() >= 1 / 4.0 / 1.001
    assert np.allclose(s.E[0], s.E[1][[1, 0]])  # H/A symmetry


def test_network_symbols_bins():
    q = np.array([[0.05, 0.95], [0.2, 0.8], [0.5, 0.5], [0.8, 0.2], [0.95, 0.05]])
    assert list(net_symbols(q)) == [0, 1, 2, 3, 4]


def test_turnover_delay_keeps_the_ball_with_the_last_team():
    if not os.path.isdir(os.path.join(paths.MTAH_ROOT, "evaluation")):
        pytest.skip("challenge repository (official metrics) not found")
    from possession.decode import delay_turnovers
    lab = np.array(list("HHHAAAAADDAAHH"))
    out = "".join(delay_turnovers(lab, 2))
    assert out == "HHHHHAAADDAAAA"  # H->A and A->H move 2 frames; stoppages are not touched


def test_bc_symbols_lead_and_coverage():
    from possession.sensors import bc_symbols
    n = 400
    wide = np.ones(n, bool); wide[162:262] = False            # the director cuts away 62 frames late
    ball = wide.copy(); ball[300:] = False
    cov = np.ones(n, bool); cov[:10] = False
    s = bc_symbols(wide, ball, cov, lead=62)
    assert (s[:10] == -1).all()                               # no broadcast picture: silent
    assert (s[100:200] == 2).all()                            # the whole cut, read 62 frames early
    assert (s[200:262] == 1).all()                            # still no picture of the pitch, hence no ball
    assert (s[262:300] == 0).all() and (s[300:338] == 1).all()


def test_cut_away_inside_a_possession_becomes_a_stoppage():
    from types import SimpleNamespace
    from possession import decode as D
    from possession.sensors import BC_CUT_E0, bc_symbols, on_steps
    n, a0 = paths.N_FRAMES, paths.GRADED[1][0]
    wide = np.ones(n, bool); wide[a0 + 2062:a0 + 2562] = False   # 20 s stoppage at a0+2000, director 62 frames late
    pre = np.zeros(n, bool); pre[paths.GRADED[2][0]:paths.KICKOFF_2] = True
    net = Sensor("net_vis", on_steps(np.full(n, 4)),              # the network keeps saying "H, sure"
                 [[.03, .04, .10, .23, .60], [.60, .23, .10, .04, .03], [.2, .2, .2, .2, .2]], [4, 3, 2, 1, 0], 500.0, cap=4.0)
    bc = Sensor("bc_cut", on_steps(bc_symbols(wide, wide.copy(), np.ones(n, bool))),
                BC_CUT_E0, [0, 1, 2], 500.0, cap=4.0)
    inp = SimpleNamespace(prior=np.array([.5, .5]), vis=np.ones(n, bool), pre=pre, border=None, dead=np.zeros(n),
                          restart=np.zeros(n), rho_h=np.full(n, .5), sensors=[net, bc])
    lab, _, _ = D.decode(inp, P, em_iters=0)
    assert (lab[a0 + 2025:a0 + 2475] == "D").all()               # the stoppage, to within 1 s at each edge
    assert (lab[a0 + 1000:a0 + 1950] == "H").all() and (lab[a0 + 2550:a0 + 3500] == "H").all()


# ---- experiment 17: stoppage edges and quick restarts (all off by default) ----
def _full(seq):
    """Whole-match label array, H everywhere in the graded halves, with seq at the first-half kickoff."""
    lab = np.full(paths.N_FRAMES, "X", dtype="<U1")
    for a, b in paths.GRADED.values():
        lab[a:b + 1] = "H"
    a0 = paths.GRADED[1][0]
    s = np.array(list(seq))
    lab[a0:a0 + len(s)] = s
    return lab, a0


def test_a_stoppage_shorter_than_two_seconds_is_not_a_stoppage():
    from possession.decode import drop_short_stoppages
    lab, a0 = _full("H" * 100 + "D" * 14 + "A" * 100 + "D" * 60 + "H" * 100)
    out = drop_short_stoppages(lab, 50)
    assert "".join(out[a0 + 100:a0 + 114]) == "H" * 14          # the ball stays with the team that had it
    assert (out[a0 + 214:a0 + 274] == "D").all()                # a real stoppage is kept
    assert (out[:a0 + 100] == lab[:a0 + 100]).all() and (out[a0 + 114:] == lab[a0 + 114:]).all()


def test_a_short_stoppage_at_the_kickoff_is_kept():
    from possession.decode import drop_short_stoppages
    lab, a0 = _full("D" * 20 + "H" * 100)
    assert (drop_short_stoppages(lab, 50)[a0:a0 + 20] == "D").all()


def test_a_stoppage_that_ends_just_before_a_tagged_set_piece_reaches_it():
    from possession.decode import extend_to_tags
    lab, a0 = _full("H" * 100 + "D" * 200 + "A" * 400)
    out = extend_to_tags(lab, [a0 + 400], max_gap=150, after=50)        # tag 100 frames after the stoppage ends
    assert (out[a0 + 100:a0 + 450] == "D").all() and (out[a0 + 450:a0 + 700] == "A").all()
    assert (extend_to_tags(lab, [a0 + 600], max_gap=150, after=50) == lab).all()   # 300 frames later: another matter


def test_the_broadcast_sensor_goes_quiet_after_a_cut():
    from possession.sensors import quiet_after_cut
    sym = np.array([0] * 10 + [2] * 5 + [0] * 20 + [2] * 3 + [1] * 4)
    out = quiet_after_cut(sym, 8)
    assert list(out[:15]) == [0] * 10 + [2] * 5
    assert list(out[15:23]) == [-1] * 8 and list(out[23:35]) == [0] * 12
    assert list(out[35:38]) == [2] * 3 and list(out[38:]) == [-1] * 4


def test_set_piece_windows():
    from possession.sensors import set_piece_symbols
    s = set_piece_symbols(300, [100, 250], before=50, after=50)
    assert s[:50].sum() == 0 and (s[50:150] == 1).all() and (s[150:200] == 0).all() and (s[200:300] == 1).all()
