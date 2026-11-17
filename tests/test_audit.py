"""Acceptance report. Synthetic labels, no match data (only the official metric module)."""
import os

import numpy as np
import pytest

from possession import paths

if not os.path.isdir(os.path.join(paths.MTAH_ROOT, "evaluation")):
    pytest.skip("challenge repository (official metrics) not found", allow_module_level=True)

from possession import audit, decode  # noqa: E402

A0 = paths.GRADED[1][0]


def _labels(seq):
    lab = np.full(paths.N_FRAMES, "X", dtype="<U1")
    lab[paths.GRADED[1][0]:paths.GRADED[1][1] + 1] = "D"
    lab[paths.GRADED[2][0]:paths.GRADED[2][1] + 1] = "D"
    s = np.array(list(seq))
    lab[A0:A0 + len(s)] = s
    return lab


SEQ = "H" * 100 + "X" * 50 + "H" * 100 + "D" * 200 + "A" * 100 + "X" * 30 + "A" * 70


def test_structure_counts_possessions_as_the_scorer_does():
    st = audit.structure(_labels(SEQ))
    assert st["possessions"] == 2 and st["x_share"] > 0


def test_match_stats_agrees_with_the_scorer():
    assert decode.match_stats(_labels(SEQ))["possessions"] == 2


def test_cuts_are_classified_against_the_labels():
    codes = audit.fill_like_scorer(_labels("H" * 300 + "A" * 300 + "D" * 100))
    out = audit.cuts_vs_labels(codes, [(A0 + 100, A0 + 200), (A0 + 250, A0 + 350), (A0 + 550, A0 + 650)])
    assert (out["inside_one"], out["straddle"], out["touches_d"]) == (1, 1, 1)


def _report(**over):
    r = dict(public=dict(spell=0.65, macro=0.92, d_precision=0.97, d_recall=0.90),
             match=dict(stoppages=105, d_share=0.48, short_stoppages=0),
             tags=dict(set_pieces=58, with_d=56), cuts=dict(inside_one=2),
             sensors=dict(bc_cut=[[.78, .17, .05], [.78, .17, .05], [.12, .23, .65]]))
    for k, v in over.items():
        r[k].update(v)
    return r


def test_verdict_passes_a_healthy_report():
    assert all(audit.verdict(_report()).values())


def test_verdict_flags_missed_stoppages():
    v = audit.verdict(_report(match=dict(stoppages=60, d_share=0.31), cuts=dict(inside_one=37)))
    assert not v["stoppages"] and not v["d_share"] and not v["cut_inside_one"]


def test_verdict_flags_a_dominating_sensor():
    v = audit.verdict(_report(sensors=dict(bc_cut=[[.50, .20, .30], [.50, .20, .30], [.05, .15, .80]])))
    assert not v["bc_sensor"]


def test_stoppage_scoring_counts_detections_and_edge_errors():
    is_dead = np.zeros(paths.N_FRAMES, bool)
    is_dead[1000:1200] = True      # under the first labelled stoppage: starts 25 frames late, ends 50 early
    is_dead[3000:3100] = True      # a predicted stoppage with nothing labelled under it
    is_dead[5000:5200] = True      # outside the window: must not count
    labelled = [(975, 1250), (2000, 2300)]          # the second one is not detected
    out = audit.score_stoppages(labelled, is_dead, (0, 4000))
    assert (out["labelled"], out["detected"], out["recall"]) == (2, 1, 0.5)
    assert (out["predicted"], out["precision"]) == (2, 0.5)
    assert (out["median_start_error_s"], out["median_end_error_s"]) == (1.0, -2.0)
