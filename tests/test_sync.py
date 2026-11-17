"""Broadcast-to-tactical synchronisation. Synthetic signals, no match data."""
import numpy as np
import pytest

from possession import paths, sync


def _track(n, seed):
    rng = np.random.default_rng(seed)
    return 0.5 + 0.4 * np.sin(np.cumsum(rng.normal(0, 0.02, (n, 2)), 0))


def _pair(seed=0, shift=4014):
    ref = _track(20000, seed)
    rng = np.random.default_rng(seed + 1)
    ref_ok = rng.random(20000) > 0.4
    qry = ref[shift:shift + 6000] + rng.normal(0, 0.005, (6000, 2))
    qry_ok = rng.random(6000) > 0.5
    return ref, ref_ok, qry, qry_ok


def test_find_offset_recovers_a_known_shift():
    off, rms, n = sync.find_offset(*_pair(), min_common=500)
    assert off == 4014 and n >= 500 and rms < 0.02


def test_alignment_survives_replays():
    ref, ref_ok, qry, qry_ok = _pair()
    rng = np.random.default_rng(9)
    bad = rng.random(len(qry)) < 0.15          # a replay shows some other moment of the match
    qry[bad] = rng.random((int(bad.sum()), 2))
    coarse = sync.find_offset(ref, ref_ok, qry, qry_ok, min_common=500)[0]
    assert abs(coarse - 4014) <= 10            # the mean-squared fit lands inside the refinement radius ...
    assert sync.refine_offset(ref, ref_ok, qry, qry_ok, coarse)[0] == 4014   # ... and the median fit is exact


def test_refine_offset_picks_the_exact_frame():
    assert sync.refine_offset(*_pair(), coarse=4008, radius=15)[0] == 4014


def test_build_fails_loudly_without_the_broadcast_files(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "BC_ATD_TXT", str(tmp_path / "missing.txt"), raising=False)
    with pytest.raises(SystemExit):
        sync.build(verbose=False)
