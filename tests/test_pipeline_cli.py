"""Plumbing that makes the submission reproducible: file hash, device choice, precision.
No match data needed."""
import importlib
import os
from types import SimpleNamespace

import pytest
import torch
import yaml

from possession import paths, pipeline

TINY_CSV = b"frame,label\n0,X\n"
TINY_SHA = "080070e6c1ee92445eda3c35a5baddc49e7d77ef5cf13a2a45b774caa271a76d"


def test_sha256_of_a_known_file(tmp_path):
    f = tmp_path / "x.csv"
    f.write_bytes(TINY_CSV)
    assert pipeline.sha256(str(f)) == TINY_SHA


def test_expect_mismatch_stops_the_run(tmp_path):
    f = tmp_path / "x.csv"
    f.write_bytes(TINY_CSV)
    pipeline.check_expect(str(f), None)        # nothing expected: nothing to check
    pipeline.check_expect(str(f), TINY_SHA)    # the right hash passes
    with pytest.raises(SystemExit):
        pipeline.check_expect(str(f), "00")


def test_device_can_be_forced_to_cpu(monkeypatch):
    monkeypatch.setenv("POSS_DEVICE", "cpu")
    from possession import train
    assert importlib.reload(train).DEV == "cpu"


def test_cpu_defaults_to_double_precision_and_gpu_to_single():
    from possession import train
    assert train.set_device("cpu") == ("cpu", torch.float64)
    assert train.set_device("cuda") == ("cuda", torch.float32)
    assert train.set_device("cpu", "float32") == ("cpu", torch.float32)


def _batch(B, L, N, seed):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g)  # noqa: E731
    team = (r(B, L, N) < 0.5).to(torch.int8)
    team[..., -3:] = -1                                        # three padded tokens
    return dict(
        p_pos=(r(B, L, N, 2) - 0.5) * 60, p_vel=(r(B, L, N, 2) - 0.5) * 8, p_acc=r(B, L, N) * 3,
        p_team=team, p_gk=r(B, L, N) < 0.07, p_age=r(B, L, N) * 30,
        b_pos=(r(B, L, 2) - 0.5) * 60, b_vel=(r(B, L, 2) - 0.5) * 20, b_acc=r(B, L) * 10,
        b_vis=r(B, L) < 0.7, cam=(r(B, L, 2) - 0.5) * 40,
        g_ts=r(B, L) * 5, g_tu=r(B, L) * 5, g_rel=r(B, L), g_last=(r(B, L, 2) - 0.5) * 60,
        g_next=(r(B, L, 2) - 0.5) * 60, g_tl=(r(B, L) < 0.5).to(torch.int8), g_tn=(r(B, L) < 0.5).to(torch.int8),
        g_dl=r(B, L) * 5, g_dn=r(B, L) * 5, g_img=r(B, L) < 0.8)


def test_double_precision_inference_computes_the_same_network():
    """The canonical CPU path must be the same function as the single-precision one, only more exact."""
    from possession import train
    cfg = yaml.safe_load(open(os.path.join(paths.REPO, "configs", "possnet.yaml"), encoding="utf-8"))
    B, L, N = 2, 12, 30
    M = SimpleNamespace(masked=torch.zeros(L, dtype=torch.bool))
    idx = torch.arange(L)[None].repeat(B, 1)

    train.set_device("cpu", "float32")
    torch.manual_seed(0)
    m32 = train.build_model(cfg).eval()
    out32 = train.run_model(m32, M, _batch(B, L, N, 1), idx, cfg, train=False)[0]

    train.set_device("cpu")                                      # canonical: double precision
    m64 = train.build_model(cfg).eval()
    m64.load_state_dict(m32.state_dict())
    out64 = train.run_model(m64, M, _batch(B, L, N, 1), idx, cfg, train=False)[0]

    for k in ("sym", "border"):
        assert out32[k].dtype == torch.float32 and out64[k].dtype == torch.float64
        assert torch.allclose(out64[k], out32[k].double(), atol=1e-3)
        assert not torch.equal(out64[k], out32[k].double())      # really computed in double, not cast at the end


def test_diff_counts_changed_rows_by_label_pair(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("frame,label\n0,H\n1,H\n2,A\n3,D\n")
    b.write_text("frame,label\n0,H\n1,A\n2,A\n3,X\n")
    assert pipeline.diff_labels(str(a), str(b)) == {"differing_rows": 2, "pairs": {"D>X": 1, "H>A": 1}}
    assert pipeline.diff_labels(str(a), str(a)) == {"differing_rows": 0, "pairs": {}}


def test_inference_uses_the_other_steps_of_the_window():
    """The temporal transformer must see its 30 s of context on every device. PyTorch's inference
    fast path (eval mode, no autocast) turns the additive ALiBi bias into a mask that leaves each
    step alone with itself: moving the ball at step 0 would then leave step 5 untouched."""
    from possession import train
    cfg = yaml.safe_load(open(os.path.join(paths.REPO, "configs", "possnet.yaml"), encoding="utf-8"))
    train.set_device("cpu")
    torch.manual_seed(0)
    model = train.build_model(cfg).eval()
    B, L, N = 1, 12, 30
    M = SimpleNamespace(masked=torch.zeros(L, dtype=torch.bool))
    idx = torch.arange(L)[None]
    a, b = _batch(B, L, N, 3), _batch(B, L, N, 3)
    b["b_vis"][:] = True
    a["b_vis"][:] = True
    b["b_pos"][:, 0] += 15.0                                     # the ball is elsewhere at step 0 only
    with torch.no_grad():
        out_a = train.run_model(model, M, a, idx, cfg, train=False)[0]["sym"]
        out_b = train.run_model(model, M, b, idx, cfg, train=False)[0]["sym"]
    assert (out_a[:, 0] - out_b[:, 0]).abs().max() > 1e-6        # step 0 itself changed
    assert (out_a[:, 5] - out_b[:, 5]).abs().max() > 1e-6        # and step 5 noticed


def test_package_names_the_files_after_the_team_and_refuses_another_file(tmp_path):
    src = tmp_path / "x.csv"
    src.write_bytes(TINY_CSV)
    doc = tmp_path / "TECHNICAL.md"
    doc.write_text("# doc", encoding="utf-8")
    out = pipeline.package("Mi Equipo", str(src), str(tmp_path / "entrega"), expect=TINY_SHA, doc=str(doc))
    assert sorted(os.listdir(out)) == ["Mi Equipo.csv", "Mi Equipo.md", "SHA256"]
    assert (tmp_path / "entrega" / "Mi Equipo.csv").read_bytes() == TINY_CSV      # the same bytes, not a rewrite
    assert (tmp_path / "entrega" / "SHA256").read_text().split()[0] == TINY_SHA
    with pytest.raises(SystemExit):                                              # not the verified file
        pipeline.package("Mi Equipo", str(src), str(tmp_path / "otra"), expect="00", doc=str(doc))
