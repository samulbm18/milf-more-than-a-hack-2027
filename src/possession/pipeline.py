"""End-to-end submission from the data bundle and the committed weights (no training).

    python -m possession.pipeline --name <team_name>          # -> outputs/<team_name>.csv
    python -m possession.pipeline --name x --models models/st1 --rebuild --expect <sha256>

Stages:
  1. caches      frames -> events -> proxy -> features   (rules and geometry, built once)
                 + sync: the broadcast recording put on the tactical timeline (sync.py)
  2. inference   PossNet weights as an out-of-fold ensemble, on the CPU in double precision (infer.py)
  3. machine     detectors as bounded sensors -> HSMM (EM 3) -> Viterbi (decode.py, sensors.py)
  4. rules       a stoppage that ends just before a tagged set piece runs up to it; a stoppage under
                 2 s is not one; H<->A changes 10 frames later; X where the largest label score < 0.75
  5. CSV         frame,label for every video frame, and its sha256
"""
import argparse
import collections
import hashlib
import importlib
import json
import os

import numpy as np

from . import paths

# the submitted configuration (experiments 04, 08, 10, 12, 16, 17). device: the networks run on the CPU
# in double precision, the path that does not depend on the GPU model; broadcast: the cut-away sensor;
# min_stop: a stoppage shorter than 2 s is not one; tag_ext: a stoppage that ends at most 6 s before a
# tagged set piece runs up to it. Earlier candidates:  --min-stop 0 --tag-ext 0  (experiment 16) and,
# on top of that,  --broadcast off  (experiment 12).
FINAL = dict(models=os.path.join(paths.REPO, "models", "st1"), em=3, delay=10, abstain=0.75, device="cpu",
             broadcast=True, alpha_bc=None, min_stop=50, tag_ext=150)
CACHES = (("frames", "t0_frames.npz"), ("events", "t1_events.npz"),
          ("proxy", "t1_proxy.npz"), ("features", "t0_steps.npz"))
# only needed by the broadcast sensor: the first submission stays reproducible without the broadcast files
BC_CACHE = ("sync", "t0_broadcast.npz")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_expect(path, expect):
    """Stop when the file is not the expected one: the rules ask for the same file, not a similar one."""
    if not expect:
        return
    got = sha256(path)
    if got != expect.strip().lower():
        raise SystemExit("sha256 mismatch for %s\n  expected %s\n  got      %s" % (path, expect.strip(), got))


def diff_labels(a_path, b_path):
    """How two submissions differ: number of rows and counts per 'old>new' label pair."""
    import pandas as pd
    a, b = (pd.read_csv(p)["label"].values for p in (a_path, b_path))
    if len(a) != len(b):
        raise SystemExit("row counts differ: %d vs %d" % (len(a), len(b)))
    m = a != b
    pairs = collections.Counter("%s>%s" % (x, y) for x, y in zip(a[m], b[m]))
    return {"differing_rows": int(m.sum()), "pairs": dict(sorted(pairs.items()))}


def package(team, csv_path, out_dir=None, expect=None, doc=None):
    """<team>.csv, a byte copy of the verified CSV named after the team as the form asks, plus a copy of
    the technical document `doc` if one is given. Refuses a CSV that is not the one recorded in
    SUBMISSION.sha256. Returns the folder."""
    import shutil
    if expect is None:
        with open(os.path.join(paths.REPO, "SUBMISSION.sha256"), encoding="utf-8") as f:
            expect = f.read()
    check_expect(csv_path, expect)
    out_dir = out_dir or os.path.join(paths.OUTPUTS, "entrega")
    os.makedirs(out_dir, exist_ok=True)
    name = "".join(ch for ch in team.strip() if ch not in '\\/:*?"<>|')
    shutil.copyfile(csv_path, os.path.join(out_dir, name + ".csv"))
    if doc:  # the technical document is submitted separately (PDF); copy one only when given
        shutil.copyfile(doc, os.path.join(out_dir, name + os.path.splitext(doc)[1]))
    with open(os.path.join(out_dir, "SHA256"), "w", encoding="utf-8", newline="\n") as f:
        f.write("%s  %s.csv\n" % (sha256(csv_path), name))
    return out_dir


def build_caches(rebuild=False, broadcast=False):
    for mod, fname in CACHES + ((BC_CACHE,) if broadcast else ()):
        if rebuild or not os.path.exists(paths.cache(fname)):
            print("[caches] building %s" % fname, flush=True)
            m = importlib.import_module("possession." + mod)
            m.build() if hasattr(m, "build") else m.main()


def ensemble(name, models=FINAL["models"], rebuild=False, device=FINAL["device"], broadcast=FINAL["broadcast"]):
    """Stages 1-2: caches and the out-of-fold network ensemble -> outputs/<name>_oof_ensemble.npz."""
    from . import train
    train.set_device(device)
    from . import infer
    build_caches(rebuild, broadcast)
    oof = os.path.join(paths.OUTPUTS, name + "_oof_ensemble.npz")
    print("[inference] %s" % models, flush=True)
    infer.run(models, oof)
    return oof


def machine(name, oof, em=FINAL["em"], delay=FINAL["delay"], abstain=FINAL["abstain"],
            broadcast=FINAL["broadcast"], alpha_bc=FINAL["alpha_bc"],
            min_stop=FINAL["min_stop"], tag_ext=FINAL["tag_ext"], bc_wait=0, set_piece=False, dead_model=None):
    """Stages 3-5 on an existing network ensemble: machine, final rules, outputs/<name>.csv.

    min_stop and tag_ext are the two rules kept from experiment 17 (stoppage edges); the last three
    arguments are the options it discarded, off in the submitted configuration.
    """
    from . import audit, decode as D, decoder as dc, frames as fr, sensors as SN
    print("[machine] sensors + HSMM, EM %d%s" % (em, ", broadcast cut-away sensor" if broadcast else ""), flush=True)
    S = np.load(paths.cache("t0_steps.npz"))
    sym = S["sym"]
    prior = np.bincount(sym[sym >= 0], minlength=2) / (sym >= 0).sum()
    inp = D.Inputs(D.q_from_oof(oof), prior, dead_model)
    inp.border = D.border_from_oof(oof)
    use = SN.MACHINE_USE_BC if broadcast else SN.MACHINE_USE
    use += (("set_piece",) if set_piece else ()) + (("dead_model",) if dead_model else ())
    inp.sensors = SN.build(inp, fr.load(), alpha=500.0, use=use, cap=4.0, alpha_bc=alpha_bc, bc_wait=bc_wait)
    p = dict(dc.DEFAULT, **D.MACHINE_P)
    labels, post, _ = D.decode(inp, p, em_iters=em)
    with open(os.path.join(paths.OUTPUTS, name + "_sensors.json"), "w", encoding="utf-8") as f:
        json.dump({s.name: np.round(s.E, 4).tolist() for s in inp.sensors}, f, indent=1)  # what the EM learnt

    if tag_ext:
        play = audit.play_mask()
        starts = [t["f0"] for t in inp.tags if t["code"] == "SET PIECES" and play[t["f0"]]]
        labels = D.extend_to_tags(labels, starts, tag_ext, SN.SP_AFTER)
    if min_stop:
        labels = D.drop_short_stoppages(labels, min_stop)
    print("[rules] stoppage to tag <= %d frames, min stoppage %d frames, delay %d frames, abstain < %.2f"
          % (tag_ext, min_stop, delay, abstain), flush=True)
    if delay:
        labels = D.delay_turnovers(labels, delay)
    if abstain:
        labels[(labels != "X") & (post.max(1) < abstain)] = "X"

    path = D.write_submission(labels, name)
    score = D.score_folds(labels, inp.gt)
    print(json.dumps(dict(public_window=score["full"], match=D.match_stats(labels))))
    print("wrote", path)
    return path


def run(name, models=FINAL["models"], em=FINAL["em"], delay=FINAL["delay"], abstain=FINAL["abstain"],
        rebuild=False, device=FINAL["device"], expect=None, broadcast=FINAL["broadcast"],
        alpha_bc=FINAL["alpha_bc"], min_stop=FINAL["min_stop"], tag_ext=FINAL["tag_ext"]):
    oof = ensemble(name, models, rebuild, device, broadcast)
    path = machine(name, oof, em, delay, abstain, broadcast, alpha_bc, min_stop, tag_ext)
    print("sha256", sha256(path))
    check_expect(path, expect)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="submission")
    ap.add_argument("--models", default=FINAL["models"])
    ap.add_argument("--em", type=int, default=FINAL["em"])
    ap.add_argument("--delay", type=int, default=FINAL["delay"])
    ap.add_argument("--abstain", type=float, default=FINAL["abstain"])
    ap.add_argument("--rebuild", action="store_true", help="rebuild every cache from the raw data")
    ap.add_argument("--device", choices=("cpu", "cuda"), default=FINAL["device"],
                    help="where the networks run; cpu runs in double precision and is the reproducible path")
    ap.add_argument("--expect", help="sha256 the CSV must have; the run fails otherwise")
    ap.add_argument("--broadcast", choices=("on", "off"), default="on" if FINAL["broadcast"] else "off",
                    help="use the broadcast cut-away sensor (experiment 16)")
    ap.add_argument("--alpha-bc", type=float, default=FINAL["alpha_bc"],
                    help="pseudo-count of the broadcast sensor in the EM (1e9 keeps it fixed)")
    ap.add_argument("--min-stop", type=int, default=FINAL["min_stop"],
                    help="frames below which a stoppage is not one (experiment 17); 0 turns the rule off")
    ap.add_argument("--tag-ext", type=int, default=FINAL["tag_ext"],
                    help="a stoppage ending at most this many frames before a tagged set piece runs up to "
                         "it (experiment 17); 0 turns the rule off")
    a = ap.parse_args()
    run(a.name, a.models, a.em, a.delay, a.abstain, a.rebuild, a.device, a.expect, a.broadcast == "on", a.alpha_bc,
        a.min_stop, a.tag_ext)


if __name__ == "__main__":
    main()
