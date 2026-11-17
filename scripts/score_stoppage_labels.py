"""Score hand-labelled stoppages (tools/label_stoppages.html) against a submission, or against the
public ground truth to calibrate the labeller. Aggregates only; nothing is written.

    python scripts/score_stoppage_labels.py --labels ../labels/stoppages_W1.csv --window 56185 63684 --csv outputs/<name>.csv
    python scripts/score_stoppage_labels.py --labels ../labels/stoppages_W0.csv --window 11185 18684 --against-gt

The labels are a CSV with the header start_frame,end_frame (video frames; end_frame is the first
frame with the ball back in play). Hand labels are used to validate only: they are never part of
the submission and they stay out of the repository.
"""
import argparse
import json

import numpy as np
import pandas as pd

from possession import audit, io

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", required=True)
    ap.add_argument("--window", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", help="submission file to score")
    g.add_argument("--against-gt", action="store_true", help="compare the labels with the public ground truth")
    a = ap.parse_args()

    df = pd.read_csv(a.labels)
    labelled = sorted((int(s), int(e)) for s, e in zip(df["start_frame"], df["end_frame"]) if e > s)
    if a.against_gt:
        is_dead = audit.pm.label_codes(io.load_gt_public()) == audit.pm.D
    else:
        is_dead = audit.fill_like_scorer(np.asarray(pd.read_csv(a.csv)["label"], dtype="<U1")) == audit.pm.D
    print(json.dumps(audit.score_stoppages(labelled, is_dead, tuple(a.window)), indent=1))
