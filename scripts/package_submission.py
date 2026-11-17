"""Prepare the files the submission form asks for, named after the team.

    python scripts/package_submission.py "<team name>"                       # runs the whole pipeline first
    python scripts/package_submission.py "<team name>" outputs/<name>.csv    # reuses a CSV already generated

Writes outputs/entrega/<team name>.csv (a byte copy of the verified CSV) and SHA256; the technical
document (PDF) is submitted separately. A CSV whose sha256 is not the one in SUBMISSION.sha256 is refused.
Then applies the checks of the official validator and prints what the file says (aggregates only).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from possession import audit, io, paths, pipeline

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    team = sys.argv[1]
    with open(os.path.join(paths.REPO, "SUBMISSION.sha256"), encoding="utf-8") as f:
        expect = f.read().strip()
    csv = sys.argv[2] if len(sys.argv) == 3 else pipeline.run("submission", expect=expect)
    out = pipeline.package(team, csv)                       # stops here if it is not the verified file

    reason = audit.pm.validate_submission(audit.pm.read_label_csv(csv), paths.N_FRAMES)
    if reason:
        sys.exit("the official validator rejects the file: %s" % reason)
    labels = np.asarray(pd.read_csv(csv)["label"], dtype="<U1")
    print("official validation: OK, %d rows" % len(labels))
    print("sha256 %s" % expect)
    print("public window:", json.dumps(audit.public_metrics(labels, io.load_gt_public())))
    print("whole match  :", json.dumps(audit.structure(labels), default=float))
    print("files in %s: %s" % (out, sorted(os.listdir(out))))
