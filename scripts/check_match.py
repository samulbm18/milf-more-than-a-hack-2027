"""Acceptance report of a submission file (aggregates only, nothing is written).

    python scripts/check_match.py --csv outputs/<name>.csv [--sensors outputs/<name>_sensors.json]

Prints the report, one boolean per criterion of possession.audit.GATE, and GATE PASS / GATE FAIL.
"""
import argparse
import json

import numpy as np
import pandas as pd

from possession import audit

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sensors", help="outputs/<name>_sensors.json written by the pipeline")
    a = ap.parse_args()
    labels = np.asarray(pd.read_csv(a.csv)["label"], dtype="<U1")
    sensors = json.load(open(a.sensors, encoding="utf-8")) if a.sensors else None
    rep = audit.report(labels, sensors)
    ver = audit.verdict(rep)
    print(json.dumps(rep, indent=1))
    print(json.dumps(ver))
    failed = [k for k, v in ver.items() if not v]
    print("GATE PASS" if not failed else "GATE FAIL: " + ", ".join(failed))
