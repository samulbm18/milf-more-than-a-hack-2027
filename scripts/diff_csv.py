"""Compare two submission files. Prints the number of differing rows and the counts per label pair
(aggregates only); exits with 1 when they differ.

    python scripts/diff_csv.py outputs/a.csv outputs/b.csv
"""
import json
import sys

from possession.pipeline import diff_labels

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    out = diff_labels(sys.argv[1], sys.argv[2])
    print(json.dumps(out))
    sys.exit(1 if out["differing_rows"] else 0)
