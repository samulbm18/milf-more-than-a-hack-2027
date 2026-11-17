"""Guard rail: nothing tracked by git may be match data, labels or a large binary."""
import os
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBIDDEN_EXT = (".csv", ".npy", ".npz", ".parquet", ".pkl", ".mp4", ".xml", ".txt.gz")
MAX_BYTES = 5 * 1024 * 1024


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    return [f for f in out.splitlines() if f]


def test_no_forbidden_extensions():
    bad = [f for f in tracked_files() if f.lower().endswith(FORBIDDEN_EXT)]
    assert not bad, "data-like files would be committed: %s" % bad


def test_no_large_files():
    bad = [f for f in tracked_files() if os.path.getsize(os.path.join(ROOT, f)) > MAX_BYTES]
    assert not bad, "files over 5 MB would be committed: %s" % bad
