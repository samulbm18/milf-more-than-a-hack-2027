#!/usr/bin/env bash
# Regenerate the submission from a CLEAN COPY of what git would publish (no cache/, runs/, outputs/),
# rebuilding every cache from the raw data, and check the sha256 of the CSV.
#   bash scripts/repro_check.sh [cpu|cuda] [sha256] [extra pipeline args...]
# Without a sha256 the one in SUBMISSION.sha256 is used. Needs the challenge data bundle
# (MTAH_ROOT, default ../more-than-a-hack-2027). The copy is deleted at the end unless KEEP=1.
set -e
cd "$(dirname "$0")/.."
REPO=$(pwd)
DEVICE=${1:-cpu}
EXPECT=${2:-$(cat SUBMISSION.sha256 2>/dev/null || true)}
shift $(( $# < 2 ? $# : 2 ))
PY=${PY:-$REPO/.venv/Scripts/python}; [ -x "$PY" ] || PY=python
MTAH=${MTAH_ROOT:-$REPO/../more-than-a-hack-2027}
native() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else echo "$1"; fi; }

TMP=$(mktemp -d)
trap '[ "${KEEP:-0}" = 1 ] || rm -rf "$TMP"' EXIT
git ls-files --cached --others --exclude-standard -z | xargs -0 cp --parents -t "$TMP"
echo "clean copy: $(git ls-files --cached --others --exclude-standard | wc -l) files in $TMP"

cd "$TMP"
export MTAH_ROOT="$(native "$(cd "$MTAH" && pwd)")"
export PYTHONPATH="$(native "$TMP/src")"
"$PY" -u -m possession.pipeline --name repro --rebuild --device "$DEVICE" ${EXPECT:+--expect "$EXPECT"} "$@"
echo "REPRO_OK device=$DEVICE"
