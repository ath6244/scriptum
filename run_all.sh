#!/usr/bin/env bash
# Run the full pipeline (steps 1-12 of the README's reproduction table) in order.
#
# Usage (from the repo root; on Windows use Git Bash):
#   ./run_all.sh                  run every step; long steps skip runs already saved in results/
#   ./run_all.sh --from 5         start at step 5 (e.g. after an interruption)
#   ./run_all.sh --from-scratch   delete results/, figures/ and splits.json first, recompute everything,
#                                 and capture the run logs the report cites
#   PYTHON=.venv/bin/python ./run_all.sh   use a specific interpreter (default: python)
#
# Without --from-scratch the committed run logs are left untouched, because re-running a step whose
# results are already saved would replace a real log with "skip" lines.

set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
FROM=1
FRESH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --from-scratch) FRESH=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

if [ "$FRESH" = 1 ]; then
  echo "Deleting results/, figures/ and splits.json (restore with: git checkout -- results figures splits.json)"
  rm -rf results figures splits.json
  mkdir -p results figures
fi

# Run a command; when recomputing from scratch, also save its console output as results/<log>.
run_logged() {
  local log="$1"; shift
  if [ "$FRESH" = 1 ]; then "$@" | tee "results/$log"; else "$@"; fi
}

step() {  # step NUMBER "description" command...
  local n="$1" description="$2"; shift 2
  if [ "$n" -lt "$FROM" ]; then return; fi
  echo; echo "=== Step $n: $description ==="
  "$@"
}

prepare_data() {
  if [ -f dataset.csv ] && [ "$FRESH" = 0 ]; then
    echo "dataset.csv exists (it is rebuilt identically every time); skipping. Delete it to rebuild."
  else
    "$PY" scripts/prepare_arena.py
  fi
}

step 1  "Build dataset.csv from LMArena"           prepare_data
step 2  "Grouped splits and masking stats"         "$PY" scripts/make_splits.py
step 3  "Baselines (majority, length, TF-IDF)"     "$PY" scripts/baselines.py
step 4  "TextCNN and BiLSTM"                       run_logged run_neural_log.txt  "$PY" -u scripts/run_neural.py
step 5  "Bootstrap CIs, McNemar, binomial tests"   "$PY" scripts/stats_rq1_rq2.py
step 6  "Near-duplicate prompt check"              "$PY" scripts/check_near_duplicates.py
step 7  "RQ3 leave-one-domain-out"                 run_logged rq3_log.txt         "$PY" -u scripts/rq3_cross_domain.py
step 8  "RQ4 ablations (CPU part)"                 run_logged rq4_tfidf_log.txt   "$PY" -u scripts/rq4_ablations.py --models tfidf_lr length_lr
step 8  "RQ4 ablations (TextCNN)"                  run_logged rq4_textcnn_log.txt "$PY" -u scripts/rq4_ablations.py --models textcnn
step 9  "RQ4 style features and top words"         "$PY" scripts/rq4_features.py
step 10 "PII scan and data/sample.csv"             "$PY" scripts/scan_pii.py
step 11 "Food for thought"                         "$PY" scripts/score_reply.py
step 12 "Build report/report.pdf"                  "$PY" scripts/build_report.py

echo; echo "All steps finished."
