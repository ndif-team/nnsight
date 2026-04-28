#!/usr/bin/env bash
# Run the full eval suite against each documentation bundle, save JSON.
#
# Usage:
#   bash scripts/run_bundle_study.sh
#
# Outputs:
#   results/<bundle>.json
#   results/<bundle>.log
#
# Takes ~80-100 minutes wall time (4 bundles × 65 tasks).

set -u  # error on unset vars (don't `set -e` — we want to keep going past per-task failures)

cd "$(dirname "$0")/.." || exit 1

source /disk/u/jadenfk/miniconda3/etc/profile.d/conda.sh
conda activate nn6

mkdir -p results

MODE="${MODE:-static}"     # "static" or "browse"
MODEL="${MODEL:-sonnet}"

# Static and browse have different bundle sets.
if [ "${MODE}" = "browse" ]; then
    BUNDLES=(minimal router full)
else
    BUNDLES=(minimal router full legacy)
fi

# Output suffix so static / browse runs don't clobber each other.
SUFFIX=""
if [ "${MODE}" = "browse" ]; then
    SUFFIX=".browse"
fi

echo "$(date '+%Y-%m-%d %H:%M:%S')  bundle study starting"
echo "  mode:    ${MODE}"
echo "  model:   ${MODEL}"
echo "  bundles: ${BUNDLES[*]}"
echo

for BUNDLE in "${BUNDLES[@]}"; do
    OUT="results/${BUNDLE}${SUFFIX}.json"
    LOG="results/${BUNDLE}${SUFFIX}.log"
    echo "$(date '+%H:%M:%S')  --- bundle: ${BUNDLE} (${MODE}) ---"
    python eval.py \
        --provider claude-code \
        --model "${MODEL}" \
        --mode "${MODE}" \
        --doc-bundle "${BUNDLE}" \
        --output "${OUT}" \
        --verbose \
        > "${LOG}" 2>&1
    EC=$?
    if [ $EC -eq 0 ]; then
        # Pull pass-rate out of the saved JSON for an inline summary.
        SUMMARY=$(python -c "
import json
d = json.load(open('${OUT}'))
s = d['summary']
k = d['by_kind']
print(f\"  pass_rate={s['pass_rate']:.1%}  ({s['passed_tasks']}/{s['total_tasks']})  code={k['code']['passed']}/{k['code']['total']}  mcq={k['mcq']['passed']}/{k['mcq']['total']}\")
")
        echo "${SUMMARY}"
    else
        echo "  exited with code ${EC} — check ${LOG}"
    fi
    echo
done

echo "$(date '+%Y-%m-%d %H:%M:%S')  bundle study complete"
echo "Results: results/*.json"
echo "Logs:    results/*.log"
