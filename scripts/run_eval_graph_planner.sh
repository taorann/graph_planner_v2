#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CONFIG="${ROOT_DIR}/configs/eval/graph_planner_eval_defaults.yaml"

CONFIG_FLAG_PRESENT=0
for arg in "$@"; do
  if [[ "$arg" == --config || "$arg" == --config=* ]]; then
    CONFIG_FLAG_PRESENT=1
    break
  fi
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    CONFIG_FLAG_PRESENT=1
    break
  fi
done

if [[ $CONFIG_FLAG_PRESENT -eq 0 ]]; then
  set -- --config "${GRAPH_PLANNER_EVAL_CONFIG:-$DEFAULT_CONFIG}" "$@"
fi

PYTHONPATH="${PYTHONPATH:-${ROOT_DIR}}" \
TOKENIZERS_PARALLELISM="false" \
python "${ROOT_DIR}/scripts/eval_graph_planner_engine.py" "$@"
