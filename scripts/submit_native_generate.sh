#!/bin/bash
# An agent writes a native C engine for a model on a Xeon node and makes its hot paths fast
# (knowledge_base/native); the tools verify it on its surrogate and on the model.
#
# Usage:
#   MODEL=Qwen/Qwen3-1.7B OUT=../native-runs/qwen3 sbatch scripts/submit_native_generate.sh
#   MODEL=<dir> OUT=... REFERENCE_SOURCE=<the model's Python, when transformers cannot run it> \
#     PREPARE="<command producing MODEL>" sbatch scripts/submit_native_generate.sh

#SBATCH --job-name=native_generate
#SBATCH --partition=emr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=23:59:00
#SBATCH --output=slurm_native_generate_%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
MODEL="${MODEL:?set MODEL to a hub id or local directory}"
OUT="${OUT:?set OUT to the run directory}"
FRAMEWORK_PYTHON="${FRAMEWORK_PYTHON:?set FRAMEWORK_PYTHON to the interpreter that runs the model}"

for f in "$REPO_DIR/.env" "$FRAMEWORK_PYTHON"; do
    [[ -e "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done
source "$REPO_DIR/.venv/bin/activate"
export PATH="$HOME/.local/bin:$PATH"
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
for tool in claude cc cmake numactl; do command -v "$tool" >/dev/null || { echo "no $tool" >&2; exit 1; }; done

echo "native-generate on $(hostname): $(lscpu | sed -n 's/^Model name: *//p') -- $MODEL -> $OUT"
cd "$REPO_DIR"
if [[ -n "${PREPARE:-}" ]]; then
    echo "prepare: $PREPARE"
    bash -c "$PREPARE"
fi
EXTRA=()
[[ -n "${REFERENCE_SOURCE:-}" ]] && EXTRA+=(--reference-source "$REFERENCE_SOURCE")
python -m xe_forge.cli generate --model "$MODEL" -o "$OUT" --framework-python "$FRAMEWORK_PYTHON" \
    --threads "${THREADS:-64}" --numa 0 --agent-max-turns "${AGENT_MAX_TURNS:-500}" "${EXTRA[@]}"
