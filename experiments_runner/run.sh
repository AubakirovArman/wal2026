#!/bin/bash
# WAL experiment runner - uses venv Python, fixes device to GPU 2/3
set -e
VENV=/mnt/hf_model_weights/arman/3bit/wal/.venv/bin/python
RUNNER_DIR=/mnt/hf_model_weights/arman/3bit/wal/experiments_runner
DIARY=$RUNNER_DIR/DIARY.md

run_exp() {
    local script=$1
    local gpu=${2:-3}
    local name=$(basename "$script" .py)
    echo "=== $name (GPU $gpu) ==="

    # Run and capture output
    CUDA_VISIBLE_DEVICES=$gpu $VENV "$script" 2>&1 | tail -20
    local rc=${PIPESTATUS[0]}

    # Log to diary
    echo "" >> $DIARY
    echo "## $name — $(date +%H:%M)" >> $DIARY
    echo "- Status: $([ $rc -eq 0 ] && echo PASS || echo FAIL)" >> $DIARY
    echo "- GPU: $gpu" >> $DIARY

    return $rc
}

# Run list of experiments
for exp in "$@"; do
    run_exp "$exp" || echo "FAILED: $exp" >> $DIARY
done
