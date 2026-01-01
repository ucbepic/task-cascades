#!/bin/bash

# Run repeated trials experiments (10 seeds per task)
# Tests consistency of Task Cascades

set -e

echo "=== Task Cascades: Repeated Trials ==="

TASKS=("game_review" "legal_doc" "ag_news" "court_opinion" "enron")
SAMPLE_SIZE=1000
NUM_TRIALS=10

for task in "${TASKS[@]}"; do
    echo ""
    echo "Running: $task ($NUM_TRIALS trials)"

    python task_cascades/experiments/run_experiments.py \
        --task=$task \
        --sample_size=$SAMPLE_SIZE \
        --seed=42 \
        --num_trials=$NUM_TRIALS \
        --repeated_trials

    echo "Done: $task"
done

echo ""
echo "=== All repeated trials completed ==="
echo "Results in: results/repeated_trials/"
