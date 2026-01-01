#!/bin/bash

# Run varying target accuracy experiments
# Tests: 0.75, 0.8, 0.85, 0.9, 0.95

set -e

echo "=== Task Cascades: Varying Target Accuracy ==="
echo "Targets: 0.75, 0.8, 0.85, 0.9, 0.95"

TASKS=("game_review" "legal_doc" "ag_news" "court_opinion")
SAMPLE_SIZE=1000

for task in "${TASKS[@]}"; do
    echo ""
    echo "Running: $task"

    python task_cascades/experiments/full_experiments.py \
        --task "$task" \
        --sample_size $SAMPLE_SIZE \
        --seed 42 \
        --varying_target

    echo "Done: $task"
done

echo ""
echo "=== All varying target experiments completed ==="
echo "Results in: results/varying_target/"
