#!/bin/bash

# Run full experiments for all tasks
# Compares Task Cascades against variants and baselines

tasks=("ag_news" "fever" "court_opinion" "game_review" "legal_doc" "pubmed")

SAMPLE_SIZE=1000
NUM_RUNS=1
MAX_PARALLEL_JOBS=3
METHODS_CONFIG="task_cascades/config/methods_config.yaml"

echo "=== Task Cascades: Full Experiments ==="
echo "Tasks: ${tasks[@]}"
echo "Sample size: $SAMPLE_SIZE"

run_experiment() {
    local task=$1
    local seed=$2

    echo "Running: $task (seed=$seed)..."
    python task_cascades/experiments/full_experiments.py \
        --task="$task" \
        --sample_size="$SAMPLE_SIZE" \
        --seed="$seed" \
        --methods_config="$METHODS_CONFIG" \
        --rerun

    if [ $? -eq 0 ]; then
        echo "Done: $task"
    else
        echo "FAILED: $task"
        return 1
    fi
}

wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
        sleep 1
    done
}

for task in "${tasks[@]}"; do
    for run in $(seq 1 $NUM_RUNS); do
        seed=$((41 + run))
        wait_for_jobs
        run_experiment "$task" "$seed" &
        sleep 0.5
    done
done

wait

echo ""
echo "=== All experiments completed ==="
echo "Results in: results/"
