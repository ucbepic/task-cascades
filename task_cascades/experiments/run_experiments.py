#!/usr/bin/env python3
"""Main experiment runner for Task Cascades.

Usage:
    python -m task_cascades.experiments.run_experiments --task game_review
    python -m task_cascades.experiments.run_experiments --task legal_doc --methods_config config/methods_config.yaml
"""

import argparse
import os
import sys
import yaml
from typing import List, Dict, Any

from rich.console import Console
from rich.table import Table

from task_cascades.config.config import ExperimentConfig, MethodConfig
from task_cascades.experiments.experiment_runner import ExperimentRunner

console = Console()
config = ExperimentConfig()
method_config = MethodConfig()


def print_results_table(results: Dict[str, Any], oracle_cost: float, target_accuracy: float):
    """Print a formatted results table."""
    table = Table(title=f"Results (Target Accuracy: {target_accuracy})")
    table.add_column("Method", style="bold")
    table.add_column("Accuracy", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Cost Reduction", justify="right")

    for method, result in sorted(results.items()):
        if isinstance(result, dict) and "overall_accuracy" in result:
            acc = result["overall_accuracy"]
            cost = result["total_cost"]
            reduction = (1 - cost / oracle_cost) * 100 if oracle_cost > 0 else 0

            style = method_config.METHOD_STYLES.get(method, "")
            table.add_row(
                style if style else method,
                f"{acc:.4f}",
                f"{cost:.4f}",
                f"{reduction:.1f}%"
            )
        elif isinstance(result, dict) and "error" in result:
            table.add_row(method, "[red]ERROR[/red]", "-", "-")

    console.print(table)


def load_methods_config(config_path: str) -> List[str]:
    """Load enabled methods from YAML config."""
    if not config_path or not os.path.exists(config_path):
        return config.DEFAULT_METHODS

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    methods = [m for m, enabled in cfg.get("methods", {}).items() if enabled]
    return methods if methods else config.DEFAULT_METHODS


def run_single_experiment(
    task: str,
    methods: List[str],
    target_accuracy: float,
    sample_size: int = 1000,
    train_split: float = 0.2,
    seed: int = 42,
    skip_cache: bool = False
) -> Dict[str, Any]:
    """Run a single experiment with specified parameters."""

    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]Task: {task.upper()} | Target: {target_accuracy} | Seed: {seed}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

    # Create runner and prepare data
    runner = ExperimentRunner(
        task=task,
        sample_size=sample_size,
        train_split=train_split,
        seed=seed,
        skip_cache=skip_cache
    )
    runner.prepare()

    # Run all methods
    results = runner.run_all(methods, target_accuracy)

    # Print results
    print_results_table(results, runner.oracle_cost, target_accuracy)

    # Save results
    results_path = runner.save_results(results, target_accuracy)
    console.print(f"\n[bold blue]Results saved to: {results_path}[/bold blue]")

    return results


def run_repeated_trials(
    task: str,
    methods: List[str],
    target_accuracy: float = 0.9,
    num_trials: int = 3,
    sample_size: int = 1000,
    train_split: float = 0.2,
    base_seed: int = 42
) -> Dict[str, Any]:
    """Run repeated trials with different seeds."""

    console.print(f"\n[bold cyan]REPEATED TRIALS: {task.upper()}[/bold cyan]")
    console.print(f"Trials: {num_trials} | Target: {target_accuracy}\n")

    all_results = []

    for trial in range(1, num_trials + 1):
        seed = base_seed + trial
        console.print(f"\n[bold yellow]Trial {trial}/{num_trials} (seed={seed})[/bold yellow]")

        runner = ExperimentRunner(
            task=task,
            sample_size=sample_size,
            train_split=train_split,
            seed=seed
        )
        runner.prepare()
        results = runner.run_all(methods, target_accuracy)
        results["_meta"] = {"trial": trial, "seed": seed}
        all_results.append(results)

    # Aggregate statistics
    console.print(f"\n[bold green]AGGREGATE RESULTS[/bold green]")
    _print_aggregate_stats(all_results, methods)

    return {"trials": all_results}


def run_varying_target(
    task: str,
    methods: List[str],
    target_accuracies: List[float] = None,
    sample_size: int = 1000,
    train_split: float = 0.2,
    seed: int = 42
) -> Dict[str, Any]:
    """Run experiments across varying target accuracies."""

    if target_accuracies is None:
        target_accuracies = [0.75, 0.8, 0.85, 0.9, 0.95]

    console.print(f"\n[bold cyan]VARYING TARGET: {task.upper()}[/bold cyan]")
    console.print(f"Targets: {target_accuracies}\n")

    # Prepare data once
    runner = ExperimentRunner(
        task=task,
        sample_size=sample_size,
        train_split=train_split,
        seed=seed
    )
    runner.prepare()

    all_results = {}
    for target in target_accuracies:
        console.print(f"\n[bold yellow]Target Accuracy: {target}[/bold yellow]")
        results = runner.run_all(methods, target)
        all_results[target] = results
        print_results_table(results, runner.oracle_cost, target)

    return all_results


def _print_aggregate_stats(all_results: List[Dict], methods: List[str]):
    """Print aggregate statistics across trials."""
    import numpy as np

    table = Table(title="Aggregate Statistics")
    table.add_column("Method", style="bold")
    table.add_column("Accuracy (mean±std)", justify="center")
    table.add_column("Cost (mean±std)", justify="center")

    for method in methods:
        accs = []
        costs = []
        for trial_result in all_results:
            if method in trial_result and "overall_accuracy" in trial_result[method]:
                accs.append(trial_result[method]["overall_accuracy"])
                costs.append(trial_result[method]["total_cost"])

        if accs:
            table.add_row(
                method,
                f"{np.mean(accs):.3f} ± {np.std(accs):.3f}",
                f"{np.mean(costs):.3f} ± {np.std(costs):.3f}"
            )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Run Task Cascades experiments")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "game_review", "legal_doc", "enron", "wiki_talk",
            "court_opinion", "screenplay", "sms_spam", "fever",
            "ag_news", "biodex", "pubmed"
        ],
        help="Task to run"
    )
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--train_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_accuracy", type=float, default=0.9)
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument(
        "--methods_config",
        type=str,
        default=None,
        help="Path to methods config YAML"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Specific methods to run (overrides config)"
    )

    # Experiment modes
    parser.add_argument("--repeated_trials", action="store_true", help="Run repeated trials")
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--varying_target", action="store_true", help="Run varying target accuracy")

    args = parser.parse_args()

    # Determine methods to run
    if args.methods:
        methods = args.methods
    else:
        methods = load_methods_config(args.methods_config)

    console.print(f"[bold]Methods to run:[/bold] {methods}")

    # Run appropriate experiment mode
    if args.repeated_trials:
        run_repeated_trials(
            task=args.task,
            methods=methods,
            target_accuracy=args.target_accuracy,
            num_trials=args.num_trials,
            sample_size=args.sample_size,
            train_split=args.train_split,
            base_seed=args.seed
        )
    elif args.varying_target:
        run_varying_target(
            task=args.task,
            methods=methods,
            sample_size=args.sample_size,
            train_split=args.train_split,
            seed=args.seed
        )
    else:
        run_single_experiment(
            task=args.task,
            methods=methods,
            target_accuracy=args.target_accuracy,
            sample_size=args.sample_size,
            train_split=args.train_split,
            seed=args.seed,
            skip_cache=args.skip_cache
        )

    console.print("\n[bold green]✓ Experiment complete![/bold green]")


if __name__ == "__main__":
    main()
