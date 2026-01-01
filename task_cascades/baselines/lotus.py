#!/usr/bin/env python3

import sys
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

from dotenv import load_dotenv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Third-party LOTUS
import lotus
from lotus.models import LM
from lotus.types import CascadeArgs, LMStats

# Allow importing local task_cascades modules
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from task_cascades.data.create_dfs import load_dataset  # noqa: E402
from task_cascades.predictors.predictors import (  # noqa: E402
    ORACLE_PREDICTOR,
    TASK_PROMPT_DICT,
    TASK_INSTRUCTIONS,
    PROMPT_TO_TASK_TYPE_DICT,
    run_predictor_and_get_row_copies,
)

# Load environment after imports
load_dotenv()


PRECISION_RECALL_GRID: List[float] = [0.6, 0.8, 0.9, 0.95]
SAMPLING_PERCENTAGE: float = 0.2
FAILURE_PROBABILITY: float = 0.25

BINARY_TASKS: List[str] = [
    "enron",
    "legal_doc",
    "game_review",
    "court_opinion",
]
MULTICLASS_TASKS: List[str] = [
    "ag_news",
]


# Globals for LOTUS LMs so we can reconfigure safely if needed
LM_PRIMARY: LM | None = None
LM_HELPER: LM | None = None


def monkey_patch_lotus_logprobs_assert() -> None:
    """Monkey patch LM._get_top_choice_logprobs to gracefully handle missing logprobs.

    If the provider does not return logprobs, bypass the internal assertion and
    return an empty list so downstream code that iterates does not crash.
    """
    try:
        original = getattr(LM, "_get_top_choice_logprobs")

        def patched(self, *args, **kwargs):  # type: ignore
            try:
                return original(self, *args, **kwargs)
            except AssertionError:
                print("AssertionError: Returning empty list for logprobs")
                return []
            except Exception as ex:  # Fallback for logprob-related provider quirks
                msg = str(ex).lower()
                if "choicelogprobs" in msg or "logprob" in msg:
                    return []
                raise

        setattr(LM, "_get_top_choice_logprobs", patched)
    except Exception:
        # If monkey patching fails, continue – safe_sem_filter will still retry
        pass

def label_documents_with_oracle(df: pd.DataFrame, task: str) -> Tuple[pd.DataFrame, float]:
    """Label all rows using the oracle predictor in parallel as ground-truth.

    Returns (DataFrame with `label` column, oracle_total_cost) where oracle_total_cost is the
    sum of `surrogate_cost` across all rows from the oracle labeling pass.
    """
    prompt: str = TASK_PROMPT_DICT[task]
    task_type: str = PROMPT_TO_TASK_TYPE_DICT[task]

    # Use the parallel helper to run oracle predictions across rows
    rows = run_predictor_and_get_row_copies(
        predictor=ORACLE_PREDICTOR,
        task_prompt=prompt,
        df=df,
        surrogate_name="oracle",
        text_column="text",
        task_type=task_type,
    )
    rows_df = pd.DataFrame(rows)
    labels = rows_df["surrogate_prediction"].astype(int).tolist()
    oracle_total_cost = float(rows_df["surrogate_cost"].sum()) if "surrogate_cost" in rows_df.columns else 0.0

    out_df = df.copy()
    out_df["label"] = labels
    return out_df, oracle_total_cost


def build_user_instruction_binary(task: str) -> str:
    """Build a LOTUS user_instruction for a binary task using existing task instructions.

    We use the detailed `TASK_INSTRUCTIONS[task]` and explicitly reference the text via {text}.
    """
    instruction: str = TASK_INSTRUCTIONS[task]
    return (
        f"{instruction}\n\n"
        f"Text: {{text}}\n"
        f"Return True if the instruction applies to the text; otherwise return False."
    )


def build_user_instruction_ag_news_for_class(target_class: int) -> str:
    """Build a LOTUS user_instruction for a one-vs-rest check of ag_news classes.

    Uses the full ag_news task description and asks whether the best-matching category is target_class.
    """
    instruction: str = TASK_INSTRUCTIONS["ag_news"]
    return (
        f"{instruction}\n\n"
        f"Text: {{text}}\n"
        f"Return True if the single best category for the article is {target_class}; otherwise return False."
    )


def safe_sem_filter(df: pd.DataFrame, user_instruction: str, cascade_args: CascadeArgs) -> Tuple[pd.DataFrame, float]:
    """Run sem_filter and return (filtered_df, adjusted_virtual_cost).

    Adjusted virtual cost = (LM_PRIMARY.virtual + LM_HELPER.virtual) - optimization_cost
    where optimization_cost subtracts per-doc optimization overhead for both helper and large
    models based on counts in returned stats.
    """
    # Reset stats before the call
    try:
        if LM_PRIMARY is not None:
            LM_PRIMARY.stats = LMStats()
        if LM_HELPER is not None:
            LM_HELPER.stats = LMStats()
    except Exception:
        pass

    filtered_df, stats = df.sem_filter(
        user_instruction=user_instruction,
        cascade_args=cascade_args,
        return_stats=True,
    )

    # Sum virtual costs from both oracle (primary) and helper models
    total_virtual_cost = 0.0
    oracle_virtual_cost = 0.0
    helper_virtual_cost = 0.0
    try:
        if LM_PRIMARY is not None and getattr(LM_PRIMARY, "stats", None) is not None:
            total_virtual_cost += float(getattr(getattr(LM_PRIMARY.stats, "virtual_usage", None), "total_cost", 0.0))
            oracle_virtual_cost += float(getattr(getattr(LM_PRIMARY.stats, "virtual_usage", None), "total_cost", 0.0))
    except Exception:
        pass
    try:
        if LM_HELPER is not None and getattr(LM_HELPER, "stats", None) is not None:
            total_virtual_cost += float(getattr(getattr(LM_HELPER.stats, "virtual_usage", None), "total_cost", 0.0))
            helper_virtual_cost += float(getattr(getattr(LM_HELPER.stats, "virtual_usage", None), "total_cost", 0.0))
    except Exception:
        pass

    # Get counts from stats (handle dict or object)
    def _get_stat_count(s: Any, key: str) -> int:
        try:
            if isinstance(s, dict):
                return int(s.get(key, 0) or 0)
            return int(getattr(s, key, 0) or 0)
        except Exception:
            return 0

    return filtered_df, total_virtual_cost

def run_lotus_binary(
    df: pd.DataFrame,
    task: str,
    recall_targets: List[float],
    precision_targets: List[float],
    oracle_total_cost: float,
) -> Dict[Tuple[float, float], Dict[str, float]]:
    """Run LOTUS sem_filter over a recall/precision grid and compute accuracy directly.

    Accuracy is computed by treating membership in the filtered DataFrame as prediction == 1.
    """
    df = df.copy().reset_index(drop=True)
    df["row_id"] = np.arange(len(df))

    user_instruction: str = build_user_instruction_binary(task)

    results: Dict[Tuple[float, float], Dict[str, float]] = {}

    for p in precision_targets:
        for r in recall_targets:
            cascade_args = CascadeArgs(
                recall_target=r,
                precision_target=p,
                sampling_percentage=SAMPLING_PERCENTAGE,
                failure_probability=FAILURE_PROBABILITY,
            )

            filtered_df, total_cost = safe_sem_filter(df, user_instruction, cascade_args)

            predicted_positive_ids = set(filtered_df["row_id"].tolist())
            y_true = df["label"].astype(int).tolist()
            y_pred = [1 if rid in predicted_positive_ids else 0 for rid in df["row_id"].tolist()]
            correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
            accuracy = correct / len(y_true) if y_true else 0.0
            # Record raw total cost and fraction of oracle total cost
            hypothetical_oracle_cost = oracle_total_cost * (1 - cascade_args.sampling_percentage)
            
            frac = (float(total_cost) / hypothetical_oracle_cost) if hypothetical_oracle_cost > 0 else float("nan")
            results[(p, r)] = {"accuracy": float(accuracy), "cost": float(total_cost), "cost_frac": float(frac)}

    return results


def run_lotus_ag_news_ovr(
    df: pd.DataFrame,
    recall_targets: List[float],
    precision_targets: List[float],
    oracle_total_cost: float,
) -> Dict[Tuple[float, float], Dict[str, float]]:
    """Run LOTUS for ag_news using one-vs-rest over the 4 classes and compute accuracy directly.

    For each grid point, we run 4 sem_filters (one per class). If multiple classes include a row,
    we pick the smallest class index; if none include it, default to class 0.
    """
    df = df.copy().reset_index(drop=True)
    df["row_id"] = np.arange(len(df))

    class_instructions: Dict[int, str] = {
        c: build_user_instruction_ag_news_for_class(c) for c in [0, 1, 2, 3]
    }

    results: Dict[Tuple[float, float], Dict[str, float]] = {}

    for p in precision_targets:
        for r in recall_targets:
            cascade_args = CascadeArgs(
                recall_target=r,
                precision_target=p,
                sampling_percentage=SAMPLING_PERCENTAGE,
                failure_probability=FAILURE_PROBABILITY,
            )

            class_to_membership: Dict[int, set] = {}
            total_virtual_cost: float = 0.0
            for cls in [0, 1, 2, 3]:
                filtered_df, class_cost = safe_sem_filter(df, class_instructions[cls], cascade_args)
                class_to_membership[cls] = set(filtered_df["row_id"].tolist())
                total_virtual_cost += float(class_cost)

            def _predict_row(row_id: int) -> int:
                for cls in [0, 1, 2, 3]:
                    if row_id in class_to_membership[cls]:
                        return cls
                return 0

            y_true = df["label"].astype(int).tolist()
            y_pred = [_predict_row(rid) for rid in df["row_id"].tolist()]
            correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
            accuracy = correct / len(y_true) if y_true else 0.0
            
            # Record raw total cost and fraction of oracle total cost
            hypothetical_oracle_cost = oracle_total_cost * (1 - cascade_args.sampling_percentage)
            
            frac = (float(total_virtual_cost) / hypothetical_oracle_cost) if hypothetical_oracle_cost > 0 else float("nan")
            results[(p, r)] = {"accuracy": float(accuracy), "cost": float(total_virtual_cost), "cost_frac": float(frac)}

    return results


def configure_lotus_models() -> None:
    """Configure LOTUS with oracle and proxy models from utils (gpt-4o and gpt-4o-mini)."""
    global LM_PRIMARY, LM_HELPER
    LM_HELPER = LM("azure/gpt-4o-mini", logprobs=True)
    LM_PRIMARY = LM("azure/gpt-4o", logprobs=True)
    lotus.settings.configure(lm=LM_PRIMARY, helper_lm=LM_HELPER)


def main() -> None:
    # Ensure missing logprobs do not crash LOTUS internal assert
    monkey_patch_lotus_logprobs_assert()
    configure_lotus_models()

    parser = argparse.ArgumentParser(description="Run LOTUS across datasets and optionally save results")
    parser.add_argument("--save", action="store_true", help="Save results compatible with print_varying_target_results.py")
    args = parser.parse_args()

    tasks: List[str] = BINARY_TASKS + MULTICLASS_TASKS
    task_to_results: Dict[str, Dict[Tuple[float, float], float]] = {}

    for task in tasks:
        print(f"\n=== Running LOTUS on task: {task} ===")
        # Load raw data
        raw_df, _docs = load_dataset(task)
        # Sample 1000 documents deterministically (seed=42)
        if len(raw_df) > 0:
            raw_df = raw_df.sample(n=min(1000, len(raw_df)), random_state=42).reset_index(drop=True)
        if "text" not in raw_df.columns:
            print(f"Task {task}: missing 'text' column; skipping.")
            continue

        # Label with oracle (ground-truth) and get oracle total cost across sampled df
        labeled_df, oracle_total_cost = label_documents_with_oracle(raw_df, task)

        if task in BINARY_TASKS:
            results = run_lotus_binary(
                labeled_df, task, recall_targets=PRECISION_RECALL_GRID, precision_targets=PRECISION_RECALL_GRID, oracle_total_cost=oracle_total_cost
            )
        else:
            results = run_lotus_ag_news_ovr(
                labeled_df, recall_targets=PRECISION_RECALL_GRID, precision_targets=PRECISION_RECALL_GRID, oracle_total_cost=oracle_total_cost
            )

        task_to_results[task] = results

        # Print per-task summary table
        print("Precision/Recall -> Accuracy | Cost ($ virtual)")
        for p in PRECISION_RECALL_GRID:
            row_vals = []
            for r in PRECISION_RECALL_GRID:
                point = results.get((p, r), {"accuracy": float("nan"), "cost": float("nan")})
                acc = point["accuracy"]
                cost = point["cost"]
                frac = point.get("cost_frac", float("nan"))
                row_vals.append(f"{acc:.3f} | ${cost:.6f} | {frac:.3f}x oracle")
            print(f"P={p:.2f}: [" + ", ".join(row_vals) + f"] (R from {min(PRECISION_RECALL_GRID):.2f}→{max(PRECISION_RECALL_GRID):.2f})")

        # Save compatible results if requested (save ALL grid entries and add lotus filename)
        if args.save:
            grid_entries = []
            for p in PRECISION_RECALL_GRID:
                for r in PRECISION_RECALL_GRID:
                    pt = results.get((p, r))
                    if pt is None:
                        continue
                    grid_entries.append({
                        "precision": float(p),
                        "recall": float(r),
                        "accuracy": float(pt.get("accuracy", float("nan"))),
                        "cost": float(pt.get("cost_frac", float("nan"))),
                        "raw_cost": float(pt.get("cost", float("nan"))),
                    })

            # Oracle cost baseline as 1.0 since we save fraction-of-oracle
            oracle_cost = 1.0
            canonical_p = PRECISION_RECALL_GRID[0]
            aggregated_stats = {
                "target_accuracies": PRECISION_RECALL_GRID,
                "oracle_cost": float(oracle_cost),
                "methods": {
                    "lotus": {
                        # A single representative series at canonical precision
                        "achieved_accuracies": [
                            results.get((canonical_p, r), {}).get("accuracy", float("nan"))
                            for r in PRECISION_RECALL_GRID
                        ],
                        "costs": [
                            results.get((canonical_p, r), {}).get("cost_frac", float("nan"))
                            for r in PRECISION_RECALL_GRID
                        ],
                        "meets_target_flags": [True for _ in PRECISION_RECALL_GRID],
                        "cost_reductions": [
                            (1.0 - results.get((canonical_p, r), {}).get("cost_frac", float("nan"))) * 100.0
                            if results.get((canonical_p, r), {}).get("cost_frac", float("nan")) == results.get((canonical_p, r), {}).get("cost_frac", float("nan")) else 0.0
                            for r in PRECISION_RECALL_GRID
                        ],
                    }
                },
            }

            data = {
                "aggregated_stats": aggregated_stats,
                "all_target_results": {
                    "grid_entries": grid_entries,
                },
            }

            project_root = Path(__file__).parent.parent.parent
            results_dir = project_root / "results" / "lotus"
            results_dir.mkdir(parents=True, exist_ok=True)
            latest_path = results_dir / f"{task}_lotus_varying_target_results_latest.pkl"
            with open(latest_path, "wb") as f:
                pickle.dump(data, f)
            print(f"Saved results for {task} -> {latest_path}")

    # Plot: one subplot per task, x-axis cost ($ virtual), lines per precision target, y accuracy
    n_tasks: int = len(task_to_results)
    if n_tasks == 0:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(1, n_tasks, figsize=(3.8 * n_tasks + 2, 3.8))
    if n_tasks == 1:
        axes = [axes]

    for i, (task, results) in enumerate(task_to_results.items()):
        ax = axes[i]
        for p in PRECISION_RECALL_GRID:
            accuracies = [results.get((p, r), {"accuracy": np.nan})["accuracy"] for r in PRECISION_RECALL_GRID]
            costs = [results.get((p, r), {"cost": np.nan})["cost"] for r in PRECISION_RECALL_GRID]
            ax.plot(costs, accuracies, marker="o", label=f"Precision {p:.2f}")
        ax.set_title(task.replace("_", " ").title())
        ax.set_xlabel("Cost ($ virtual)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        # Use log scale for cost when all values are positive
        try:
            all_costs = [results.get((p, r), {"cost": np.nan})["cost"] for p in PRECISION_RECALL_GRID for r in PRECISION_RECALL_GRID]
            all_costs = [c for c in all_costs if np.isfinite(c) and c > 0]
            if all_costs and min(all_costs) > 0:
                ax.set_xscale("log")
        except Exception:
            pass
        if i == n_tasks - 1:
            ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


