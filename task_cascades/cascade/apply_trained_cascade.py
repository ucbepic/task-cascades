import os
import pickle
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from task_cascades.cascade.cascade_utils import (
    simulate_cascade,
    find_thresholds_for_surrogate,
    _build_shifted_thresholds,
)
from task_cascades.stats.bargain_utils import test_if_true_mean_is_above_m
from task_cascades.predictors.predictors import ORACLE_PREDICTOR, run_predictor_and_get_row_copies, PROMPT_TO_CLASSES_DICT


def apply_cascade(
    test_df: pd.DataFrame,
    ordering: List[Tuple[str, str, float]],
    surrogate_to_prompt: Dict[str, str],
    thresholds: Dict[Tuple[str, str, float], Tuple[float, float]],
    task_type: str
) -> Dict[str, Any]:
    """
    Apply a trained cascade to the test set.
    
    Args:
        test_df: DataFrame containing test documents
        ordering: The cascade ordering from find_surrogates
        surrogate_to_prompt: Mapping from surrogate name to prompt
        thresholds: Dict mapping candidate tuple to (thresh_pos, thresh_neg) thresholds
        
    Returns:
        Dictionary with cascade results including accuracy, cost, and stage usage
    """
    print("Running model predictions for surrogate cascade evaluation...")
    all_executions = []
    
    # Keep track of which surrogate/model combinations we've already processed
    processed_combinations = set()
    
    # Process each candidate in the ordering
    for surrogate_name, predictor_model, doc_fraction in ordering:
        # Get the prompt for this surrogate
        task_prompt = surrogate_to_prompt.get(surrogate_name)
        if not task_prompt:
            print(f"Warning: No prompt found for surrogate {surrogate_name}")
            continue
            
        # Skip if we've already processed this surrogate_name/predictor_model combination
        if (surrogate_name, predictor_model, doc_fraction) in processed_combinations:
            continue
            
        # Add to processed combinations
        processed_combinations.add((surrogate_name, predictor_model, doc_fraction))
        
        # Get subset of test_df for this doc_fraction
        test_df_subset = test_df[test_df["fraction"] == doc_fraction].reset_index(drop=True)
        
        # Run predictions for this surrogate/model combination
        rows = run_predictor_and_get_row_copies(
            predictor_model, task_prompt, test_df_subset, surrogate_name, task_type=task_type
        )
        assert len(rows) == len(test_df_subset), f"Number of rows ({len(rows)}) does not match number of rows in test df subset ({len(test_df_subset)})"
        all_executions.extend(rows)
    
    # Convert to DataFrame
    all_executions_df = pd.DataFrame(all_executions)
    
    # See which uuids are missing from all_executions_df and run the predictor (oracle) on them
    uuids_missing = test_df[~test_df["uuid"].isin(all_executions_df["uuid"])]["uuid"].unique()
    if len(uuids_missing) > 0:
        missing_df = test_df[test_df["uuid"].isin(uuids_missing)]
        rows = run_predictor_and_get_row_copies(
            ORACLE_PREDICTOR, surrogate_to_prompt.get("s1"), missing_df, surrogate_name, task_type=task_type
        )
        all_executions.extend(rows)
    
    # Convert to DataFrame
    all_executions_df = pd.DataFrame(all_executions)
    
    # Simulate the cascade
    num_predictions_in_test_df = test_df["uuid"].nunique()
    overall_accuracy, total_cost, stage_usage, predictions, _ = simulate_cascade(
        ordering, thresholds, all_executions_df
    )
    num_predictions_post_cascade = len(predictions)
    
    assert num_predictions_post_cascade == num_predictions_in_test_df, f"Number of predictions post cascade ({num_predictions_post_cascade}) does not match number of predictions in test df ({num_predictions_in_test_df})"
    
    return {
        "overall_accuracy": overall_accuracy,
        "total_cost": total_cost,
        "stage_usage": stage_usage
    }
 
def train_and_apply_baseline_cascade(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    accuracy_target: float,
    task: str,
    max_shift: int = 10,
    delta: float = 0.25,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train a two-stage *proxy → oracle* baseline cascade and evaluate it.

    Returns two results:
      1. *No guarantee* – thresholds chosen on the training split only.
      2. *With guarantee* – thresholds are first fit on **half** of the training
         data, then globally shifted (+0‥+``max_shift``) until a Bargain
         hypothesis test on the held-out *dev* split would fail.  The largest
         shift that still passes is selected, and the resulting thresholds are
         finally evaluated on the test split.
    """

    # Deduplicate by document
    train_df_unique = train_df.copy().drop_duplicates(subset="uuid")
    test_df_unique = test_df.copy().drop_duplicates(subset="uuid")

    # ------------------------------------------------------------------
    # Train / Dev split for the *guarantee* path (50 / 50)
    # ------------------------------------------------------------------
    dev_df = train_df_unique.sample(frac=0.5, random_state=42)
    train_df_split = train_df_unique.drop(dev_df.index)

    # ------------------------------------------------------------------
    # 1) Fit thresholds on the *full* train set for the no-guarantee path
    # ------------------------------------------------------------------
    class_thresholds, _class_sequences_unused = _compute_thresholds_for_accuracy(
        train_df_unique, accuracy_target, task
    )
    
    # Helper to apply proxy-oracle cascade with arbitrary thresholds
    def _apply_baseline(
        df: pd.DataFrame, thresholds: Dict[Any, float]
    ) -> Tuple[float, float, Dict[str, float], List[Any]]:
        predictions: List[Any] = []
        total_cost = 0.0
        proxy_ct = 0

        for _, row in df.iterrows():
            total_cost += row["baseline_cost"]

            pred_class = row["baseline_prediction"]
            conf = row["baseline_confidence"]
            thr = thresholds.get(pred_class, float("inf"))

            if conf >= thr:
                # keep proxy prediction
                predictions.append(pred_class)
                proxy_ct += 1
            else:
                # fall back to oracle (always correct)
                predictions.append(row["label"])
                total_cost += row["oracle_cost"]

        labels = df["label"].tolist()
        correct_preds = sum(p == l for p, l in zip(predictions, labels))
        accuracy = correct_preds / len(predictions) if predictions else 0.0
        stage_usage = {
            "proxy": proxy_ct / len(predictions) if predictions else 0.0,
            "oracle": 1.0 - (proxy_ct / len(predictions) if predictions else 0.0),
        }

        return accuracy, total_cost, stage_usage, predictions

    # ------------------------------------------------------------
    # 2) Evaluate on test set with *no guarantee*
    # ------------------------------------------------------------
    acc_ng, cost_ng, usage_ng, _preds_ng = _apply_baseline(test_df_unique, class_thresholds)
    no_guarantee_result = {
        "overall_accuracy": acc_ng,
        "total_cost": cost_ng,
        "stage_usage": usage_ng,
    }

    # ------------------------------------------------------------
    # 3) Fit thresholds on TRAIN split and find global shift on DEV
    # ------------------------------------------------------------
    _g_class_thresholds, g_class_sequences = _compute_thresholds_for_accuracy(
        train_df_split, accuracy_target, task
    )

    best_shift = 0
    # Iterate from strictest (max_shift) downwards
    for shift in range(max_shift, -1, -1):
        shifted_thr = _build_shifted_thresholds(g_class_sequences, shift)
        acc, _cost, _usage, preds = _apply_baseline(dev_df, shifted_thr)

        acc_indicators = (
            (np.array(preds) == np.array(dev_df["label"].tolist())).astype(int)
        )
        passed = test_if_true_mean_is_above_m(acc_indicators, accuracy_target, delta)

        if passed:
            best_shift = shift
            # continue to try stricter shift
            continue
        else:
            # first failure encountered; previous shift was last valid
            break

    guaranteed_thresholds = _build_shifted_thresholds(g_class_sequences, best_shift)
    acc_g, cost_g, usage_g, _preds_g = _apply_baseline(test_df_unique, guaranteed_thresholds)

    guarantee_result = {
        "overall_accuracy": acc_g,
        "total_cost": cost_g,
        "stage_usage": usage_g,
        "meta_shift": best_shift,
    }

    return no_guarantee_result, guarantee_result

def _compute_thresholds_for_accuracy(
    train_df: pd.DataFrame, accuracy_target: float, task: str
) -> Tuple[Dict[int, float], Dict[int, List[float]]]:
    """
    Determine a confidence threshold for each class independently so that, when
    applying a simple two-stage cascade (proxy → oracle), the **per-class** accuracy
    on the training set meets or exceeds `accuracy_target`.

    For a document of class *c* we keep the proxy prediction if its confidence is
    above the threshold for *c*; otherwise we fall back to the oracle (which is
    always correct).  Errors therefore only arise when the proxy both predicts
    incorrectly **and** its confidence surpasses the threshold.  By scanning
    thresholds from low → high we find the smallest value that reaches the
    desired accuracy, defaulting to `inf` (oracle-only) if necessary.

    Args:
        train_df: Training DataFrame with one row per document and the columns
            `label`, `baseline_prediction`, and `baseline_confidence`.
        accuracy_target: Minimum required accuracy for the proxy-oracle cascade
            on the training set (e.g. 0.9).
        task: Task identifier used to fetch the list of class labels from
            `PROMPT_TO_CLASSES_DICT`.

    Returns:
        Dict[class_label, threshold] mapping each class to its confidence
        threshold.
    """
    classes = PROMPT_TO_CLASSES_DICT[task]
    thresholds: Dict[Any, float] = {}
    sequences: Dict[Any, List[float]] = {}
    num_already_predicted = 0
    target = accuracy_target

    for class_label in classes:
        print(f"Computing thresholds for class: {class_label}")
        # Rows whose *true* label is the current class
        class_df = train_df[train_df["baseline_prediction"] == class_label]
        if class_df.empty:
            # No examples of this class – fall back to oracle only.
            thresholds[class_label] = float("inf")
            continue

        # Candidate thresholds: all observed confidences + 0 & inf (sorted asc)
        candidate_thresholds = sorted(
            set(class_df["baseline_confidence"].tolist() + [0.0, float("inf")])
        )

        selected_threshold = float("inf")  # default if target never reached
        for idx, thresh in enumerate(candidate_thresholds):
            # Keep proxy prediction when its confidence ≥ thresh; otherwise oracle
            accept_proxy_mask = class_df["baseline_confidence"] >= thresh
            predictions = class_df["baseline_prediction"].where(
                accept_proxy_mask, class_df["label"]
            )
            accuracy = (predictions == class_df["label"]).mean()
            # high_conf_df = class_df[class_df["baseline_confidence"] >= thresh]
            # accuracy = (high_conf_df["baseline_prediction"]  == high_conf_df["label"]).mean()
            
            if round(accuracy, 2) >= round(target, 2):
                print(f"Current accuracy: {accuracy} (target: {target}) and # proxy predictions: {accept_proxy_mask.sum()}")
                selected_threshold = thresh
                valid_seq = candidate_thresholds[idx:]
                sequences[class_label] = valid_seq
                num_already_predicted += accept_proxy_mask.sum()
                # # Compute target accuracy for remaining documents so overall we can get the desired accuracy
                if num_already_predicted < len(train_df):
                    target = (accuracy_target * len(train_df) - num_already_predicted) / (len(train_df) - num_already_predicted)
                    print(f"New target accuracy: {target}")
                break

        thresholds[class_label] = selected_threshold
        if class_label not in sequences:
            # No threshold met the target; sequence is just [inf]
            sequences[class_label] = [float("inf")]

    return thresholds, sequences

