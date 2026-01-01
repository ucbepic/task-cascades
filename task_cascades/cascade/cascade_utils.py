from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Tuple

from scipy.stats import norm, t

from task_cascades.predictors.predictors import cost_given_token_breakdown, PROMPT_TO_CLASSES_DICT
from task_cascades.stats.bargain_utils import test_if_true_mean_is_above_m


def compute_min_samples_for_class(target_accuracy: float, delta: float = 0.25) -> int:
    """Return the minimum #samples needed so Hoeffding's bound can certify
    that the true accuracy is at least ``target_accuracy`` with confidence
    ``1 - delta`` *assuming* the empirical accuracy is perfect.

    This provides a simple, distribution-free rule of thumb for deciding
    whether a surrogate has observed *enough* instances of a class to be
    considered in the cascade design.

    n >= ln(1/delta) / (2 * (1 - target_accuracy)^2)
    """
    epsilon = 1.0 - target_accuracy
    if epsilon <= 0:
        # Degenerate but guard against division by zero when target=1.0
        return 1
    n_required = np.log(1.0 / delta) / (2.0 * epsilon ** 2)
    return int(np.ceil(n_required))


def calculate_variance_corrected_target(
    accuracies: np.ndarray,
    target_accuracy: float,
    confidence_level: float = 0.75
) -> float:
    """
    Calculate variance-corrected target accuracy using normal distribution.
    
    Args:
        accuracies: Binary array of accuracy indicators (0 or 1)
        target_accuracy: Base target accuracy 
        confidence_level: Confidence level for correction (default 0.75)
    
    Returns:
        Corrected target accuracy accounting for sample variance
    """
    n = len(accuracies)
    if n <= 1:
        return target_accuracy
    sample_variance = np.var(accuracies, ddof=1)
    alpha = 1 - confidence_level
    z_critical = norm.ppf(1 - alpha)
    margin = z_critical * np.sqrt(sample_variance / n)
    corrected_target = target_accuracy + margin
    return min(corrected_target, 1.0)


def find_thresholds_for_surrogate(
    df_subset: pd.DataFrame,
    target_accuracy: float,
    task: str,
    delta: float = 0.25
) -> Dict[str, Any]:
    """
    Analyze prediction accuracy and find optimal thresholds for a classifier,
    ensuring the sample accuracy meets the variance-corrected target accuracy.
    
    Args:
        df_subset: DataFrame with columns ['surrogate_name', 'surrogate_model', 'surrogate_prediction', 'surrogate_confidence', 'label']
        target_accuracy: Target accuracy to achieve with statistical confidence.
        task: Task name to determine classes from PROMPT_TO_CLASSES_DICT
        delta: Confidence level for statistical testing (1-delta is confidence level)
    
    Returns:
        Dict containing thresholds and coverage metrics, or None if criteria not met.
    """
    if len(df_subset) == 0:
        return None
    
    classes = PROMPT_TO_CLASSES_DICT[task]
    confidence_level = 1 - delta

    # Determine the minimum number of samples required per class
    min_samples_required = compute_min_samples_for_class(target_accuracy, delta)
    min_samples_required = min(int(len(df_subset) / 10), min_samples_required)

    # Find threshold for each class
    class_thresholds = {}
    class_coverages = {}
    class_sequences = {}
    
    for class_label in classes:
        class_df = df_subset[df_subset['surrogate_prediction'] == class_label].sort_values('surrogate_confidence')
        
        threshold_found = float('inf')
        coverage_at_threshold = 0.0
        
        # Iterate unique confidence scores in ascending order
        # We want the smallest threshold (which means largest coverage) that meets the target accuracy.
        class_thresholds_unique = np.sort(np.unique(class_df['surrogate_confidence']))

        for current_conf_threshold in class_thresholds_unique:
            mask = class_df['surrogate_confidence'] >= current_conf_threshold
            high_conf_class_subset = class_df[mask]
            
            n_class = len(high_conf_class_subset)
            # Require enough *qualified* examples
            # if n_class < int(min_samples_required):
            #     break
                
            correct_indicators = (high_conf_class_subset['surrogate_prediction'] == high_conf_class_subset['label']).to_numpy()
            
            sample_accuracy = np.mean(correct_indicators)
            
            # Directly compare against the target (no variance correction here)
            if round(sample_accuracy, 2) >= round(target_accuracy, 2):
                threshold_found = current_conf_threshold
                coverage_at_threshold = n_class / len(class_df)
                break

        # Record results; if no threshold satisfied both accuracy and sample-count requirements we mark as unsupported.
        class_thresholds[class_label] = threshold_found
        class_coverages[class_label] = coverage_at_threshold
        if np.isfinite(threshold_found):
            above_idx = np.where(class_thresholds_unique >= threshold_found)[0]
            seq = class_thresholds_unique[above_idx] if len(above_idx) > 0 else np.array([threshold_found])
        else:
            seq = np.array([float('inf')])
        class_sequences[class_label] = seq.tolist()

    # Calculate overall metrics based on the found thresholds
    total_selected_count = 0
    
    for class_label in classes:
        class_df = df_subset[df_subset['surrogate_prediction'] == class_label]
        final_high_conf_class = class_df[class_df['surrogate_confidence'] >= class_thresholds[class_label]]
        total_selected_count += len(final_high_conf_class)
        
    # Check that total_selected_count > min_samples_required
    if total_selected_count < min_samples_required:
        print(f"Not enough samples for {task} with target {target_accuracy} and delta {delta}. Got {total_selected_count} out of {min_samples_required} required.")
        return None
    
    total_coverage_overall = total_selected_count / len(df_subset) if len(df_subset) > 0 else 0.0
    
    if total_selected_count == 0: # No items selected by any threshold
        return None

    # Final check: coverage must be > 0
    if total_coverage_overall == 0:
        return None

    result = {
        'coverage': float(total_coverage_overall),
    }
    
    # Add class-specific thresholds and coverages
    for class_label in classes:
        result[f'threshold_{class_label}'] = float(class_thresholds[class_label])
        result[f'coverage_{class_label}'] = float(class_coverages[class_label])
        result[f'sequence_{class_label}'] = class_sequences[class_label]

    return result

def compute_marginal_cost(largest_tokens_processed: int, doc_to_try: Tuple[int, int], full_cost: float, predictor: str) -> float:
    """
    Compute the marginal cost of trying a new document, based on token counts only.
    
    Args:
        largest_tokens_processed: Maximum length of a previously processed version of this document
        doc_to_try: Tuple of (prompt_tokens, completion_tokens) for current document
        full_cost: Full cost of processing the document
        predictor: Model identifier
        
    Returns:
        Float representing the marginal cost
    """
            
    # If no previous processing, the marginal cost is the full cost
    if largest_tokens_processed == 0:
        return full_cost
    
    # Otherwise, pay for only the additional tokens
    prompt_tokens, completion_tokens = doc_to_try
    
    if prompt_tokens > largest_tokens_processed:
        input_tokens_not_cached = prompt_tokens - largest_tokens_processed
        input_tokens_cached = largest_tokens_processed
    else:
        input_tokens_not_cached = 0
        input_tokens_cached = prompt_tokens
        
    return cost_given_token_breakdown(predictor, input_tokens_not_cached, input_tokens_cached, completion_tokens)

def simulate_cascade(
    ordering: List[Tuple[str, str, float]], 
    thresholds: Dict[Tuple[str, str, float], Dict[int, float]], 
    results_df: pd.DataFrame,
    return_labels: bool = False,
) -> Tuple:
    """
    Simulates the cascade on results_df using the given ordering and thresholds.
    
    For each document, iterate over the ordering:
      - For candidate c in ordering, check if confidence ≥ threshold for predicted class
      - If a candidate qualifies, stop and output its prediction (accumulating its cost).
      - If no candidate qualifies, use the oracle's prediction (adding the oracle cost).
    
    Args:
        ordering: List of candidate tuples in cascade order
        thresholds: Dict mapping candidate tuple to dict of {class_label: threshold}
        results_df: DataFrame with predictions, confidences, and costs
    
    Returns:
        Tuple of:
        - overall cascade accuracy
        - total cost
        - stage usage (dict with fraction of docs stopping at each stage)
        - array of final predictions
        - list of confidence scores for predictions made by the cascade (excluding oracle)
    """
    n = len(results_df["uuid"].unique())
    unique_docs = results_df["uuid"].unique()
    predictions = []
    labels = []
    total_cost = 0.0
    stage_counts = {cand: 0 for cand in ordering}
    stage_accuracies = defaultdict(list)
    stage_counts["oracle"] = 0
    
    for doc_id in unique_docs:
        chosen = None
        doc_cost = 0.0
        
        # Get all the rows for this document
        doc_rows = results_df[results_df["uuid"] == doc_id]
        
        doc_label = doc_rows["label"].values[0]
        labels.append(doc_label)
        largest_tokens_processed_per_model = {}
        
        for cand_name, cand_model, cand_fraction in ordering:
            # Find the row for this candidate if it exists
            row = doc_rows[(doc_rows["surrogate_name"] == cand_name) & (doc_rows["surrogate_model"] == cand_model) & (doc_rows["fraction"] == cand_fraction)]
            if len(row) == 0:
                continue
            
            conf = row["surrogate_confidence"].values[0]
            pred = row["surrogate_prediction"].values[0]
            cost = compute_marginal_cost(
                largest_tokens_processed_per_model.get(cand_model, 0),
                doc_to_try=(row["surrogate_usage"].values[0].prompt_tokens, row["surrogate_usage"].values[0].completion_tokens),
                full_cost=row["surrogate_cost"].values[0],
                predictor=cand_model
            )
            doc_cost += cost
            
            # Check if confidence meets threshold for predicted class
            candidate_thresholds = thresholds[(cand_name, cand_model, cand_fraction)]
            threshold_for_pred = candidate_thresholds.get(pred, float('inf'))
            
            if conf >= threshold_for_pred:
                chosen = pred
                stage_counts[(cand_name, cand_model, cand_fraction)] += 1
                break
                
            doc_tried_len = row["surrogate_usage"].values[0].prompt_tokens
            if doc_tried_len > largest_tokens_processed_per_model.get(cand_model, 0):
                largest_tokens_processed_per_model[cand_model] = doc_tried_len
        
        if chosen is None:
            chosen = doc_label
            doc_cost += doc_rows["oracle_cost"].values[0]
            stage_counts["oracle"] += 1
        else:
            stage_accuracies[(cand_name, cand_model, cand_fraction)].append(int(chosen == doc_label))
            
        predictions.append(chosen)
        total_cost += doc_cost
        
    assert len(predictions) == n, f"Number of predictions ({len(predictions)}) does not match number of unique docs ({n})"
    
    overall_accuracy = np.mean(np.array(predictions) == np.array(labels)) if len(predictions) > 0 else 0.0
    stage_usage = {k: v / n for k, v in stage_counts.items()} if n > 0 else {k: 0 for k in stage_counts.items()}
    
    # Turn stage_accuracies into a list of np arrays
    stage_accuracies = [np.array(accuracies) for _, accuracies in stage_accuracies.items()]
    
    if return_labels:
        return overall_accuracy, total_cost, stage_usage, np.array(predictions), stage_accuracies, np.array(labels)
    else:
        return overall_accuracy, total_cost, stage_usage, np.array(predictions), stage_accuracies


def find_false_positives_and_negatives(
    results_df: pd.DataFrame,
    ordering: List[Tuple[str, str, float]],
    thresholds: Dict[Tuple[str, str, float], Dict[int, float]],
    task: str,
    num_examples: int = 10
) -> List[Dict[str, Any]]:
    """
    Find actual misclassification examples for each class from the cascade candidates.
    
    For each class, we look for examples where:
    - The model predicted incorrectly (prediction != ground truth)
    - These are actual errors the agent can learn from
    - We return examples with various confidence levels to show different types of errors
    
    Args:
        results_df: DataFrame with predictions, confidences, and costs
        ordering: List of candidate tuples in cascade order
        thresholds: Dict mapping candidate tuple to {class_label: threshold}
        task: Task name to determine classes
        num_examples: Number of examples to return per class per candidate
    
    Returns:
        List of dictionaries, one per class. Each dict maps candidate tuples to DataFrames
        of misclassification examples for that true class.
    """
    classes = PROMPT_TO_CLASSES_DICT[task]
    docs_considered = set()
    
    # Initialize result structure: one dict per class
    class_results = []
    for class_label in classes:
        class_results.append({})
    
    # If no ordering (empty cascade), sample from baseline examples
    if not ordering:
        # Get examples from the baseline predictor (s1)
        baseline_subset = results_df[
            (results_df["surrogate_name"] == "s1") & 
            (~results_df["uuid"].isin(docs_considered))
        ]
        
        for i, true_class in enumerate(classes):
            # Find misclassifications where the true label is this class
            misclassified = baseline_subset[
                (baseline_subset["label"] == true_class) & 
                (baseline_subset["surrogate_prediction"] != true_class)
            ]
            
            # Get diverse examples (both high and low confidence errors)
            if len(misclassified) > 0:
                # Sort by confidence to get a mix
                diverse_examples = misclassified.sort_values(
                    "surrogate_confidence", ascending=False
                ).head(num_examples)
                
                if len(diverse_examples) > 0:
                    class_results[i][("s1", diverse_examples.iloc[0]["surrogate_model"], 1.0)] = diverse_examples
                    docs_considered.update(diverse_examples["uuid"].unique())
        
        return class_results
    
    # For existing cascade candidates, find their misclassifications
    for cand_name, cand_model, cand_fraction in ordering:
        subset = results_df[
            (results_df["surrogate_name"] == cand_name) & 
            (results_df["surrogate_model"] == cand_model) & 
            (results_df["fraction"] == cand_fraction) & 
            (~results_df["uuid"].isin(docs_considered))
        ]
        
        # For each true class, find examples where this class was the true label but predicted wrong
        for i, true_class in enumerate(classes):
            # Find misclassifications where the true label is this class
            misclassified_mask = (
                (subset["label"] == true_class) & 
                (subset["surrogate_prediction"] != true_class)
            )
            
            # Get examples with various confidence levels (both high and low confidence errors)
            misclassified_examples = subset[misclassified_mask].sort_values(
                "surrogate_confidence", ascending=False
            ).head(num_examples)
            
            # Store in the appropriate class result dict
            if len(misclassified_examples) > 0:
                class_results[i][(cand_name, cand_model, cand_fraction)] = misclassified_examples
                docs_considered.update(misclassified_examples["uuid"].unique())
    
    return class_results


def design_cascade_greedy(
    data: pd.DataFrame,
    candidates: List[Tuple[str, str, float]],
    target_accuracy: float,
    task: str,
    delta: float = 0.25,
) -> Tuple[
    float,
    List[Tuple[str, str, float]],
    Dict[Tuple[str, str, float], Dict[int, float]],
    Dict[Tuple[str, str, float], Dict[int, List[float]]],
    int,
]:
    """
    Design a cascade using a bottom-up greedy approach with accurate simulation at each step.
    
    The algorithm builds the cascade one candidate at a time:
    1. Start with an empty cascade (default to oracle)
    2. In each iteration, try adding each remaining candidate to the current cascade
    3. Evaluate each candidate using simulate_cascade to get accurate costs
    4. Pick the candidate that results in the lowest overall cost
    5. Stop when adding another candidate doesn't improve the cost
    
    Args:
        data: DataFrame with predictions, confidences, and cost columns
        candidates: List of candidate tuples (surrogate_name, surrogate_model, doc_fraction)
        target_accuracy: The desired accuracy threshold
        task: Task name to determine classes
        delta: Confidence level for statistical testing
    
    Returns:
        A tuple (best_cost, best_ordering, best_thresholds, best_sequences, num_evals) where:
         - best_cost is the total cost incurred by the cascade
         - best_ordering is a list of candidate tuples in cascade order
         - best_thresholds maps each candidate to its class→threshold dict
         - best_sequences maps each candidate to its class→ascending-thresholds list
         - num_evals is the number of candidate evaluations performed
    """
    # Initialize with empty cascade (oracle-only)
    best_ordering = []
    best_thresholds = {}
    best_sequences: Dict[Tuple[str, str, float], Dict[int, List[float]]] = {}
    oracle_cost = data.drop_duplicates(subset=["uuid"])["oracle_cost"].sum()
    best_cost = oracle_cost
    used_candidates = set()
    candidates_that_dont_work = set()
    num_evals = 0
    classes = PROMPT_TO_CLASSES_DICT[task]
    confidence_level = 1 - delta
    
    # Step 1: Precompute thresholds for all candidates for efficiency
    candidate_metrics = {}
    for candidate in candidates:
        s_name, s_model, s_fraction = candidate
        subset_df = data[(data["surrogate_name"] == s_name) & 
                         (data["surrogate_model"] == s_model) & 
                         (data["fraction"] == s_fraction)]
        
        if len(subset_df) > 0:
            thresholds = find_thresholds_for_surrogate(subset_df, target_accuracy, task, delta)
            if thresholds is not None:
                # Convert to new format: {class_label: threshold}
                class_thresholds = {}
                class_sequences = {}
                for class_label in classes:
                    class_thresholds[class_label] = thresholds[f'threshold_{class_label}']
                    class_sequences[class_label] = thresholds[f'sequence_{class_label}']

                candidate_metrics[candidate] = {
                    'thresholds': class_thresholds,
                    'sequences': class_sequences
                }
    
    print(f"Precomputed thresholds for {len(candidate_metrics)} out of {len(candidates)} candidates")
    
    # Track current best cascade configuration
    current_ordering = []
    current_thresholds = {}
    current_sequences: Dict[Tuple[str, str, float], Dict[int, List[float]]] = {}
    current_cost = oracle_cost
    
    # Continue adding to cascade until no improvement
    improvement = True
    while improvement and len(used_candidates) < len(candidates):
        # print("iterating in this loop")
        improvement = False
        best_candidate = None
        best_candidate_cost = float('inf')
        
        # Try each remaining candidate
        for candidate in candidates:
            if candidate in used_candidates or candidate not in candidate_metrics or candidate in candidates_that_dont_work:
                continue
                
            # Create trial cascade with this candidate added
            trial_ordering = current_ordering + [candidate]
            trial_thresholds = current_thresholds.copy()
            trial_thresholds[candidate] = candidate_metrics[candidate]['thresholds']
            trial_sequences = current_sequences.copy()
            trial_sequences[candidate] = candidate_metrics[candidate]['sequences']
            
            # Simulate to get accurate cost
            num_evals += 1
            trial_accuracy, trial_cost, _, _, individual_accuracies = simulate_cascade(trial_ordering, trial_thresholds, data)
            # print(f"Trial accuracy: {trial_accuracy} and trial cost: {trial_cost} and ordering length: {len(trial_ordering)}")
                
            if trial_cost < best_candidate_cost and round(trial_accuracy, 2) >= round(target_accuracy, 2): # Is this candidate potentially better cost-wise?
                # Check each stage accuracy with variance correction
                all_stages_valid = True
                
                # Flatten the individual accuracies into a single array
                for stage_accuracies in individual_accuracies:
                    stage_accuracy = np.mean(stage_accuracies)
                    if round(stage_accuracy, 2) < round(target_accuracy, 2):
                        all_stages_valid = False
                        candidates_that_dont_work.add(candidate)
                        break
                  
                if all_stages_valid:
                    best_candidate = candidate
                    best_candidate_cost = trial_cost
                else:
                    # Optional: Log why a candidate was skipped
                    print(f"Skipping candidate '{candidate}' in trial cascade because cascade accuracy does not meet target")
            
            
        # Update cascade if we found an improvement
        if best_candidate is not None and best_candidate_cost < current_cost:
            current_ordering.append(best_candidate)
            current_thresholds[best_candidate] = candidate_metrics[best_candidate]['thresholds']
            current_sequences[best_candidate] = candidate_metrics[best_candidate]['sequences']
            used_candidates.add(best_candidate)
            current_cost = best_candidate_cost
            improvement = True
            
            # Update overall best if this is better
            if current_cost < best_cost:
                best_cost = current_cost
                best_ordering = current_ordering.copy()
                best_thresholds = current_thresholds.copy()
                best_sequences = current_sequences.copy()
                
            print(f"Added {best_candidate} to cascade. New cost: {current_cost}")
        
        # print(f"Current ordering: {current_ordering} and current cost: {current_cost}")
        # print(f"Num candidates used: {len(used_candidates)}; num candidates total: {len(candidates)}")
    
    # If oracle is better than any cascade, return empty cascade
    if oracle_cost <= best_cost:
        print(f"Oracle cost ({oracle_cost}) is better than best cascade cost ({best_cost}). Using oracle only.")
        return oracle_cost, [], {}, {}, num_evals
    
    print(f"Greedy cascade design complete. Cascade has {len(best_ordering)} candidates.")
    return best_cost, best_ordering, best_thresholds, best_sequences, num_evals

def design_cascade_selectivity_ranking(
    data: pd.DataFrame,
    candidates: List[Tuple[str, str, float]],
    target_accuracy: float,
    task: str,
    delta: float = 0.25
) -> Tuple[
    float,
    List[Tuple[str, str, float]],
    Dict[Tuple[str, str, float], Dict[int, float]],
    Dict[Tuple[str, str, float], Dict[int, List[float]]],
    int,
]:
    """
    Design a cascade using selectivity-based ranking approach.
    
    The algorithm ranks candidates by selectivity(coverage) - 1/cost and builds cascade in that order:
    1. For each candidate, compute coverage and average cost per document
    2. Rank candidates by coverage - 1/avg_cost (highest first)
    3. Build cascade in this order, validating accuracy constraints at each step
    4. Stop when adding another candidate doesn't improve the cost or violates accuracy
    
    Args:
        data: DataFrame with predictions, confidences, and cost columns
        candidates: List of candidate tuples (surrogate_name, surrogate_model, doc_fraction)
        target_accuracy: The desired accuracy threshold
        task: Task name to determine classes
        delta: Confidence level for lower bound accuracy calculation
    
    Returns:
        A tuple (best_cost, best_ordering, best_thresholds, best_sequences, num_evals) where:
         - best_cost is the total cost incurred by the cascade
         - best_ordering is a list of candidate tuples in cascade order
         - best_thresholds is a dict mapping candidate tuple to {class_label: threshold}
         - best_sequences maps each candidate to its class→ascending-thresholds list
         - num_evals is the number of candidate evaluations performed
    """
    # Initialize with empty cascade (oracle-only)
    oracle_cost = data.drop_duplicates(subset=["uuid"])["oracle_cost"].sum()
    num_docs = len(data.drop_duplicates(subset=["uuid"]))
    num_evals = 0
    classes = PROMPT_TO_CLASSES_DICT[task]
    confidence_level = 1 - delta
    
    # Step 1: Precompute metrics for all candidates
    candidate_metrics = {}
    candidate_rankings = []
    
    for candidate in candidates:
        s_name, s_model, s_fraction = candidate
        subset_df = data[(data["surrogate_name"] == s_name) & 
                         (data["surrogate_model"] == s_model) & 
                         (data["fraction"] == s_fraction)]
        
        if len(subset_df) > 0:
            thresholds = find_thresholds_for_surrogate(subset_df, target_accuracy, task, delta)
            if thresholds is not None:
                # Calculate average cost per document for this candidate
                avg_cost_per_doc = subset_df["surrogate_cost"].mean()
                
                # Selectivity ranking: (coverage - 1) / avg_cost
                # Use a small epsilon to avoid division by zero
                epsilon = 1e-6
                ranking_score = (thresholds['coverage'] - 1.0) / max(avg_cost_per_doc, epsilon)
                
                # Convert to new format: {class_label: threshold}
                class_thresholds = {}
                for class_label in classes:
                    class_thresholds[class_label] = thresholds[f'threshold_{class_label}']
                
                # Build sequences dict per class
                class_sequences = {}
                for class_label in classes:
                    class_sequences[class_label] = thresholds[f'sequence_{class_label}']

                candidate_metrics[candidate] = {
                    'thresholds': class_thresholds,
                    'sequences': class_sequences,
                    'coverage': thresholds['coverage'],
                    'avg_cost_per_doc': avg_cost_per_doc,
                    'ranking_score': ranking_score
                }
                
                candidate_rankings.append((candidate, ranking_score))
    
    # Sort candidates by ranking score (highest first)
    candidate_rankings.sort(key=lambda x: x[1], reverse=True)
    print(f"Selectivity ranking: candidates ranked by (coverage - 1) / cost")
    for candidate, score in candidate_rankings:
        coverage = candidate_metrics[candidate]['coverage']
        cost = candidate_metrics[candidate]['avg_cost_per_doc']
        print(f"  {candidate}: coverage={coverage:.3f}, avg_cost={cost:.6f}, score={score:.3f}")
    
    # Step 2: Build cascade in ranking order
    best_ordering = []
    best_thresholds = {}
    best_sequences: Dict[Tuple[str, str, float], Dict[int, List[float]]] = {}
    best_cost = oracle_cost
    
    current_ordering = []
    current_thresholds = {}
    current_sequences: Dict[Tuple[str, str, float], Dict[int, List[float]]] = {}
    current_cost = oracle_cost
    
    # Try adding candidates in ranking order
    for candidate, _ in candidate_rankings:
        if candidate not in candidate_metrics:
            continue
            
        # Create trial cascade with this candidate added
        trial_ordering = current_ordering + [candidate]
        trial_thresholds = current_thresholds.copy()
        trial_thresholds[candidate] = candidate_metrics[candidate]['thresholds']
        trial_sequences = current_sequences.copy()
        trial_sequences[candidate] = candidate_metrics[candidate]['sequences']
        
        # Simulate to get accurate cost and accuracy
        num_evals += 1
        trial_accuracy, trial_cost, _, _, individual_accuracies = simulate_cascade(trial_ordering, trial_thresholds, data)

        # Check if this candidate improves cost while maintaining accuracy
        if trial_cost < current_cost and round(trial_accuracy, 2) >= round(target_accuracy, 2):
            # Check each stage accuracy with variance correction
            all_stages_valid = True
            for stage_accuracies in individual_accuracies:
                if len(stage_accuracies) > 0:
                    stage_accuracy = np.mean(stage_accuracies)
                    if round(stage_accuracy, 2) < round(target_accuracy, 2):
                        all_stages_valid = False
                        break
            
            if all_stages_valid:
                current_ordering.append(candidate)
                current_thresholds[candidate] = candidate_metrics[candidate]['thresholds']
                current_sequences[candidate] = candidate_metrics[candidate]['sequences']
                current_cost = trial_cost
                
                # Update overall best if this is better
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_ordering = current_ordering.copy()
                    best_thresholds = current_thresholds.copy()
                    best_sequences = current_sequences.copy()
                    
                print(f"Added {candidate} to selectivity cascade. New cost: {current_cost:.4f}")
        
    # If oracle is better than any cascade, return empty cascade
    if oracle_cost <= best_cost:
        print(f"Oracle cost ({oracle_cost}) is better than best selectivity cascade cost ({best_cost}). Using oracle only.")
        return oracle_cost, [], {}, {}, num_evals
    
    print(f"Selectivity cascade design complete. Cascade has {len(best_ordering)} candidates.")
    return best_cost, best_ordering, best_thresholds, best_sequences, num_evals

def design_cascade_optimal_greedy(
    results_df: pd.DataFrame,
    candidates: List[Tuple[str, str, float]],
    target_accuracy: float,
    task: str,
    max_shift: int = 5,
    delta: float = 0.25,
    guarantee_accuracy: bool = False,
) -> Dict[str, Any]:
    """
    Design the cascade using a greedy approach.
    
    Args:
        results_df: DataFrame with predictions, confidences, and cost columns.
        candidates: List of candidate tuples (surrogate_name, surrogate_model, doc_fraction).
        target_accuracy: The desired target accuracy.
        task: Task name to determine classes.
        max_shift: Maximum shift index to consider (inclusive).
        delta: Total type-I error budget. Each Bargain test is run at delta/(max_shift+1).
        guarantee_accuracy: Whether to guarantee accuracy using proper data splitting.
    
    Returns:
        A dict containing:
         - 'ordering': Greedy cascade ordering (list of candidate tuples).
         - 'thresholds': Mapping from candidate tuple to {class_label: threshold}.
         - 'total_cost': Total cost of the cascade (with accurate simulation).
         - 'accuracy': Overall accuracy (via simulate_cascade).
         - 'stage_usage': Fraction of docs handled at each stage.
         - 'num_evals': Number of candidate evaluations performed.
         - 'meta_shift': The chosen shift index (0 == original thresholds).
    """
    print(f"Designing greedy cascade for {len(candidates)} candidates")
    
    if guarantee_accuracy:
        # Split data 50-50: first half for cascade design, second half for Bargain test
        unique_docs = results_df["uuid"].unique()
        np.random.shuffle(unique_docs)
        
        split_point = len(unique_docs) // 2
        train_docs = unique_docs[:split_point]
        test_docs = unique_docs[split_point:]
        
        train_df = results_df[results_df["uuid"].isin(train_docs)].copy()
        test_df = results_df[results_df["uuid"].isin(test_docs)].copy()
        
        print(f"Data split for accuracy guarantee: {len(train_docs)} docs for cascade design, {len(test_docs)} docs for Bargain test")
        
        # Design cascade on training split
        best_cost, best_ordering, best_thresholds, best_sequences, num_evals = design_cascade_greedy(train_df, candidates, target_accuracy, task)
        
        if best_ordering:
            # Apply global threshold shift with Bonferroni/Bargain on test split
            shifted_thresholds, best_shift = tune_global_threshold_shift_bonferroni(
                test_df,
                best_ordering,
                best_sequences,
                target_accuracy,
                task,
                max_shift=max_shift,
                delta=delta,
            )
            print(f"Global threshold shift: {best_shift}")
            final_thresholds = shifted_thresholds if shifted_thresholds else best_thresholds
            
            # Final simulation on test split (holdout set)
            cascade_acc, total_cost, stage_usage, preds, _ = simulate_cascade(
                best_ordering, final_thresholds, test_df
            )
        else:
            best_shift = 0
            final_thresholds = best_thresholds
            # Simulate on test split
            cascade_acc, total_cost, stage_usage, preds, _ = simulate_cascade(
                best_ordering, final_thresholds, test_df
            )
    else:
        # Use all data for cascade design (no accuracy guarantee)
        best_cost, best_ordering, best_thresholds, best_sequences, num_evals = design_cascade_greedy(results_df, candidates, target_accuracy, task)
        
        best_shift = 0
        final_thresholds = best_thresholds

        # Simulate the cascade using the final thresholds on full data
        cascade_acc, total_cost, stage_usage, preds, _ = simulate_cascade(
            best_ordering, final_thresholds, results_df
        )

    return {
        'ordering': best_ordering,
        'thresholds': final_thresholds,
        'total_cost': total_cost,
        'accuracy': cascade_acc,
        'stage_usage': stage_usage,
        'num_evals': num_evals,
        'meta_shift': best_shift,
    }

def design_cascade_optimal_selectivity(
    results_df: pd.DataFrame,
    candidates: List[Tuple[str, str, float]],
    target_accuracy: float,
    task: str,
    max_shift: int = 5,
    delta: float = 0.25,
    guarantee_accuracy: bool = False,
) -> Dict[str, Any]:
    """
    Design the cascade using a selectivity-based ranking approach.
    
    Args:
        results_df: DataFrame with predictions, confidences, and cost columns.
        candidates: List of candidate tuples (surrogate_name, surrogate_model, doc_fraction).
        target_accuracy: The desired target accuracy.
        task: Task name to determine classes.
        max_shift: Maximum shift index to consider (inclusive).
        delta: Total type-I error budget. Each Bargain test is run at delta/(max_shift+1).
        guarantee_accuracy: Whether to guarantee accuracy using proper data splitting.
    
    Returns:
        A dict containing:
         - 'ordering': Selectivity-based cascade ordering (list of candidate tuples).
         - 'thresholds': Mapping from candidate tuple to {class_label: threshold}.
         - 'total_cost': Total cost of the cascade (with accurate simulation).
         - 'accuracy': Overall accuracy (via simulate_cascade).
         - 'stage_usage': Fraction of docs handled at each stage.
         - 'num_evals': Number of candidate evaluations performed.
         - 'meta_shift': The chosen shift index (0 == original thresholds).
    """
    print(f"Designing selectivity-based cascade for {len(candidates)} candidates")
    
    if guarantee_accuracy:
        # Split data 50-50: first half for cascade design, second half for Bargain test
        unique_docs = results_df["uuid"].unique()
        np.random.shuffle(unique_docs)
        
        split_point = len(unique_docs) // 2
        train_docs = unique_docs[:split_point]
        test_docs = unique_docs[split_point:]
        
        train_df = results_df[results_df["uuid"].isin(train_docs)].copy()
        test_df = results_df[results_df["uuid"].isin(test_docs)].copy()
        
        print(f"Data split for accuracy guarantee: {len(train_docs)} docs for cascade design, {len(test_docs)} docs for Bargain test")
        
        # Design cascade on training split
        best_cost, best_ordering, best_thresholds, best_sequences, num_evals = design_cascade_selectivity_ranking(train_df, candidates, target_accuracy, task)
        
        if best_ordering:
            # Apply global threshold shift with Bonferroni/Bargain on test split
            shifted_thresholds, best_shift = tune_global_threshold_shift_bonferroni(
                test_df,
                best_ordering,
                best_sequences,
                target_accuracy,
                task,
                max_shift=max_shift,
                delta=delta,
            )
            final_thresholds = shifted_thresholds if shifted_thresholds else best_thresholds
            
            # Final simulation on test split (holdout set)
            cascade_acc, total_cost, stage_usage, preds, _ = simulate_cascade(
                best_ordering, final_thresholds, test_df
            )
        else:
            best_shift = 0
            final_thresholds = best_thresholds
            # Simulate on test split
            cascade_acc, total_cost, stage_usage, preds, _ = simulate_cascade(
                best_ordering, final_thresholds, test_df
            )
    else:
        # Use all data for cascade design (no accuracy guarantee)
        best_cost, best_ordering, best_thresholds, best_sequences, num_evals = design_cascade_selectivity_ranking(results_df, candidates, target_accuracy, task)
        
        best_shift = 0
        final_thresholds = best_thresholds

        # Simulate cascade with final thresholds on full data
        cascade_acc, total_cost, stage_usage, preds, _ = simulate_cascade(
            best_ordering, final_thresholds, results_df
        )

    return {
        'ordering': best_ordering,
        'thresholds': final_thresholds,
        'total_cost': total_cost,
        'accuracy': cascade_acc,
        'stage_usage': stage_usage,
        'num_evals': num_evals,
        'meta_shift': best_shift,
    }

def _build_shifted_thresholds(sequence_dict: Dict[Any, List[float]], shift: int) -> Dict[Any, float]:
    """Build a per-class threshold dict from a sequence dict and shift index.

    Args:
        sequence_dict: Mapping from class label to *ascending* list of valid thresholds
        shift: Non-negative integer. 0 means use the minimum viable threshold (first
               element of the sequence), 1 means the next stricter threshold, etc.

    Returns
    -------
    Dict[class_label, float]
        Thresholds to use for this shift. If the requested shift is beyond the
        sequence length we fall back to `float('inf')`, effectively disabling
        selections for that class at this candidate stage.
    """
    thresholds = {}
    for class_label, seq in sequence_dict.items():
        if shift < len(seq):
            thresholds[class_label] = seq[shift]
        else:
            thresholds[class_label] = float('inf')
    return thresholds


def tune_global_threshold_shift_bonferroni(
    data: pd.DataFrame,
    ordering: List[Tuple[str, str, float]],
    sequences_per_candidate: Dict[Tuple[str, str, float], Dict[Any, List[float]]],
    base_target_accuracy: float,
    task: str,
    max_shift: int = 5,
    delta: float = 0.25,
) -> Tuple[Dict[Tuple[str, str, float], Dict[Any, float]], int]:
    """Search for the largest global shift that passes Bargain with Bonferroni.

    Parameters
    ----------
    data : pd.DataFrame
        Full results dataframe.
    ordering : list
        Candidate ordering of the cascade.
    sequences_per_candidate : dict
        Mapping from candidate tuple to dict of {class_label: ascending_list_of_thresholds}
    base_target_accuracy : float
        Target accuracy we must guarantee.
    task : str
        Task identifier used for class mapping.
    max_shift : int, default 5
        Maximum shift index to consider (inclusive).
    delta : float, default 0.25
        Same delta used for threshold computation. Total type-I error budget. Each Bargain test is run at delta/(max_shift+1).

    Returns
    -------
    new_thresholds : Dict[candidate, Dict[class_label, float]]
        Thresholds after applying the chosen global shift.
    best_shift : int
        The chosen shift index (0 == original thresholds).
    """
    if not ordering:
        # No cascade candidates — nothing to tune
        return {}, 0

    # We removed this because we actually don't need it.
    # Bonferroni correction
    # k = max_shift + 1  # number of hypotheses to be tested (shift = 0 .. max_shift)
    # alpha_each = delta / k

    best_shift = 0

    # Iterate from strictest (max_shift) down to 0
    shift_range = list(range(0, max_shift + 1))
    shift_range.reverse()

    for shift in shift_range:
        # Build thresholds for this shift
        shifted_thresholds: Dict[Tuple[str, str, float], Dict[Any, float]] = {}
        for cand in ordering:
            seq_dict = sequences_per_candidate[cand]
            shifted_thresholds[cand] = _build_shifted_thresholds(seq_dict, shift)

        # Simulate cascade and collect predictions + labels
        (
            _acc,
            _cost,
            _stage_usage,
            preds,
            _stage_accs,
            labels,
        ) = simulate_cascade(ordering, shifted_thresholds, data, return_labels=True)

        accuracy_indicators = (preds == labels).astype(int)

        passed = test_if_true_mean_is_above_m(
            accuracy_indicators, base_target_accuracy, delta
        )

        if passed:
            best_shift = shift
            # Keep going to see if a more lenient threshold also passes
            continue
        else:
            # First failure encountered; previous shift was the last valid
            break

    # Re-build thresholds for best_shift to return
    final_thresholds: Dict[Tuple[str, str, float], Dict[Any, float]] = {}
    for cand in ordering:
        final_thresholds[cand] = _build_shifted_thresholds(
            sequences_per_candidate[cand], best_shift
        )

    return final_thresholds, best_shift
    