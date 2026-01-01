import argparse
import json
import logging
import os
from pathlib import Path
import pickle
import sys
import time
from os.path import dirname, abspath
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import yaml

# Add the parent directory to sys.path to allow task_cascades imports
root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)

from task_cascades.config.config import ExperimentConfig, MethodConfig
from task_cascades.experiments.experiment_runner import ExperimentRunner
from task_cascades.data.create_dfs import prepare_data, load_dataset, apply_filtering_calibrator_to_dataframe
from task_cascades.filtering.train_classifier_for_filtering import train_data_filtering, simple_similarity_data_filtering, position_based_data_filtering
from task_cascades.filtering.data_filtering_utils import chunk_and_get_confidences
from task_cascades.filtering.calibrators import train_filtering_calibrator
from task_cascades.cascade.find_surrogates import find_surrogates
from task_cascades.cascade.apply_trained_cascade import apply_cascade, train_and_apply_baseline_cascade
from task_cascades.predictors.predictors import PROMPT_TO_TASK_TYPE_DICT, TASK_PROMPT_DICT

# Create rich console for pretty printing
console = Console()

# Updated configuration - now using config classes
config = ExperimentConfig()
method_config = MethodConfig()

# Legacy constants for backward compatibility
SAMPLE_SIZE = config.SAMPLE_SIZE
TARGET_ACCURACIES = config.TARGET_ACCURACIES
CACHE_DIR = config.CACHE_DIR
RESULTS_DIR = config.RESULTS_DIR

def ensure_cache_dir():
    """Ensure the cache directory exists"""
    os.makedirs(CACHE_DIR, exist_ok=True)

def ensure_results_dir():
    """Ensure the results directory exists"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

def get_cache_path(task: str, sample_size: int, seed: int, suffix: str = "") -> str:
    """Get the cache file path for a given task, sample size, and seed"""
    return os.path.join(CACHE_DIR, f"{task}_{sample_size}_seed_{seed}{suffix}_cache.pkl")

def get_filtering_calibrator_path(task: str, seed: int, target_accuracy: float) -> str:
    """Get path to the filtering calibrator cache file"""
    ensure_cache_dir()
    return os.path.join(CACHE_DIR, f"{task}_seed_{seed}_target_{target_accuracy}_filtering_calibrator.pkl")

def save_to_cache(data: dict, cache_path: str):
    """Save data to cache"""
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

def load_from_cache(cache_path: str) -> dict:
    """Load data from cache"""
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def save_filtering_calibrator(task: str, seed: int, target_accuracy: float, filtering_calibrator):
    """Save filtering calibrator to cache"""
    with open(get_filtering_calibrator_path(task, seed, target_accuracy), 'wb') as f:
        pickle.dump(filtering_calibrator, f)

def load_filtering_calibrator(task: str, seed: int, target_accuracy: float):
    """Load filtering calibrator from cache"""
    with open(get_filtering_calibrator_path(task, seed, target_accuracy), 'rb') as f:
        return pickle.load(f)
    
def get_classifier_path(task: str, seed: int) -> str:
    """Get path to the classifier cache file"""
    ensure_cache_dir()
    return os.path.join(CACHE_DIR, f"{task}_seed_{seed}_classifier.pkl")

def load_classifier(task: str, seed: int):
    """Load classifier from cache"""
    with open(get_classifier_path(task, seed), 'rb') as f:
        d = pickle.load(f)
        return d['classifier'], d['chunk_size']
    
def save_classifier(task: str, seed: int, classifier, chunk_size: int):
    """Save classifier and metadata to cache"""
    with open(get_classifier_path(task, seed), 'wb') as f:
        pickle.dump({
            'classifier': classifier,
            'chunk_size': chunk_size
        }, f)

def get_cascade_results_path(task: str, seed: int, target_accuracy: float) -> str:
    """Get path to the cascade results cache file"""
    ensure_cache_dir()
    return os.path.join(CACHE_DIR, f"{task}_seed_{seed}_target_{target_accuracy}_cascade_results.pkl")

def save_cascade_results(task: str, seed: int, target_accuracy: float, cascade_results: dict):
    """Save cascade results to cache"""
    with open(get_cascade_results_path(task, seed, target_accuracy), 'wb') as f:
        pickle.dump(cascade_results, f)

def load_cascade_results(task: str, seed: int, target_accuracy: float) -> dict:
    """Load cascade results from cache"""
    with open(get_cascade_results_path(task, seed, target_accuracy), 'rb') as f:
        return pickle.load(f)

def create_no_data_filtering_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a version of the dataframe with no data filtering applied.
    Sets filtered_text to the full text and document_fraction to 1.0.
    """
    df_no_filtering = df.copy()
    df_no_filtering['filtered_text'] = df_no_filtering['text']
    df_no_filtering['fraction'] = 1.0
    return df_no_filtering

def create_no_surrogate_cascade(train_df: pd.DataFrame, task: str, target_accuracy: float) -> dict:
    """
    Create a cascade with only the baseline task (no surrogate tasks).
    This is essentially just the baseline task applied with different thresholds.
    """
    from task_cascades.cascade.cascade_utils import design_cascade_optimal_greedy
    from task_cascades.predictors.predictors import BASELINE_PREDICTOR, ORACLE_PREDICTOR, PREDICTORS
    from task_cascades.config.consts import CANDIDATE_FRACTIONS
    
    # Run baseline and oracle predictors on the main task
    from task_cascades.predictors.predictors import run_predictor_and_get_row_copies
    
    all_executions = []
    task_prompt = TASK_PROMPT_DICT[task]
    
    # Run predictors for the main task only
    baseline_results = run_predictor_and_get_row_copies(
        BASELINE_PREDICTOR, task_prompt, train_df, "s1", 
        task_type=PROMPT_TO_TASK_TYPE_DICT[task]
    )
    oracle_results = run_predictor_and_get_row_copies(
        ORACLE_PREDICTOR, task_prompt, train_df, "s1",
        task_type=PROMPT_TO_TASK_TYPE_DICT[task]
    )
    
    all_executions.extend(baseline_results)
    all_executions.extend(oracle_results)
    
    all_executions_df = pd.DataFrame(all_executions)
    
    # Create candidates from s1 only
    all_candidates = []
    for doc_fraction in CANDIDATE_FRACTIONS:
        for predictor in PREDICTORS:
            if predictor == ORACLE_PREDICTOR and doc_fraction == 1.0:
                continue
            all_candidates.append(("s1", predictor, doc_fraction))
    
    # Design cascade (only greedy for ablation)
    cascade_greedy = design_cascade_optimal_greedy(all_executions_df, all_candidates, target_accuracy, task)
    
    # Create surrogate_to_prompt mapping
    surrogate_to_prompt = {"s1": task_prompt}
    
    return {
        "greedy": {**cascade_greedy, "surrogate_to_prompt": surrogate_to_prompt},
        "surrogate_to_prompt": surrogate_to_prompt
    }

def save_full_experiment_results(
    task: str, 
    sample_size: int, 
    seed: int,
    target_accuracy: float,
    results_dict: dict,
    oracle_cost: float
):
    """Save comprehensive experiment results to a structured file"""
    ensure_results_dir()
    
    # Create a comprehensive results structure
    full_results = {
        "task": task,
        "sample_size": sample_size,
        "seed": seed,
        "target_accuracy": target_accuracy,
        "timestamp": pd.Timestamp.now().isoformat(),
        "methods": results_dict,
        "oracle_cost": oracle_cost
    }
    
    # Save to a timestamped file
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(RESULTS_DIR, f"{task}_full_experiment_seed_{seed}_target_{target_accuracy}_{timestamp}.pkl")
    
    with open(results_path, 'wb') as f:
        pickle.dump(full_results, f)
    
    # Also save a "latest" version for easy access
    latest_path = os.path.join(RESULTS_DIR, f"{task}_full_experiment_seed_{seed}_target_{target_accuracy}_latest.pkl")
    with open(latest_path, 'wb') as f:
        pickle.dump(full_results, f)
    
    return results_path, latest_path

def print_comprehensive_results(results_dict: dict, oracle_only_cost: float, target_accuracy: float):
    """Print a comprehensive comparison table of all methods"""
    
    # Create main comparison table
    table = Table(title=f"Full Experiment Results: All Methods Comparison (Target Accuracy: {target_accuracy})")
    table.add_column("Method", style="bold white")
    table.add_column("Accuracy", justify="right")
    table.add_column("Total Cost", justify="right")
    table.add_column("Cost Reduction", justify="right")
    table.add_column("Relative Cost", justify="right")
    table.add_column("Runtime (s)", justify="right")
    table.add_column("Description", style="dim")
    
    # Use method styles from config
    method_styles = method_config.METHOD_STYLES
    
    # Determine display order based on the canonical style list, but include only
    # those for which we have results (plus the special oracle_only row).
    method_order = [
        m for m in method_styles.keys()
        if (m in results_dict) or (m == "oracle_only")
    ]

    # Use method descriptions from config
    method_descriptions = method_config.METHOD_DESCRIPTIONS
    
    # Add rows in the specified order
    for method_key in method_order:
        if method_key == "oracle_only":
            table.add_row(
                method_styles[method_key],
                "1.0000",
                f"{oracle_only_cost:.4f}",
                "0.00%",
                "1.00x",
                "-",
                method_descriptions[method_key]
            )
        elif method_key in results_dict:
            result = results_dict[method_key]
            cost_reduction = (1 - result['total_cost'] / oracle_only_cost) * 100
            relative_cost = result['total_cost'] / oracle_only_cost
            runtime_str = f"{result.get('runtime', 0):.1f}" if 'runtime' in result else "-"
            
            table.add_row(
                method_styles[method_key],
                f"{result['overall_accuracy']:.4f}",
                f"{result['total_cost']:.4f}",
                f"{cost_reduction:.2f}%",
                f"{relative_cost:.2f}x",
                runtime_str,
                method_descriptions[method_key]
            )
    
    console.print("\n")
    console.print(table)
    
    # Print method comparisons
    console.print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[bold]ABLATION STUDY INSIGHTS:[/bold]")
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Compare main methods
    if "main_greedy" in results_dict and "main_selectivity" in results_dict:
        main_greedy_cost = results_dict["main_greedy"]["total_cost"]
        main_selectivity_cost = results_dict["main_selectivity"]["total_cost"]
        if main_greedy_cost < main_selectivity_cost:
            improvement = (main_selectivity_cost - main_greedy_cost) / main_selectivity_cost * 100
            console.print(f"• 🏆 Greedy method outperforms selectivity by [bold green]{improvement:.1f}%[/bold green]")
        else:
            improvement = (main_greedy_cost - main_selectivity_cost) / main_greedy_cost * 100
            console.print(f"• 🏆 Selectivity method outperforms greedy by [bold purple]{improvement:.1f}%[/bold purple]")
    
    # Impact of data filtering
    if "main_greedy" in results_dict and "no_data_filtering_greedy" in results_dict:
        main_cost = results_dict["main_greedy"]["total_cost"]
        no_filtering_cost = results_dict["no_data_filtering_greedy"]["total_cost"]
        filtering_benefit = (no_filtering_cost - main_cost) / no_filtering_cost * 100
        console.print(f"• 📊 Data filtering provides [bold cyan]{filtering_benefit:.1f}%[/bold cyan] cost reduction")
    
    # Impact of surrogate tasks
    if "main_greedy" in results_dict and "no_surrogate_greedy" in results_dict:
        main_cost = results_dict["main_greedy"]["total_cost"]
        no_surrogate_cost = results_dict["no_surrogate_greedy"]["total_cost"]
        surrogate_benefit = (no_surrogate_cost - main_cost) / no_surrogate_cost * 100
        console.print(f"• 🎯 Surrogate tasks provide [bold magenta]{surrogate_benefit:.1f}%[/bold magenta] cost reduction")
    
    # Impact of iterative feedback vs single iteration
    if "main_greedy" in results_dict and "single_iteration_agent_greedy" in results_dict:
        main_cost = results_dict["main_greedy"]["total_cost"]
        single_iter_cost = results_dict["single_iteration_agent_greedy"]["total_cost"]
        if main_cost < single_iter_cost:
            feedback_benefit = (single_iter_cost - main_cost) / single_iter_cost * 100
            console.print(f"• 🔄 Iterative feedback provides [bold green]{feedback_benefit:.1f}%[/bold green] improvement over single iteration")
        else:
            single_iter_benefit = (main_cost - single_iter_cost) / main_cost * 100
            console.print(f"• 🤖 Single iteration outperforms iterative by [bold magenta]{single_iter_benefit:.1f}%[/bold magenta]")
    
    # Impact of sophisticated vs simple data filtering
    if "main_greedy" in results_dict and "simple_similarity_filtering_greedy" in results_dict:
        main_cost = results_dict["main_greedy"]["total_cost"]
        simple_filtering_cost = results_dict["simple_similarity_filtering_greedy"]["total_cost"]
        if main_cost < simple_filtering_cost:
            sophisticated_benefit = (simple_filtering_cost - main_cost) / simple_filtering_cost * 100
            console.print(f"• 🧠 Sophisticated filtering provides [bold green]{sophisticated_benefit:.1f}%[/bold green] improvement over simple similarity")
        else:
            simple_benefit = (main_cost - simple_filtering_cost) / main_cost * 100
            console.print(f"• 📐 Simple similarity filtering outperforms sophisticated by [bold yellow]{simple_benefit:.1f}%[/bold yellow]")
    
    # Impact of embeddings vs position-based filtering
    if "simple_similarity_filtering_greedy" in results_dict and "position_based_filtering_greedy" in results_dict:
        simple_cost = results_dict["simple_similarity_filtering_greedy"]["total_cost"]
        position_cost = results_dict["position_based_filtering_greedy"]["total_cost"]
        if simple_cost < position_cost:
            embedding_benefit = (position_cost - simple_cost) / position_cost * 100
            console.print(f"• 🔗 Embedding-based filtering provides [bold yellow]{embedding_benefit:.1f}%[/bold yellow] improvement over position-based")
        else:
            position_benefit = (simple_cost - position_cost) / simple_cost * 100
            console.print(f"• 📍 Position-based filtering outperforms embedding-based by [bold red]{position_benefit:.1f}%[/bold red]")
    
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    # Accuracy-guaranteed results are computed in the main execution block; no cascade execution here.

def load_latest_experiment_results(task: str, seed: int, target_accuracy: float) -> dict:
    """Load the latest experiment results if they exist"""
    ensure_results_dir()
    latest_path = os.path.join(RESULTS_DIR, f"{task}_full_experiment_seed_{seed}_target_{target_accuracy}_latest.pkl")
    
    if os.path.exists(latest_path):
        with open(latest_path, 'rb') as f:
            full_results = pickle.load(f)
            return full_results.get("methods", {})
    return {}

def determine_methods_to_run(methods_config: list, existing_results: dict) -> tuple[list, dict]:
    """
    Determine which methods need to be run based on config and existing results.
    
    Returns:
        - List of methods that need to be run
        - Dictionary of existing results to preserve
    """
    methods_to_run = []
    existing_to_preserve = {}
    
    for method in methods_config:
        if method in existing_results:
            console.print(f"[bold green]✓ Found existing results for: {method}[/bold green]")
            existing_to_preserve[method] = existing_results[method]
        else:
            console.print(f"[bold yellow]⚡ Will run: {method}[/bold yellow]")
            methods_to_run.append(method)
    
    return methods_to_run, existing_to_preserve

def run_repeated_trials_experiment(
    task: str,
    sample_size: int = SAMPLE_SIZE,
    train_split: float = 0.2,
    base_seed: int = 42,
    num_trials: int = 3,
    target_accuracy: float = 0.9,
    skip_cache: bool = False,
    methods_config: list = None
) -> dict:
    """
    Run repeated trials experiment with different training samples.
    
    Args:
        task: Task name
        sample_size: Number of documents to sample for each trial
        train_split: Fraction of documents to use for training
        base_seed: Base random seed (each trial will use base_seed + trial_num)
        num_trials: Number of trials to run
        target_accuracy: Target accuracy for experiments
        skip_cache: Whether to skip cache
        methods_config: List of methods to run
        
    Returns:
        Dictionary containing all trial results and aggregated statistics
    """
    
    # Default methods if not specified - focused on core comparison
    if methods_config is None:
        methods_config = [
            "single_iteration_agent_greedy", 
            "single_iteration_agent_guaranteed",
            "baseline", 
            "baseline_with_guarantees"
        ]
    
    # Debug: Print the methods that will be run
    console.print(f"[bold blue]🔧 Methods configured for repeated trials:[/bold blue]")
    for i, method in enumerate(methods_config, 1):
        console.print(f"  {i}. {method}")
    console.print()
    
    # Create results directory
    repeated_results_dir = "results/repeated_trials"
    os.makedirs(repeated_results_dir, exist_ok=True)
    
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print(f"[bold cyan]🔄 REPEATED TRIALS EXPERIMENT - TASK: {task.upper()}[/bold cyan]")
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print(f"📊 Sample Size: {sample_size} | Training Split: {train_split} | Target Accuracy: {target_accuracy}")
    console.print(f"🎲 Number of Trials: {num_trials} | Base Seed: {base_seed}")
    console.print(f"🧪 Core Methods: Ours (Single-Shot 15), Ours (Guaranteed), Baseline (2-Model), Baseline (Guaranteed)")
    console.print(f"🎯 Focus: Core comparison between our method and baseline with/without guarantees")
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    
    # Load the original dataset once
    console.print("[bold magenta]🔮 Loading[/bold magenta]: Original dataset...")
    df, documents = load_dataset(task)
    
    # Use the first trial to establish the test set (keep it constant across all trials)
    console.print("[bold magenta]🔮 Establishing[/bold magenta]: Fixed test set for all trials...")
    _, test_df_fixed, _, _ = prepare_data(
        task, df, documents, sample_size, train_split, random_seed=base_seed
    )
    
    # Train filtering classifier once on the first trial's training data (reuse across all trials)
    console.print("[bold magenta]🔮 Training[/bold magenta]: Filtering classifier once for all trials...")
    train_df_first_trial, _, _, _ = prepare_data(
        task, df, documents, sample_size, train_split, random_seed=base_seed + 1
    )
    shared_classifier, shared_chunk_size = train_data_filtering(task, train_df_first_trial)
    
    # Process test data once with shared classifier (reuse across all trials)
    console.print("[bold blue]🔍 Processing[/bold blue]: Test data once with shared classifier...")
    test_df_chunked_fixed = chunk_and_get_confidences(test_df_fixed, shared_chunk_size, shared_classifier)
    
    # Calculate oracle cost once (same test set for all trials)
    oracle_only_cost = test_df_fixed.drop_duplicates(subset=["uuid"])["oracle_cost"].sum()
    
    # Storage for all trial results
    all_trial_results = []
    
    # Run each trial
    for trial_num in range(1, num_trials + 1):
        trial_seed = base_seed + trial_num
        console.print(f"\n[bold yellow]📋 TRIAL {trial_num}/{num_trials} (Seed: {trial_seed})[/bold yellow]")
        
        # Sample new training data for this trial (with replacement)
        console.print(f"[bold blue]🎲 Sampling[/bold blue]: New training data for trial {trial_num}...")
        train_df_trial, _, _, _ = prepare_data(
            task, df, documents, sample_size, train_split, random_seed=trial_seed
        )
        
        # Process training data with shared classifier (test data already processed)
        console.print(f"[bold blue]🔍 Processing[/bold blue]: Chunking training data for trial {trial_num}...")
        train_df_chunked_trial = chunk_and_get_confidences(train_df_trial, shared_chunk_size, shared_classifier)
        
        # Train filtering calibrator for this trial
        filtering_calibrator_trial = train_filtering_calibrator(train_df_chunked_trial, task)
        
        # Apply filtering calibrator
        train_df_filtered_trial = apply_filtering_calibrator_to_dataframe(train_df_chunked_trial, filtering_calibrator_trial)
        test_df_filtered_trial = apply_filtering_calibrator_to_dataframe(test_df_chunked_fixed, filtering_calibrator_trial)
        

        
        # Initialize results for this trial
        trial_results = {}
        
        # ===== BASELINE METHODS =====
        if "baseline" in methods_config or "baseline_with_guarantees" in methods_config:
            console.print(f"[bold yellow]📏 Trial {trial_num}[/bold yellow]: Running baseline methods...")
            baseline_results, baseline_results_guaranteed = train_and_apply_baseline_cascade(
                train_df_filtered_trial,
                test_df_filtered_trial,
                target_accuracy,
                task,
            )
            if "baseline" in methods_config:
                trial_results["baseline"] = baseline_results
                console.print(f"  ✓ Saved baseline results: accuracy={baseline_results['overall_accuracy']:.4f}")
            if "baseline_with_guarantees" in methods_config:
                trial_results["baseline_with_guarantees"] = baseline_results_guaranteed
                console.print(f"  ✓ Saved baseline_with_guarantees results: accuracy={baseline_results_guaranteed['overall_accuracy']:.4f}")
        

        
        # ===== SINGLE ITERATION AGENT METHODS =====
        single_iteration_needed = any(m in methods_config for m in ["single_iteration_agent_greedy", "single_iteration_agent_guaranteed"])
        if single_iteration_needed:
            console.print(f"[bold magenta]🤖 Trial {trial_num}[/bold magenta]: Single iteration agent...")
            
            # Run single iteration agent with guarantees if needed
            guarantee_accuracy = "single_iteration_agent_guaranteed" in methods_config
            console.print(f"  🔧 Guarantee accuracy enabled: {guarantee_accuracy}")
            single_iteration_cascade_results = find_surrogates(
                train_df_filtered_trial, 
                task, 
                target_accuracy, 
                num_iterations=1,
                provide_feedback=True,
                num_surrogate_requests=12,
                include_selectivity=False,
                guarantee_accuracy=guarantee_accuracy
            )
            
            if "single_iteration_agent_greedy" in methods_config:
                single_iteration_greedy_results = apply_cascade(
                    test_df_filtered_trial, 
                    single_iteration_cascade_results["greedy"]["ordering"], 
                    single_iteration_cascade_results["surrogate_to_prompt"], 
                    single_iteration_cascade_results["greedy"]["thresholds"], 
                    PROMPT_TO_TASK_TYPE_DICT[task]
                )
                trial_results["single_iteration_agent_greedy"] = single_iteration_greedy_results
                console.print(f"  ✓ Saved single_iteration_agent_greedy results: accuracy={single_iteration_greedy_results['overall_accuracy']:.4f}")
            
            if "single_iteration_agent_guaranteed" in methods_config:
                if "greedy_guaranteed" in single_iteration_cascade_results:
                    greedy_guaranteed_cfg = single_iteration_cascade_results["greedy_guaranteed"]
                    single_iteration_guaranteed_results = apply_cascade(
                        test_df_filtered_trial,
                        greedy_guaranteed_cfg["ordering"],
                        single_iteration_cascade_results["surrogate_to_prompt"],
                        greedy_guaranteed_cfg["thresholds"],
                        PROMPT_TO_TASK_TYPE_DICT[task]
                    )
                    trial_results["single_iteration_agent_guaranteed"] = single_iteration_guaranteed_results
                    console.print(f"  ✓ Saved single_iteration_agent_guaranteed results: accuracy={single_iteration_guaranteed_results['overall_accuracy']:.4f}")
                else:
                    console.print(f"  ❌ Warning: greedy_guaranteed not found in cascade results!")
                    console.print(f"  Available keys: {list(single_iteration_cascade_results.keys())}")
        

        
        # Add trial metadata
        trial_results["trial_metadata"] = {
            "trial_num": trial_num,
            "trial_seed": trial_seed,
            "target_accuracy": target_accuracy,
            "oracle_cost": oracle_only_cost,
            "task": task,
            "sample_size": sample_size,
            "train_split": train_split
        }
        
        # Save individual trial results
        trial_filename = f"{task}_trial_{trial_num}_seed_{trial_seed}_target_{target_accuracy}.pkl"
        trial_path = os.path.join(repeated_results_dir, trial_filename)
        with open(trial_path, 'wb') as f:
            pickle.dump(trial_results, f)
        
        all_trial_results.append(trial_results)
        
        # Print trial summary
        console.print(f"[bold green]✅ Trial {trial_num} completed[/bold green] - Results saved to {trial_filename}")
        for method, result in trial_results.items():
            if method != "trial_metadata" and isinstance(result, dict) and 'overall_accuracy' in result:
                accuracy_meets_target = "✓" if round(result['overall_accuracy'], 2) >= round(target_accuracy, 2) else "✗"
                console.print(f"  {method}: Accuracy {result['overall_accuracy']:.4f} {accuracy_meets_target}, Cost {result['total_cost']:.4f}")
    
    # ===== COMPUTE AGGREGATED STATISTICS =====
    console.print(f"\n[bold cyan]📊 COMPUTING AGGREGATED STATISTICS[/bold cyan]")
    
    aggregated_stats = {
        "task": task,
        "target_accuracy": target_accuracy,
        "num_trials": num_trials,
        "base_seed": base_seed,
        "sample_size": sample_size,
        "train_split": train_split,
        "oracle_cost": oracle_only_cost,
        "methods": {}
    }
    
    # Get all method names (excluding metadata)
    all_methods = set()
    for trial_result in all_trial_results:
        all_methods.update(k for k in trial_result.keys() if k != "trial_metadata")
    
    for method in all_methods:
        method_accuracies = []
        method_costs = []
        meets_target_count = 0
        
        for trial_result in all_trial_results:
            if method in trial_result and isinstance(trial_result[method], dict):
                result = trial_result[method]
                if 'overall_accuracy' in result and 'total_cost' in result:
                    accuracy = result['overall_accuracy']
                    cost = result['total_cost']
                    method_accuracies.append(accuracy)
                    method_costs.append(cost)
                    
                    # Check if meets target (rounded to 2 decimal places)
                    if round(accuracy, 2) >= round(target_accuracy, 2):
                        meets_target_count += 1
        
        if method_accuracies:  # Only compute stats if we have data
            aggregated_stats["methods"][method] = {
                "accuracy_mean": np.mean(method_accuracies),
                "accuracy_std": np.std(method_accuracies),
                "accuracy_min": np.min(method_accuracies),
                "accuracy_max": np.max(method_accuracies),
                "cost_mean": np.mean(method_costs),
                "cost_std": np.std(method_costs),
                "cost_min": np.min(method_costs),
                "cost_max": np.max(method_costs),
                "meets_target_count": meets_target_count,
                "meets_target_rate": meets_target_count / len(method_accuracies),
                "num_trials": len(method_accuracies)
            }
    
    # Save aggregated results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    aggregated_filename = f"{task}_aggregated_results_target_{target_accuracy}_{timestamp}.pkl"
    aggregated_path = os.path.join(repeated_results_dir, aggregated_filename)
    with open(aggregated_path, 'wb') as f:
        pickle.dump({
            "aggregated_stats": aggregated_stats,
            "all_trial_results": all_trial_results
        }, f)
    
    # Also save a "latest" version
    latest_aggregated_filename = f"{task}_aggregated_results_target_{target_accuracy}_latest.pkl"
    latest_aggregated_path = os.path.join(repeated_results_dir, latest_aggregated_filename)
    with open(latest_aggregated_path, 'wb') as f:
        pickle.dump({
            "aggregated_stats": aggregated_stats,
            "all_trial_results": all_trial_results
        }, f)
    
    # ===== PRINT AGGREGATED RESULTS =====
    console.print(f"\n[bold green]📈 AGGREGATED RESULTS SUMMARY[/bold green]")
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Create summary table
    table = Table(title=f"Repeated Trials Summary: {task.upper()} (Target Accuracy: {target_accuracy}, {num_trials} trials)")
    table.add_column("Method", style="bold white")
    table.add_column("Accuracy\n(Mean ± Std)", justify="center")
    table.add_column("Cost\n(Mean ± Std)", justify="center")
    table.add_column("Meets Target\n(Count/Rate)", justify="center")
    table.add_column("Cost Reduction\nvs Oracle (%)", justify="center")
    
    for method, stats in aggregated_stats["methods"].items():
        accuracy_str = f"{stats['accuracy_mean']:.3f} ± {stats['accuracy_std']:.3f}"
        cost_str = f"{stats['cost_mean']:.3f} ± {stats['cost_std']:.3f}"
        meets_target_str = f"{stats['meets_target_count']}/{stats['num_trials']} ({stats['meets_target_rate']:.1%})"
        cost_reduction = (1 - stats['cost_mean'] / oracle_only_cost) * 100
        cost_reduction_str = f"{cost_reduction:.1f}%"
        
        table.add_row(
            method,
            accuracy_str,
            cost_str,
            meets_target_str,
            cost_reduction_str
        )
    
    console.print(table)
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    console.print(f"\n[bold blue]💾 Aggregated results saved to:[/bold blue]")
    console.print(f"  📁 [bold underline]{aggregated_path}[/bold underline]")
    console.print(f"  🔗 [bold underline]{latest_aggregated_path}[/bold underline] (latest)")
    
    return {
        "aggregated_stats": aggregated_stats,
        "all_trial_results": all_trial_results
    }

def run_varying_target_experiment(
    task: str,
    sample_size: int = SAMPLE_SIZE,
    train_split: float = 0.2,
    seed: int = 42,
    target_accuracies: list = None,
    skip_cache: bool = False
) -> dict:
    """
    Run varying target accuracy experiment with 5 core methods.
    
    Args:
        task: Task name
        sample_size: Number of documents to sample
        train_split: Fraction of documents to use for training
        seed: Random seed
        target_accuracies: List of target accuracies to test
        skip_cache: Whether to skip cache
        
    Returns:
        Dictionary containing all target accuracy results and aggregated statistics
    """
    
    # Default target accuracies if not specified
    if target_accuracies is None:
        target_accuracies = [0.75, 0.8, 0.85, 0.9, 0.95]
    
    # Fixed methods for this experiment
    methods_config = [
        "single_iteration_agent_greedy", 
        "single_iteration_agent_guaranteed",
        "baseline", 
        "baseline_with_guarantees",
        "oracle_only"
    ]
    
    # Create results directory
    varying_target_results_dir = "results/varying_target"
    os.makedirs(varying_target_results_dir, exist_ok=True)
    
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print(f"[bold cyan]🎯 VARYING TARGET ACCURACY EXPERIMENT - TASK: {task.upper()}[/bold cyan]")
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print(f"📊 Sample Size: {sample_size} | Training Split: {train_split} | Seed: {seed}")
    console.print(f"🎯 Target Accuracies: {target_accuracies}")
    console.print(f"🧪 Core Methods: Single-Shot 15, Single-Shot 15 (Guaranteed), 2-Model Baseline, 2-Model Baseline (Guaranteed), Oracle Only")
    console.print(f"🎯 Focus: Performance across different target accuracy levels")
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    
    # Load the dataset once
    console.print("[bold magenta]🔮 Loading[/bold magenta]: Dataset...")
    df, documents = load_dataset(task)
    
    # Prepare data once (fixed train/test split)
    console.print("[bold magenta]🔮 Preparing[/bold magenta]: Train/test split...")
    train_df, test_df, documents, train_indices = prepare_data(
        task, df, documents, sample_size, train_split, random_seed=seed
    )
    
    # Train filtering classifier once
    console.print("[bold magenta]🔮 Training[/bold magenta]: Filtering classifier...")
    classifier, chunk_size = train_data_filtering(task, train_df)
    
    # Process data once
    console.print("[bold blue]🔍 Processing[/bold blue]: Chunking documents...")
    train_df_chunked = chunk_and_get_confidences(train_df, chunk_size, classifier)
    test_df_chunked = chunk_and_get_confidences(test_df, chunk_size, classifier)
    
    # Train filtering calibrator once (we'll use the same one for all target accuracies)
    console.print("[bold magenta]🔮 Training[/bold magenta]: Filtering calibrator...")
    filtering_calibrator = train_filtering_calibrator(train_df_chunked, task)
    
    # Apply filtering calibrator
    console.print("[bold blue]🔍 Processing[/bold blue]: Applying filtering calibrator...")
    train_df_filtered = apply_filtering_calibrator_to_dataframe(train_df_chunked, filtering_calibrator)
    test_df_filtered = apply_filtering_calibrator_to_dataframe(test_df_chunked, filtering_calibrator)
    
    # Calculate oracle cost once
    oracle_only_cost = test_df.drop_duplicates(subset=["uuid"])["oracle_cost"].sum()
    
    # Storage for all target accuracy results
    all_target_results = {}
    
    # Run each target accuracy
    for target_accuracy in target_accuracies:
        console.print(f"\n[bold yellow]📋 TARGET ACCURACY: {target_accuracy}[/bold yellow]")
        
        # Initialize results for this target accuracy
        target_results = {}
        
        # ===== BASELINE METHODS =====
        console.print(f"[bold yellow]📏 Target {target_accuracy}[/bold yellow]: Running baseline methods...")
        baseline_results, baseline_results_guaranteed = train_and_apply_baseline_cascade(
            train_df_filtered,
            test_df_filtered,
            target_accuracy,
            task,
        )
        target_results["baseline"] = baseline_results
        target_results["baseline_with_guarantees"] = baseline_results_guaranteed
        
        # ===== SINGLE ITERATION AGENT METHODS =====
        console.print(f"[bold magenta]🤖 Target {target_accuracy}[/bold magenta]: Single iteration agent with 15 surrogates...")
        
        # Run single iteration agent with 15 surrogates and guarantees
        single_iteration_cascade_results = find_surrogates(
            train_df_filtered, 
            task, 
            target_accuracy, 
            num_iterations=1,
            provide_feedback=False,
            num_surrogate_requests=15,  # Use 15 as requested
            include_selectivity=False,
            guarantee_accuracy=True  # Include guarantees
        )
        
        # Single iteration greedy
        single_iteration_greedy_results = apply_cascade(
            test_df_filtered, 
            single_iteration_cascade_results["greedy"]["ordering"], 
            single_iteration_cascade_results["surrogate_to_prompt"], 
            single_iteration_cascade_results["greedy"]["thresholds"], 
            PROMPT_TO_TASK_TYPE_DICT[task]
        )
        target_results["single_iteration_agent_greedy"] = single_iteration_greedy_results
        
        # Single iteration guaranteed
        greedy_guaranteed_cfg = single_iteration_cascade_results["greedy_guaranteed"]
        single_iteration_guaranteed_results = apply_cascade(
            test_df_filtered,
            greedy_guaranteed_cfg["ordering"],
            single_iteration_cascade_results["surrogate_to_prompt"],
            greedy_guaranteed_cfg["thresholds"],
            PROMPT_TO_TASK_TYPE_DICT[task]
        )
        target_results["single_iteration_agent_guaranteed"] = single_iteration_guaranteed_results
        
        # ===== ORACLE ONLY =====
        target_results["oracle_only"] = {
            "overall_accuracy": 1.0,
            "total_cost": oracle_only_cost,
            "runtime": 0.0
        }
        
        # Add target accuracy metadata
        target_results["target_metadata"] = {
            "target_accuracy": target_accuracy,
            "oracle_cost": oracle_only_cost,
            "task": task,
            "sample_size": sample_size,
            "train_split": train_split,
            "seed": seed
        }
        
        # Store results for this target accuracy
        all_target_results[target_accuracy] = target_results
        
        # Print target accuracy summary
        console.print(f"[bold green]✅ Target accuracy {target_accuracy} completed[/bold green]")
        for method, result in target_results.items():
            if method != "target_metadata" and isinstance(result, dict) and 'overall_accuracy' in result:
                accuracy_meets_target = "✓" if round(result['overall_accuracy'], 2) >= round(target_accuracy, 2) else "✗"
                cost_reduction = (1 - result['total_cost'] / oracle_only_cost) * 100
                console.print(f"  {method}: Accuracy {result['overall_accuracy']:.4f} {accuracy_meets_target}, Cost {result['total_cost']:.4f} ({cost_reduction:.1f}% reduction)")
    
    # ===== COMPUTE AGGREGATED STATISTICS =====
    console.print(f"\n[bold cyan]📊 COMPUTING AGGREGATED STATISTICS[/bold cyan]")
    
    aggregated_stats = {
        "task": task,
        "target_accuracies": target_accuracies,
        "seed": seed,
        "sample_size": sample_size,
        "train_split": train_split,
        "oracle_cost": oracle_only_cost,
        "methods": {}
    }
    
    # Get all method names (excluding metadata)
    all_methods = set()
    for target_result in all_target_results.values():
        all_methods.update(k for k in target_result.keys() if k != "target_metadata")
    
    for method in all_methods:
        method_data = {
            "target_accuracies": [],
            "achieved_accuracies": [],
            "costs": [],
            "cost_reductions": [],
            "meets_target_flags": []
        }
        
        for target_accuracy in target_accuracies:
            if target_accuracy in all_target_results and method in all_target_results[target_accuracy]:
                result = all_target_results[target_accuracy][method]
                if isinstance(result, dict) and 'overall_accuracy' in result and 'total_cost' in result:
                    accuracy = result['overall_accuracy']
                    cost = result['total_cost']
                    cost_reduction = (1 - cost / oracle_only_cost) * 100
                    meets_target = round(accuracy, 2) >= round(target_accuracy, 2)
                    
                    method_data["target_accuracies"].append(target_accuracy)
                    method_data["achieved_accuracies"].append(accuracy)
                    method_data["costs"].append(cost)
                    method_data["cost_reductions"].append(cost_reduction)
                    method_data["meets_target_flags"].append(meets_target)
        
        if method_data["achieved_accuracies"]:  # Only compute stats if we have data
            aggregated_stats["methods"][method] = method_data
    
    # Save all results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"{task}_varying_target_results_{timestamp}.pkl"
    results_path = os.path.join(varying_target_results_dir, results_filename)
    with open(results_path, 'wb') as f:
        pickle.dump({
            "aggregated_stats": aggregated_stats,
            "all_target_results": all_target_results
        }, f)
    
    # Also save a "latest" version
    latest_results_filename = f"{task}_varying_target_results_latest.pkl"
    latest_results_path = os.path.join(varying_target_results_dir, latest_results_filename)
    with open(latest_results_path, 'wb') as f:
        pickle.dump({
            "aggregated_stats": aggregated_stats,
            "all_target_results": all_target_results
        }, f)
    
    # ===== PRINT FINAL RESULTS SUMMARY =====
    console.print(f"\n[bold green]📈 VARYING TARGET ACCURACY RESULTS SUMMARY[/bold green]")
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Create summary table
    table = Table(title=f"Varying Target Accuracy Summary: {task.upper()}")
    table.add_column("Method", style="bold white")
    table.add_column("Target Range", justify="center")
    table.add_column("Avg Accuracy", justify="center")
    table.add_column("Avg Cost", justify="center")
    table.add_column("Avg Cost Reduction", justify="center")
    table.add_column("Success Rate", justify="center")
    
    for method, stats in aggregated_stats["methods"].items():
        if stats["achieved_accuracies"]:
            target_range = f"{min(stats['target_accuracies']):.2f}-{max(stats['target_accuracies']):.2f}"
            avg_accuracy = np.mean(stats["achieved_accuracies"])
            avg_cost = np.mean(stats["costs"])
            avg_cost_reduction = np.mean(stats["cost_reductions"])
            success_rate = np.mean(stats["meets_target_flags"])
            
            table.add_row(
                method,
                target_range,
                f"{avg_accuracy:.3f}",
                f"{avg_cost:.3f}",
                f"{avg_cost_reduction:.1f}%",
                f"{success_rate:.1%}"
            )
    
    console.print(table)
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    console.print(f"\n[bold blue]💾 Varying target accuracy results saved to:[/bold blue]")
    console.print(f"  📁 [bold underline]{results_path}[/bold underline]")
    console.print(f"  🔗 [bold underline]{latest_results_path}[/bold underline] (latest)")
    
    return {
        "aggregated_stats": aggregated_stats,
        "all_target_results": all_target_results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run comprehensive cascade experiments with ablations')
    parser.add_argument(
        '--task', 
        type=str, 
        required=True,
        choices=['game_review', 'legal_doc', 'enron', 'wiki_talk', 
                'court_opinion', 'screenplay', 'sms_spam', 'fever', 'ag_news', 'biodex', 'pubmed'],
        help='Task type to analyze'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=SAMPLE_SIZE,
        help='Number of documents to sample'
    )
    parser.add_argument(
        '--train_split',
        type=float,
        default=0.2,
        help='Fraction of documents to use for training'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for data splitting and reproducibility'
    )
    parser.add_argument(
        '--skip_cache',
        action='store_true',
        help='If set, skip loading from cache and recompute all data'
    )
    parser.add_argument(
        '--methods_config',
        type=str,
        default=None,
        help='Path to YAML file that specifies which experiment methods to run (under key "methods").'
    )
    parser.add_argument(
        '--rerun',
        action='store_true',
        help='If set, re-run all configured methods even if cached results exist.'
    )
    parser.add_argument(
        '--repeated_trials',
        action='store_true',
        help='If set, run repeated trials experiment (10 trials with different training samples for target accuracy 0.9).'
    )
    parser.add_argument(
        '--num_trials',
        type=int,
        default=3,
        help='Number of trials to run in repeated trials experiment (default: 10).'
    )
    parser.add_argument(
        '--varying_target',
        action='store_true',
        help='If set, run varying target accuracy experiment (0.75, 0.8, 0.85, 0.9, 0.95).'
    )
    args = parser.parse_args()
    
    # Determine which methods to run
    default_methods = [
        "main_greedy", "main_greedy_guaranteed", "main_selectivity",
        "no_data_filtering_greedy", "no_surrogate_greedy",
        "single_iteration_agent_greedy", "simple_similarity_filtering_greedy",
        "position_based_filtering_greedy", "simple_similarity_filtering_no_surrogate_greedy",
        "baseline", "oracle_only"
    ]

    if args.methods_config and os.path.exists(args.methods_config):
        with open(args.methods_config, "r") as f:
            config_yaml = yaml.safe_load(f) or {}
        configured_methods = [m for m, enabled in config_yaml.get("methods", {}).items() if enabled]
        if not configured_methods:
            configured_methods = default_methods
    else:
        configured_methods = default_methods

    # Check if running repeated trials experiment
    if args.repeated_trials:
        console.print(f"[bold cyan]🔄 Starting repeated trials experiment for task: {args.task}[/bold cyan]")
        
        # Use the default repeated trials methods (not the main experiment methods)
        # This ensures we get the guaranteed methods
        methods_for_trials = None  # Let the function use its own defaults
        
        # Run repeated trials experiment
        repeated_results = run_repeated_trials_experiment(
            task=args.task,
            sample_size=args.sample_size,
            train_split=args.train_split,
            base_seed=args.seed,
            num_trials=args.num_trials,
            target_accuracy=0.9,  # Fixed to 0.9 as requested
            skip_cache=args.skip_cache,
            methods_config=methods_for_trials  # None = use function defaults
        )
        
        console.print(f"[bold green]✅ Repeated trials experiment completed for {args.task}![/bold green]")
        exit(0)

    # Check if running varying target accuracy experiment
    if args.varying_target:
        console.print(f"[bold cyan]🎯 Starting varying target accuracy experiment for task: {args.task}[/bold cyan]")
        varying_target_results = run_varying_target_experiment(
            task=args.task,
            sample_size=args.sample_size,
            train_split=args.train_split,
            seed=args.seed,
            target_accuracies=[0.75, 0.8, 0.85, 0.9, 0.95], # Use the requested target accuracies
            skip_cache=args.skip_cache
        )
        console.print(f"[bold green]✅ Varying target accuracy experiment completed for {args.task}![/bold green]")
        exit(0)

    # Setup
    ensure_cache_dir()
    ensure_results_dir()
    cache_path = get_cache_path(args.task, args.sample_size, args.seed)
    
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print(f"[bold cyan]🔬 COMPREHENSIVE CASCADE EXPERIMENTS - TASK: {args.task.upper()}[/bold cyan]")
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print(f"📊 Sample Size: {args.sample_size} | Training Split: {args.train_split} | Seed: {args.seed}")
    console.print(f"🔧 Task Cascade Configuration: {config.NUM_ITERATIONS} iterations with {config.SURROGATES_PER_ITERATION} surrogates per iteration")
    console.print("🧪 Methods: Main Cascade (Greedy + Selectivity), Ablations (No Data Filtering, No Surrogate Tasks, Single Iteration Agent, Simple Similarity Filtering, Position-Based Filtering), Baseline")
    
    # Print configured methods
    console.print("\n[bold]Configured Methods:[/bold]")
    for method in configured_methods:
        console.print(f"✓ {method}")
        
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    
    # Try to load data from cache, otherwise prepare and save it
    if not args.skip_cache and os.path.exists(cache_path):
        console.print("[bold cyan]📂 Loading[/bold cyan]: Data from cache")
        cached_data = load_from_cache(cache_path)
        train_df = cached_data['train_df']
        test_df = cached_data['test_df']
        documents = cached_data['documents']
        train_indices = cached_data['train_indices']
    else:
        console.print("[bold magenta]🔮 Preparing[/bold magenta]: Loading dataset and preparing data...")
        df, documents = load_dataset(args.task)
        train_df, test_df, documents, train_indices = prepare_data(
            args.task, df, documents, args.sample_size, args.train_split, random_seed=args.seed
        )
        
        # Save to cache
        save_to_cache({
            'train_df': train_df,
            'test_df': test_df,
            'documents': documents,
            'train_indices': train_indices
        }, cache_path)
    
    # Load or train classifier (shared across all target accuracies)
    if not args.skip_cache and os.path.exists(get_classifier_path(args.task, args.seed)):
        console.print("[bold cyan]📂 Loading[/bold cyan]: Data filtering classifier from cache")
        classifier, chunk_size = load_classifier(args.task, args.seed)
    else:
        console.print("[bold magenta]🔮 Training[/bold magenta]: Training data filtering classifier...")
        classifier, chunk_size = train_data_filtering(args.task, train_df)
        save_classifier(args.task, args.seed, classifier, chunk_size)
        
    # Apply classifier to get chunks and confidences (shared across all target accuracies)
    chunked_cache_path = get_cache_path(args.task, args.sample_size, args.seed, suffix="_chunked")
    if not args.skip_cache and os.path.exists(chunked_cache_path):
        console.print("[bold cyan]📂 Loading[/bold cyan]: Chunked documents with confidences from cache")
        cached_chunked_data = load_from_cache(chunked_cache_path)
        train_df_chunked = cached_chunked_data['train_df']
        test_df_chunked = cached_chunked_data['test_df']
    else:
        console.print("[bold blue]🔍 Processing[/bold blue]: Chunking documents and calculating confidences...")
        train_df_chunked = chunk_and_get_confidences(train_df, chunk_size, classifier)
        test_df_chunked = chunk_and_get_confidences(test_df, chunk_size, classifier)
        save_to_cache({
            'train_df': train_df_chunked,
            'test_df': test_df_chunked
        }, chunked_cache_path)

    # ----- SHARED PRE-PROCESSING (reused for all target accuracies) -----
    filtering_calibrator_path_global = get_filtering_calibrator_path(args.task, args.seed, TARGET_ACCURACIES[0])
    if not args.skip_cache and os.path.exists(filtering_calibrator_path_global):
        console.print("[bold cyan]📂 Loading[/bold cyan]: Data filtering calibrator from cache (shared)")
        filtering_calibrator = load_filtering_calibrator(args.task, args.seed, TARGET_ACCURACIES[0])
    else:
        console.print("[bold magenta]🔮 Training[/bold magenta]: Data filtering calibrator (shared across targets)...")
        filtering_calibrator = train_filtering_calibrator(train_df_chunked, args.task)
        # save for backward-compatibility under the first target
        save_filtering_calibrator(args.task, args.seed, TARGET_ACCURACIES[0], filtering_calibrator)

    filtered_cache_path_global = get_cache_path(args.task, args.sample_size, args.seed, suffix="_filtered")
    if not args.skip_cache and os.path.exists(filtered_cache_path_global):
        console.print("[bold cyan]📂 Loading[/bold cyan]: Filtered dataframes from cache (shared)")
        _cached = load_from_cache(filtered_cache_path_global)
        train_df_filtered = _cached['train_df']
        test_df_filtered = _cached['test_df']
    else:
        console.print("[bold blue]🔍 Processing[/bold blue]: Applying filtering calibrator (shared across targets)...")
        train_df_filtered = apply_filtering_calibrator_to_dataframe(train_df_chunked, filtering_calibrator)
        test_df_filtered = apply_filtering_calibrator_to_dataframe(test_df_chunked, filtering_calibrator)
        save_to_cache({'train_df': train_df_filtered, 'test_df': test_df_filtered}, filtered_cache_path_global)

    # Versions with no filtering (used by ablation)
    train_df_no_filtering = create_no_data_filtering_df(train_df)
    test_df_no_filtering = create_no_data_filtering_df(test_df)

    # Determine once which extra ablations are needed
    simple_similarity_needed_global = any(m in configured_methods for m in [
        "simple_similarity_filtering_greedy",
        "simple_similarity_filtering_no_surrogate_greedy"
    ])
    position_based_needed_global = "position_based_filtering_greedy" in configured_methods

    # ----- SIMPLE SIMILARITY PREPROCESSING -----
    if simple_similarity_needed_global:
        console.print("[bold yellow]📐 Pre-computing simple similarity data filtering (shared)...")
        simple_classifier, simple_chunk_size = simple_similarity_data_filtering(args.task, train_df)

        simple_chunked_cache_path = get_cache_path(args.task, args.sample_size, args.seed, suffix="_simple_chunked")
        if not args.skip_cache and os.path.exists(simple_chunked_cache_path):
            _cached = load_from_cache(simple_chunked_cache_path)
            train_df_simple_chunked = _cached['train_df']
            test_df_simple_chunked = _cached['test_df']
        else:
            train_df_simple_chunked = chunk_and_get_confidences(train_df, simple_chunk_size, simple_classifier)
            test_df_simple_chunked = chunk_and_get_confidences(test_df, simple_chunk_size, simple_classifier)
            save_to_cache({'train_df': train_df_simple_chunked, 'test_df': test_df_simple_chunked}, simple_chunked_cache_path)

        simple_filtering_calibrator_path = get_cache_path(args.task, args.sample_size, args.seed, suffix="_simple_filtering_calibrator.pkl")
        if not args.skip_cache and os.path.exists(simple_filtering_calibrator_path):
            simple_filtering_calibrator = load_from_cache(simple_filtering_calibrator_path)
        else:
            simple_filtering_calibrator = train_filtering_calibrator(train_df_simple_chunked, args.task)
            save_to_cache(simple_filtering_calibrator, simple_filtering_calibrator_path)

        simple_filtered_cache_path = get_cache_path(args.task, args.sample_size, args.seed, suffix="_simple_filtered")
        if not args.skip_cache and os.path.exists(simple_filtered_cache_path):
            _cached = load_from_cache(simple_filtered_cache_path)
            train_df_simple_filtered = _cached['train_df']
            test_df_simple_filtered = _cached['test_df']
        else:
            train_df_simple_filtered = apply_filtering_calibrator_to_dataframe(train_df_simple_chunked, simple_filtering_calibrator)
            test_df_simple_filtered = apply_filtering_calibrator_to_dataframe(test_df_simple_chunked, simple_filtering_calibrator)
            save_to_cache({'train_df': train_df_simple_filtered, 'test_df': test_df_simple_filtered}, simple_filtered_cache_path)

    # ----- POSITION-BASED PREPROCESSING -----
    if position_based_needed_global:
        console.print("[bold red]📍 Pre-computing position-based data filtering (shared)...")
        position_classifier, position_chunk_size = position_based_data_filtering(args.task, train_df)

        position_chunked_cache_path = get_cache_path(args.task, args.sample_size, args.seed, suffix="_position_chunked")
        if not args.skip_cache and os.path.exists(position_chunked_cache_path):
            _cached = load_from_cache(position_chunked_cache_path)
            train_df_position_chunked = _cached['train_df']
            test_df_position_chunked = _cached['test_df']
        else:
            train_df_position_chunked = chunk_and_get_confidences(train_df, position_chunk_size, position_classifier)
            test_df_position_chunked = chunk_and_get_confidences(test_df, position_chunk_size, position_classifier)
            save_to_cache({'train_df': train_df_position_chunked, 'test_df': test_df_position_chunked}, position_chunked_cache_path)

        position_filtering_calibrator_path = get_cache_path(args.task, args.sample_size, args.seed, suffix="_position_filtering_calibrator.pkl")
        if not args.skip_cache and os.path.exists(position_filtering_calibrator_path):
            position_filtering_calibrator = load_from_cache(position_filtering_calibrator_path)
        else:
            position_filtering_calibrator = train_filtering_calibrator(train_df_position_chunked, args.task)
            save_to_cache(position_filtering_calibrator, position_filtering_calibrator_path)

        position_filtered_cache_path = get_cache_path(args.task, args.sample_size, args.seed, suffix="_position_filtered")
        if not args.skip_cache and os.path.exists(position_filtered_cache_path):
            _cached = load_from_cache(position_filtered_cache_path)
            train_df_position_filtered = _cached['train_df']
            test_df_position_filtered = _cached['test_df']
        else:
            train_df_position_filtered = apply_filtering_calibrator_to_dataframe(train_df_position_chunked, position_filtering_calibrator)
            test_df_position_filtered = apply_filtering_calibrator_to_dataframe(test_df_position_chunked, position_filtering_calibrator)
            save_to_cache({'train_df': train_df_position_filtered, 'test_df': test_df_position_filtered}, position_filtered_cache_path)

    # Get oracle-only cost for comparison (computed once)
    oracle_only_cost = test_df.drop_duplicates(subset=["uuid"])["oracle_cost"].sum()

    # ===== MAIN LOOP: Run experiments for each target accuracy =====
    for target_accuracy in TARGET_ACCURACIES:
        console.print(f"\n{'=' * 100}")
        console.print(f"[bold cyan]🎯 RUNNING EXPERIMENTS FOR TARGET ACCURACY: {target_accuracy}[/bold cyan]")
        console.print(f"{'=' * 100}")

        # Decide which methods need to be executed for this target
        if args.rerun:
            existing_results = {}
            methods_to_run = configured_methods.copy()
            existing_to_preserve = {}
            console.print("[bold red]⚠️  --rerun specified: ignoring cached method results and re-running all configured methods[/bold red]")
        else:
            existing_results = load_latest_experiment_results(args.task, args.seed, target_accuracy)
            methods_to_run, existing_to_preserve = determine_methods_to_run(configured_methods, existing_results)
        
        def should_run(method_key: str) -> bool:
            if args.rerun:
                return method_key in configured_methods
            # original logic
            if method_key in ["main_greedy", "main_selectivity", "main_greedy_guaranteed"]:
                return method_key in configured_methods
            return method_key in methods_to_run
        
        # Initialize results dictionary with existing results
        results = existing_to_preserve.copy()
        
        if not methods_to_run:
            console.print(f"[bold green]✅ All configured methods already completed for target accuracy {target_accuracy}![/bold green]")
            console.print("Skipping to results display...")
            # Skip data preparation and go directly to results display
        else:
            console.print(f"\n[bold yellow]📋 Methods to run for target accuracy {target_accuracy}:[/bold yellow]")
            for method in methods_to_run:
                console.print(f"  ⚡ {method}")

        # Only do data preparation if we have methods to run
        # Shared preprocessing done already; no additional per-target data preparation needed here

        # Results dictionary is already initialized above with existing results
        
        # ===== BASELINE METHODS =====
        if should_run("baseline") or True:
            console.print("\n[bold yellow]📏 BASELINE METHODS: Traditional Approaches[/bold yellow]")
            start_time = time.perf_counter()
            
            # Compute baseline cascade
            console.print("[bold magenta]🔮 Computing[/bold magenta]: Baseline cascade...")
            baseline_results, baseline_results_guaranteed = train_and_apply_baseline_cascade(
                train_df_filtered,
                test_df_filtered,
                target_accuracy,
                args.task,
            )
            baseline_results["runtime"] = time.perf_counter() - start_time
            results["baseline"] = baseline_results
            # Store guaranteed variant and propagate runtime as well
            baseline_results_guaranteed["runtime"] = baseline_results["runtime"]
            results["baseline_with_guarantees"] = baseline_results_guaranteed
            
            # Pretty print baseline results using rich, not as a table
            console.print(
                f"[bold cyan]Baseline cost:[/bold cyan] [green]{baseline_results['total_cost']:.4f}[/green]    "
                f"[bold cyan]Baseline accuracy:[/bold cyan] [yellow]{baseline_results['overall_accuracy']:.4f}[/yellow]"
            )
            console.print(
                f"[bold cyan]Baseline guaranteed cost:[/bold cyan] [green]{baseline_results_guaranteed['total_cost']:.4f}[/green]    "
                f"[bold cyan]Baseline guaranteed accuracy:[/bold cyan] [yellow]{baseline_results_guaranteed['overall_accuracy']:.4f}[/yellow]"
            )
    
        # ===== MAIN METHODS =====
        console.print(f"\n[bold green]🔄 MAIN METHODS: Full Pipeline with Surrogates + Data Filtering (Target: {target_accuracy})[/bold green]")
        
        # Load or find surrogates for main methods
        cascade_results_path = get_cascade_results_path(args.task, args.seed, target_accuracy)
        if not args.skip_cache and os.path.exists(cascade_results_path):
            console.print("[bold cyan]📂 Loading[/bold cyan]: Cascade results from cache")
            cascade_results = load_cascade_results(args.task, args.seed, target_accuracy)
        else:
            console.print("[bold magenta]🔮 Discovering[/bold magenta]: Finding surrogates for cascade...")
            cascade_results = find_surrogates(
                train_df_filtered, 
                args.task, 
                target_accuracy, 
                guarantee_accuracy=True, 
                num_iterations=config.NUM_ITERATIONS,  # 3 iterations
                num_surrogate_requests=config.SURROGATES_PER_ITERATION  # 5 surrogates per iteration
            )
            save_cascade_results(args.task, args.seed, target_accuracy, cascade_results)
    
        # Apply main cascades to test set
        if (should_run("main_greedy") or should_run("main_greedy_guaranteed") or should_run("main_selectivity")):
            console.print("[bold magenta]🔮 Testing[/bold magenta]: Applying main cascades to test set...")
            if should_run("main_greedy"):
                start_time = time.perf_counter()
                main_greedy_results = apply_cascade(
                    test_df_filtered,
                    cascade_results["greedy"]["ordering"],
                    cascade_results["surrogate_to_prompt"],
                    cascade_results["greedy"]["thresholds"],
                    PROMPT_TO_TASK_TYPE_DICT[args.task]
                )
                main_greedy_results["runtime"] = time.perf_counter() - start_time
                results["main_greedy"] = main_greedy_results

            if should_run("main_selectivity"):
                start_time = time.perf_counter()
                main_selectivity_results = apply_cascade(
                    test_df_filtered,
                    cascade_results["selectivity"]["ordering"],
                    cascade_results["surrogate_to_prompt"],
                    cascade_results["selectivity"]["thresholds"],
                    PROMPT_TO_TASK_TYPE_DICT[args.task]
                )
                main_selectivity_results["runtime"] = time.perf_counter() - start_time
                results["main_selectivity"] = main_selectivity_results
            
            if should_run("main_greedy_guaranteed"):
                start_time = time.perf_counter()
                greedy_guaranteed_cfg = cascade_results["greedy_guaranteed"]

                main_greedy_guaranteed_results = apply_cascade(
                    test_df_filtered,
                    greedy_guaranteed_cfg["ordering"],
                    cascade_results["surrogate_to_prompt"],
                    greedy_guaranteed_cfg["thresholds"],
                    PROMPT_TO_TASK_TYPE_DICT[args.task]
                )
                main_greedy_guaranteed_results["runtime"] = time.perf_counter() - start_time
                results["main_greedy_guaranteed"] = main_greedy_guaranteed_results
            
            # Print the main results that exist
            for method in ["main_greedy", "main_selectivity", "main_greedy_guaranteed"]:
                if method in results:
                    console.print(f"[bold cyan]{method}:[/bold cyan] [green]{results[method]['total_cost']:.4f}[/green]    "
                                  f"[bold cyan]Accuracy:[/bold cyan] [yellow]{results[method]['overall_accuracy']:.4f}[/yellow]")
    
        # ===== ABLATION 1: NO DATA FILTERING =====
        if should_run("no_data_filtering_greedy"):
            console.print("\n[bold orange1]📄 ABLATION 1: No Data Filtering (Surrogates on Full Documents)[/bold orange1]")
            start_time = time.perf_counter()
            
            # Using shared no-filtering DataFrames
            console.print("[bold magenta]🔮 Discovering[/bold magenta]: Finding surrogates without data filtering...")
            no_filtering_cascade_results = find_surrogates(
                train_df_no_filtering, 
                args.task, 
                target_accuracy, 
                include_selectivity=False, 
                num_iterations=config.NUM_ITERATIONS,  # 3 iterations
                num_surrogate_requests=config.SURROGATES_PER_ITERATION  # 5 surrogates per iteration
            )
            
            # Apply no-filtering cascade to test set
            console.print("[bold magenta]🔮 Testing[/bold magenta]: Applying no-filtering cascade to test set...")
            no_filtering_greedy_results = apply_cascade(
                test_df_no_filtering, 
                no_filtering_cascade_results["greedy"]["ordering"], 
                no_filtering_cascade_results["surrogate_to_prompt"], 
                no_filtering_cascade_results["greedy"]["thresholds"], 
                PROMPT_TO_TASK_TYPE_DICT[args.task]
            )
            no_filtering_greedy_results["runtime"] = time.perf_counter() - start_time
            results["no_data_filtering_greedy"] = no_filtering_greedy_results
    
        # ===== ABLATION 2: NO SURROGATE TASKS =====
        if should_run("no_surrogate_greedy"):
            console.print("\n[bold cyan]🎯 ABLATION 2: No Surrogate Tasks (Only Baseline Task with Data Filtering)[/bold cyan]")
            start_time = time.perf_counter()
            
            # Create no-surrogate cascade
            console.print("[bold magenta]🔮 Creating[/bold magenta]: No-surrogate cascade...")
            no_surrogate_cascade_results = create_no_surrogate_cascade(train_df_filtered, args.task, target_accuracy)
            
            # Apply no-surrogate cascade to test set
            console.print("[bold magenta]🔮 Testing[/bold magenta]: Applying no-surrogate cascade to test set...")
            no_surrogate_greedy_results = apply_cascade(
                test_df_filtered, 
                no_surrogate_cascade_results["greedy"]["ordering"], 
                no_surrogate_cascade_results["surrogate_to_prompt"], 
                no_surrogate_cascade_results["greedy"]["thresholds"], 
                PROMPT_TO_TASK_TYPE_DICT[args.task]
            )
            no_surrogate_greedy_results["runtime"] = time.perf_counter() - start_time
            results["no_surrogate_greedy"] = no_surrogate_greedy_results
    
        # ===== ABLATION 3: SINGLE ITERATION AGENT =====
        if should_run("single_iteration_agent_greedy"):
            console.print("\n[bold magenta]🤖 ABLATION 3: Single Iteration Agent (15 Surrogates, No Feedback)[/bold magenta]")
            start_time = time.perf_counter()
            
            # Single iteration agent no feedback
            console.print("[bold magenta]🔮 Discovering[/bold magenta]: Finding surrogates with single iteration agent...")
            single_iteration_cascade_results = find_surrogates(
                train_df_filtered, 
                args.task, 
                target_accuracy, 
                num_iterations=1,
                provide_feedback=False,
                include_selectivity=False
            )
            
            # Apply single-iteration cascade to test set
            console.print("[bold magenta]🔮 Testing[/bold magenta]: Applying single-iteration cascade to test set...")
            single_iteration_greedy_results = apply_cascade(
                test_df_filtered, 
                single_iteration_cascade_results["greedy"]["ordering"], 
                single_iteration_cascade_results["surrogate_to_prompt"], 
                single_iteration_cascade_results["greedy"]["thresholds"], 
                PROMPT_TO_TASK_TYPE_DICT[args.task]
            )
            single_iteration_greedy_results["runtime"] = time.perf_counter() - start_time
            results["single_iteration_agent_greedy"] = single_iteration_greedy_results
    
        # Simple similarity data prepared globally earlier

        # ===== ABLATION 4: SIMPLE SIMILARITY FILTERING =====
        if should_run("simple_similarity_filtering_greedy"):
            console.print("\n[bold yellow]📐 ABLATION 4: Simple Similarity-Based Data Filtering[/bold yellow]")
            
            # Find surrogates with simple filtering (greedy only)
            console.print("[bold magenta]🔮 Discovering[/bold magenta]: Finding surrogates with simple filtering...")
            simple_filtering_cascade_results = find_surrogates(
                train_df_simple_filtered, 
                args.task, 
                target_accuracy, 
                include_selectivity=False, 
                num_iterations=config.NUM_ITERATIONS,  # 3 iterations
                num_surrogate_requests=config.SURROGATES_PER_ITERATION  # 5 surrogates per iteration
            )
        
            # Apply simple filtering cascade to test set
            console.print("[bold magenta]🔮 Testing[/bold magenta]: Applying simple filtering cascade to test set...")
            simple_filtering_greedy_results = apply_cascade(
                test_df_simple_filtered, 
                simple_filtering_cascade_results["greedy"]["ordering"], 
                simple_filtering_cascade_results["surrogate_to_prompt"], 
                simple_filtering_cascade_results["greedy"]["thresholds"], 
                PROMPT_TO_TASK_TYPE_DICT[args.task]
            )
            
            results["simple_similarity_filtering_greedy"] = simple_filtering_greedy_results
    
        # Position-based filtered data prepared globally earlier; reusing here

        # ===== ABLATION 5: POSITION-BASED DATA FILTERING =====
        if should_run("position_based_filtering_greedy"):
            console.print("\n[bold red]📍 ABLATION 5: Position-Based Data Filtering (No Embeddings)[/bold red]")
            
            # Find surrogates with position filtering (greedy only)
            console.print("[bold magenta]🔮 Discovering[/bold magenta]: Finding surrogates with position filtering...")
            position_filtering_cascade_results = find_surrogates(
                train_df_position_filtered, 
                args.task, 
                target_accuracy, 
                include_selectivity=False, 
                num_iterations=config.NUM_ITERATIONS,  # 3 iterations
                num_surrogate_requests=config.SURROGATES_PER_ITERATION  # 5 surrogates per iteration
            )
            
            # Apply position filtering cascade to test set
            console.print("[bold magenta]🔮 Testing[/bold magenta]: Applying position filtering cascade to test set...")
            position_filtering_greedy_results = apply_cascade(
                test_df_position_filtered, 
                position_filtering_cascade_results["greedy"]["ordering"], 
                position_filtering_cascade_results["surrogate_to_prompt"], 
                position_filtering_cascade_results["greedy"]["thresholds"], 
                PROMPT_TO_TASK_TYPE_DICT[args.task]
            )
            
            results["position_based_filtering_greedy"] = position_filtering_greedy_results
        
        # ===== ABLATION 6: SIMPLE SIMILARITY FILTERING + NO SURROGATE TASKS =====
        if should_run("simple_similarity_filtering_no_surrogate_greedy"):
            console.print("\n[bold yellow]📐🎯 ABLATION 6: Simple Similarity Filtering + No Surrogate Tasks[/bold yellow]")
            
            # Create no-surrogate cascade using simple filtered data
            console.print("[bold magenta]🔮 Creating[/bold magenta]: No-surrogate cascade with simple filtering...")
            simple_no_surrogate_cascade_results = create_no_surrogate_cascade(train_df_simple_filtered, args.task, target_accuracy)
        
            # Apply combined cascade to test set
            console.print("[bold magenta]🔮 Testing[/bold magenta]: Applying simple filtering + no surrogate cascade to test set...")
            simple_no_surrogate_greedy_results = apply_cascade(
                test_df_simple_filtered, 
                simple_no_surrogate_cascade_results["greedy"]["ordering"], 
                simple_no_surrogate_cascade_results["surrogate_to_prompt"], 
                simple_no_surrogate_cascade_results["greedy"]["thresholds"], 
                PROMPT_TO_TASK_TYPE_DICT[args.task]
            )
            
            results["simple_similarity_filtering_no_surrogate_greedy"] = simple_no_surrogate_greedy_results

        
        # ===== PRINT COMPREHENSIVE RESULTS =====
        print_comprehensive_results(results, oracle_only_cost, target_accuracy)
        
        # ===== SAVE RESULTS =====
        results_path, latest_path = save_full_experiment_results(
            args.task, args.sample_size, args.seed, target_accuracy, results, oracle_only_cost
        )
    
        console.print(f"[bold blue]💾 Results for target accuracy {target_accuracy} saved to:[/bold blue]")
        console.print(f"  📁 [bold underline]{results_path}[/bold underline]")
        console.print(f"  🔗 [bold underline]{latest_path}[/bold underline] (latest)")
        
    console.print(f"\n[bold green]✅ ALL EXPERIMENTS COMPLETED FOR ALL TARGET ACCURACIES: {TARGET_ACCURACIES}[/bold green]") 