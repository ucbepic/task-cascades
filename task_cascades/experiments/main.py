import argparse
import json
import logging
import os
from pathlib import Path
import pickle
import sys
from os.path import dirname, abspath
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# Add the parent directory to sys.path to allow task_cascades imports
root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)

from task_cascades.data.create_dfs import prepare_data, load_dataset, apply_filtering_calibrator_to_dataframe
from task_cascades.filtering.train_classifier_for_filtering import train_data_filtering
from task_cascades.filtering.data_filtering_utils import chunk_and_get_confidences
from task_cascades.filtering.calibrators import train_filtering_calibrator
from task_cascades.cascade.find_surrogates import find_surrogates
from task_cascades.cascade.apply_trained_cascade import apply_cascade, train_and_apply_baseline_cascade, apply_baseline_limit
from task_cascades.predictors.predictors import PROMPT_TO_TASK_TYPE_DICT

# Create rich console for pretty printing
console = Console()

SAMPLE_SIZE = 1000
TARGET_ACCURACY = 0.9
CACHE_DIR = "cache"
RESULTS_DIR = "results"

def ensure_cache_dir():
    """Ensure the cache directory exists"""
    os.makedirs(CACHE_DIR, exist_ok=True)

def ensure_results_dir():
    """Ensure the results directory exists"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

def get_cache_path(task: str, sample_size: int, seed: int, suffix: str = "") -> str:
    """Get the cache file path for a given task, sample size, and seed"""
    return os.path.join(CACHE_DIR, f"{task}_{sample_size}_seed_{seed}{suffix}_cache.pkl")


def get_filtering_calibrator_path(task: str, seed: int) -> str:
    """Get path to the filtering calibrator cache file"""
    ensure_cache_dir()
    return os.path.join(CACHE_DIR, f"{task}_seed_{seed}_filtering_calibrator.pkl")

def save_to_cache(data: dict, cache_path: str):
    """Save data to cache"""
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

def load_from_cache(cache_path: str) -> dict:
    """Load data from cache"""
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def save_filtering_calibrator(task: str, seed: int, filtering_calibrator):
    """Save filtering calibrator to cache"""
    with open(get_filtering_calibrator_path(task, seed), 'wb') as f:
        pickle.dump(filtering_calibrator, f)


def load_filtering_calibrator(task: str, seed: int):
    """Load filtering calibrator from cache"""
    with open(get_filtering_calibrator_path(task, seed), 'rb') as f:
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

def get_cascade_results_path(task: str, seed: int) -> str:
    """Get path to the cascade results cache file"""
    ensure_cache_dir()
    return os.path.join(CACHE_DIR, f"{task}_seed_{seed}_cascade_results.pkl")

def save_cascade_results(task: str, seed: int, cascade_results: dict):
    """Save cascade results to cache"""
    with open(get_cascade_results_path(task, seed), 'wb') as f:
        pickle.dump(cascade_results, f)

def load_cascade_results(task: str, seed: int) -> dict:
    """Load cascade results from cache"""
    with open(get_cascade_results_path(task, seed), 'rb') as f:
        return pickle.load(f)

def get_results_path(task: str, sample_size: int, seed: int) -> str:
    """Get the path to save experiment results"""
    ensure_results_dir()
    return os.path.join(RESULTS_DIR, f"{task}_{sample_size}_seed_{seed}_experiment_results.pkl")

def save_experiment_results(
    task: str, 
    sample_size: int, 
    seed: int,
    oracle_only_cost: float, 
    baseline_results: dict, 
    cascade_greedy_results: dict,
    cascade_selectivity_results: dict,
    baseline_limit_results: dict,
    surrogate_to_prompt: dict
):
    """Save experiment results to a pickle file for later analysis"""
    results_data = {
        "task": task,
        "sample_size": sample_size,
        "seed": seed,
        "oracle_only_cost": oracle_only_cost,
        "baseline": {
            "cost": baseline_results["total_cost"],
            "accuracy": baseline_results["overall_accuracy"],
            "stage_usage": baseline_results["stage_usage"]
        },
        "cascade_greedy": {
            "cost": cascade_greedy_results["total_cost"],
            "accuracy": cascade_greedy_results["overall_accuracy"],
            "stage_usage": cascade_greedy_results["stage_usage"]
        },
        "cascade_selectivity": {
            "cost": cascade_selectivity_results["total_cost"],
            "accuracy": cascade_selectivity_results["overall_accuracy"],
            "stage_usage": cascade_selectivity_results["stage_usage"]
        },
        "baseline_limit": {
            "cost": baseline_limit_results["total_cost"],
            "accuracy": baseline_limit_results["overall_accuracy"],
            "stage_usage": baseline_limit_results["stage_usage"]
        },
        "surrogate_to_prompt": surrogate_to_prompt,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    results_path = get_results_path(task, sample_size, seed)
    with open(results_path, 'wb') as f:
        pickle.dump(results_data, f)
    
    return results_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run predicate selection analysis')
    parser.add_argument(
        '--task', 
        type=str, 
        required=True,
        choices=['game_review', 'legal_doc', 'enron', 'wiki_talk', 
                'court_opinion', 'screenplay', 'sms_spam', 'fever', 'ag_news', 'biodex'],
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
    args = parser.parse_args()
    
    # Setup
    ensure_cache_dir()
    cache_path = get_cache_path(args.task, args.sample_size, args.seed)
    
    # Try to load data from cache, otherwise prepare and save it
    if not args.skip_cache and os.path.exists(cache_path):
        console.print("\n[bold cyan]📂 Loading[/bold cyan]: Data from cache", style="dim")
        console.print(f"  └─ [italic]{cache_path}[/italic]")
        cached_data = load_from_cache(cache_path)
        train_df = cached_data['train_df']
        test_df = cached_data['test_df']
        documents = cached_data['documents']
        train_indices = cached_data['train_indices']
    else:
        console.print("\n[bold magenta]🔮 Preparing[/bold magenta]: Loading dataset and preparing data...")
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
    
    # Print the accuracy of the baseline predictor on the train set
    console.print(f"Baseline predictor accuracy: {np.mean(train_df['label'] == train_df['baseline_prediction'])}")
    
    # Load or train classifier
    if not args.skip_cache and os.path.exists(get_classifier_path(args.task, args.seed)):
        console.print("\n[bold cyan]📂 Loading[/bold cyan]: Data filtering classifier from cache", style="dim")
        console.print(f"  └─ [italic]{get_classifier_path(args.task, args.seed)}[/italic]")
        classifier, chunk_size = load_classifier(args.task, args.seed)
    else:
        console.print("\n[bold magenta]🔮 Training[/bold magenta]: Training data filtering classifier from scratch")
        classifier, chunk_size = train_data_filtering(args.task, train_df)
        save_classifier(args.task, args.seed, classifier, chunk_size)
        
    # Apply classifier to both train and test sets to get the chunks
    chunked_cache_path = get_cache_path(args.task, args.sample_size, args.seed, suffix="_chunked")
    if not args.skip_cache and os.path.exists(chunked_cache_path):
        console.print("\n[bold cyan]📂 Loading[/bold cyan]: Chunked documents with confidences from cache", style="dim")
        console.print(f"  └─ [italic]{chunked_cache_path}[/italic]")
        cached_chunked_data = load_from_cache(chunked_cache_path)
        train_df = cached_chunked_data['train_df']
        test_df = cached_chunked_data['test_df']
    else:
        console.print("\n[bold blue]🔍 Processing[/bold blue]: Chunking documents and calculating confidences...")
        train_df = chunk_and_get_confidences(train_df, chunk_size, classifier)
        test_df = chunk_and_get_confidences(test_df, chunk_size, classifier)
        # Save chunked dataframes to cache
        save_to_cache({
            'train_df': train_df,
            'test_df': test_df
        }, chunked_cache_path)

    # Train or load filtering calibrator
    filtering_calibrator_path = get_filtering_calibrator_path(args.task, args.seed)
    filtering_calibrator_exists = os.path.exists(filtering_calibrator_path)
    if not args.skip_cache and filtering_calibrator_exists:
        console.print("\n[bold cyan]📂 Loading[/bold cyan]: Data filtering calibrator from cache", style="dim")
        console.print(f"  └─ [italic]{filtering_calibrator_path}[/italic]")
        filtering_calibrator = load_filtering_calibrator(args.task, args.seed)
    else:
        console.print("\n[bold magenta]🔮 Training[/bold magenta]: Training data filtering calibrator from scratch")
        filtering_calibrator = train_filtering_calibrator(train_df, args.task)
        save_filtering_calibrator(args.task, args.seed, filtering_calibrator)
        
    # Try to load filtered dataframes from cache, otherwise apply classifier and calibrator
    filtered_cache_path = get_cache_path(args.task, args.sample_size, args.seed, suffix="_filtered")
    if not args.skip_cache and os.path.exists(filtered_cache_path):
        console.print("\n[bold cyan]📂 Loading[/bold cyan]: Filtered dataframes from cache", style="dim")
        console.print(f"  └─ [italic]{filtered_cache_path}[/italic]")
        cached_filtered_data = load_from_cache(filtered_cache_path)
        train_df = cached_filtered_data['train_df']
        test_df = cached_filtered_data['test_df']
    else:
        # Apply the classifier and calibrator to the train and test sets
        console.print("\n[bold magenta]🔮 Processing[/bold magenta]: Applying data filtering classifier and calibrator to train and test sets...")
        train_df = apply_filtering_calibrator_to_dataframe(train_df, filtering_calibrator)
        test_df = apply_filtering_calibrator_to_dataframe(test_df, filtering_calibrator)
        # Save filtered dataframes to cache
        save_to_cache({
            'train_df': train_df,
            'test_df': test_df
        }, filtered_cache_path)

    # Load or find surrogates
    cascade_results_path = get_cascade_results_path(args.task, args.seed)
    if False and not args.skip_cache and os.path.exists(cascade_results_path):
        console.print("\n[bold cyan]📂 Loading[/bold cyan]: Cascade results from cache", style="dim")
        console.print(f"  └─ [italic]{cascade_results_path}[/italic]")
        cascade_results = load_cascade_results(args.task, args.seed)
    else:
        # Find surrogates!
        console.print("\n[bold magenta]🔮 Discovering[/bold magenta]: Finding surrogates for cascade...")
        cascade_results = find_surrogates(train_df, args.task, TARGET_ACCURACY)
        # Save the results to cache
        save_cascade_results(args.task, args.seed, cascade_results)
    
    # Extract both cascade methods
    cascade_results_greedy = cascade_results["greedy"]
    cascade_results_selectivity = cascade_results["selectivity"]
    surrogate_to_prompt = cascade_results["surrogate_to_prompt"]
    
    # Apply both cascades to test set
    console.print(f"\n[bold magenta]🔮 Processing[/bold magenta]: Applying greedy cascade of {len(cascade_results_greedy['ordering'])} surrogates to test set...")
    cascade_results_greedy_test = apply_cascade(test_df, cascade_results_greedy["ordering"], surrogate_to_prompt, cascade_results_greedy["thresholds"], PROMPT_TO_TASK_TYPE_DICT[args.task])

    console.print(f"\n[bold magenta]🔮 Processing[/bold magenta]: Applying selectivity cascade of {len(cascade_results_selectivity['ordering'])} surrogates to test set...")
    cascade_results_selectivity_test = apply_cascade(test_df, cascade_results_selectivity["ordering"], surrogate_to_prompt, cascade_results_selectivity["thresholds"], PROMPT_TO_TASK_TYPE_DICT[args.task])

    # Get the ordering and surrogate to prompt mapping from cascade results
    # ordering = cascade_results["ordering"]
    # thresholds = cascade_results["thresholds"]
    
    # Compute a baseline cascade
    console.print("\n[bold magenta]🔮 Processing[/bold magenta]: Computing baseline cascade for comparison...")
    baseline_cascade_results, baseline_cascade_results_guaranteed = train_and_apply_baseline_cascade(
        train_df,
        test_df,
        TARGET_ACCURACY,
        args.task,
    )

    # Compute baseline_limit cascade
    console.print("\n[bold magenta]🔮 Processing[/bold magenta]: Computing baseline limit cascade for comparison...")
    baseline_limit_results = apply_baseline_limit(test_df, TARGET_ACCURACY, args.task)

    # Get oracle-only cost
    oracle_only_cost = test_df.drop_duplicates(subset=["uuid"])["oracle_cost"].sum()
    
    # Print experiment setup information
    console.print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print(f"[bold cyan]📊 EXPERIMENT SETUP - TASK: {args.task.upper()} (SEED: {args.seed})[/bold cyan]")
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print(f"📊 [bold]Sample Size[/bold]: {args.sample_size} | [bold]Training Split[/bold]: {args.train_split} | [bold]Target Accuracy[/bold]: {TARGET_ACCURACY} | [bold]Seed[/bold]: {args.seed}")
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    
    # Create a table for results comparison
    table = Table(title=f"Model Performance Comparison")
    table.add_column("Model", style="bold white")
    table.add_column("Accuracy", justify="right")
    table.add_column("Total Cost", justify="right")
    table.add_column("Cost Reduction", justify="right")
    table.add_column("Relative Cost", justify="right")
    
    # Calculate cost metrics
    cascade_reduction = (1 - cascade_results_greedy_test['total_cost'] / oracle_only_cost) * 100
    baseline_reduction = (1 - baseline_cascade_results['total_cost'] / oracle_only_cost) * 100
    baseline_limit_reduction = (1 - baseline_limit_results['total_cost'] / oracle_only_cost) * 100
    cascade_relative = cascade_results_greedy_test['total_cost'] / oracle_only_cost
    baseline_relative = baseline_cascade_results['total_cost'] / oracle_only_cost
    baseline_limit_relative = baseline_limit_results['total_cost'] / oracle_only_cost
    
    # Calculate cost metrics for selectivity cascade
    cascade_selectivity_reduction = (1 - cascade_results_selectivity_test['total_cost'] / oracle_only_cost) * 100
    cascade_selectivity_relative = cascade_results_selectivity_test['total_cost'] / oracle_only_cost
    
    # Add the models to the table with their respective metrics
    table.add_row(
        "[bold green]Cascade Model (Greedy)[/bold green]", 
        f"{cascade_results_greedy_test['overall_accuracy']:.4f}",
        f"{cascade_results_greedy_test['total_cost']:.4f}",
        f"{cascade_reduction:.2f}%",
        f"{cascade_relative:.2f}x"
    )
    
    table.add_row(
        "[bold purple]Cascade Model (Selectivity)[/bold purple]", 
        f"{cascade_results_selectivity_test['overall_accuracy']:.4f}",
        f"{cascade_results_selectivity_test['total_cost']:.4f}",
        f"{cascade_selectivity_reduction:.2f}%",
        f"{cascade_selectivity_relative:.2f}x"
    )
    
    table.add_row(
        "[bold yellow]Baseline Model[/bold yellow]", 
        f"{baseline_cascade_results['overall_accuracy']:.4f}",
        f"{baseline_cascade_results['total_cost']:.4f}",
        f"{baseline_reduction:.2f}%",
        f"{baseline_relative:.2f}x"
    )
    
    table.add_row(
        "[bold blue]Baseline Limit Model[/bold blue]", 
        f"{baseline_limit_results['overall_accuracy']:.4f}",
        f"{baseline_limit_results['total_cost']:.4f}",
        f"{baseline_limit_reduction:.2f}%",
        f"{baseline_limit_relative:.2f}x"
    )
    
    table.add_row(
        "[bold red]Oracle Only[/bold red]", 
        "1.0000",
        f"{oracle_only_cost:.4f}",
        "0.00%",
        "1.00x"
    )
    
    # Print the main comparison table
    console.print("\n")
    console.print(table)
    
    # --- Helper functions for stage usage visualization ---
    def process_stage_usage(stage_usage):
        """
        Process stage usage data into a standardized format.
        Returns stage_groups (aggregated usage by stage) and model_info (model details for each stage)
        """
        if not isinstance(stage_usage, dict):
            return {}, {}
            
        # Simple case (baseline with string keys)
        if all(isinstance(k, str) for k in stage_usage.keys()):
            return stage_usage, {k: ("oracle" if k == "oracle" else "proxy", None) for k in stage_usage}
        
        # Complex case (cascade with tuple keys)
        stage_groups = {}
        model_info = {}
        
        for key, value in stage_usage.items():
            if isinstance(key, tuple) and len(key) >= 2:
                stage_name = key[0]
                model_name = key[1]
                doc_fraction = key[2] if len(key) > 2 else None
                
                if stage_name not in stage_groups:
                    stage_groups[stage_name] = 0
                    model_info[stage_name] = (model_name, doc_fraction)
                stage_groups[stage_name] += value
            elif isinstance(key, str):
                if key not in stage_groups:
                    stage_groups[key] = 0
                    model_info[key] = ("oracle", None)
                stage_groups[key] += value
                
        return stage_groups, model_info
    
    def get_ordered_stages(stage_groups, ordering=None):
        """Get stages in the specified order, with non-ordered stages at the end"""
        ordered_stages = []
        
        # First add stages that are in the ordering
        if ordering: # Non-baseline
            for stage in ordering:
                if stage in stage_groups and stage_groups[stage] >= 0:
                    ordered_stages.append((stage, stage_groups[stage]))
        
        # Then add any remaining stages not in the ordering
        for stage, pct in stage_groups.items():
            if pct >= 0 and (not ordering or stage not in ordering):
                ordered_stages.append((stage, pct))
                
        return ordered_stages
    
    def format_stage_for_display(stage, pct, model_info):
        """Format a stage's information for display"""
        if stage not in model_info:
            return f"{stage}: [bold]{pct*100:.1f}%[/bold]"
            
        model, doc_fraction = model_info[stage]
        doc_frac_display = f", doc_fraction={doc_fraction}" if doc_fraction is not None else ""
        return f"{stage} ({model}{doc_frac_display}): [bold]{pct*100:.1f}%[/bold]"
    
    def create_visual_bar(percentage, color="green"):
        """Create a visual bar representation of the percentage"""
        bar_width = 30  # characters
        filled = int(percentage * bar_width / 100)
        return f"[{color}]{'█' * filled}{'░' * (bar_width - filled)}[/{color}] {percentage:.1f}%"
    
    # --- Stage Usage Details Table ---
    usage_table = Table(title="Stage Usage Details")
    usage_table.add_column("Model", style="bold white")
    usage_table.add_column("Stage Distribution", style="dim", no_wrap=False)
    
    # Process and add cascade model stage usage
    cascade_stage_groups, cascade_model_info = process_stage_usage(cascade_results_greedy_test['stage_usage'])
    cascade_ordered_stages = get_ordered_stages(cascade_stage_groups, cascade_results_greedy["ordering"])
    
    cascade_display = "\n".join([
        format_stage_for_display(stage, pct, cascade_model_info) 
        for stage, pct in cascade_ordered_stages
    ])
    
    usage_table.add_row("[bold green]Cascade Model (Greedy)[/bold green]", cascade_display)
    
    # Process and add selectivity cascade model stage usage  
    cascade_selectivity_stage_groups, cascade_selectivity_model_info = process_stage_usage(cascade_results_selectivity_test['stage_usage'])
    cascade_selectivity_ordered_stages = get_ordered_stages(cascade_selectivity_stage_groups, cascade_results_selectivity["ordering"])
    
    cascade_selectivity_display = "\n".join([
        format_stage_for_display(stage, pct, cascade_selectivity_model_info) 
        for stage, pct in cascade_selectivity_ordered_stages
    ])
    
    usage_table.add_row("[bold purple]Cascade Model (Selectivity)[/bold purple]", cascade_selectivity_display)
    
    # Process and add baseline model stage usage
    baseline_stage_groups, baseline_model_info = process_stage_usage(baseline_cascade_results['stage_usage'])
    baseline_ordered_stages = get_ordered_stages(baseline_stage_groups)
    
    baseline_display = " | ".join([
        f"{stage}: {pct*100:.1f}%" for stage, pct in baseline_ordered_stages
    ])
    
    usage_table.add_row("[bold yellow]Baseline Model[/bold yellow]", baseline_display)
    
    # Process and add baseline limit model stage usage
    baseline_limit_stage_groups, baseline_limit_model_info = process_stage_usage(baseline_limit_results['stage_usage'])
    baseline_limit_ordered_stages = get_ordered_stages(baseline_limit_stage_groups)
    
    baseline_limit_display = " | ".join([
        f"{stage}: {pct*100:.1f}%" for stage, pct in baseline_limit_ordered_stages
    ])
    
    usage_table.add_row("[bold blue]Baseline Limit Model[/bold blue]", baseline_limit_display)
    
    # Add oracle-only row
    usage_table.add_row("[bold red]Oracle Only[/bold red]", "oracle: [bold]100.0%[/bold]")
    
    # Print the stage usage table
    console.print("\n")
    console.print(usage_table)
    
    # --- Visual Stage Distribution Breakdown ---
    console.print("\n[bold]📊 Stage Distribution Breakdown:[/bold]")
    
    # Cascade model visualization
    console.print("\n[bold green]🔄 Cascade Model (Greedy) - Stages in Cascade Order:[/bold green]")
    for stage, pct in cascade_ordered_stages:
        stage_display = format_stage_for_display(stage, pct, cascade_model_info).split(': [bold]')[0]
        console.print(f"  {stage_display}: {create_visual_bar(pct*100, 'green')}")
    
    # Selectivity cascade model visualization
    console.print("\n[bold purple]🔄 Cascade Model (Selectivity) - Stages in Cascade Order:[/bold purple]")
    for stage, pct in cascade_selectivity_ordered_stages:
        stage_display = format_stage_for_display(stage, pct, cascade_selectivity_model_info).split(': [bold]')[0]
        console.print(f"  {stage_display}: {create_visual_bar(pct*100, 'purple')}")
    
    # Baseline model visualization
    console.print("\n[bold yellow]📏 Baseline Model - Stage Distribution:[/bold yellow]")
    for stage, pct in baseline_ordered_stages:
        console.print(f"  {stage}: {create_visual_bar(pct*100, 'yellow')}")
    
    # Baseline limit model visualization
    console.print("\n[bold blue]📐 Baseline Limit Model - Stage Distribution:[/bold blue]")
    for stage, pct in baseline_limit_ordered_stages:
        console.print(f"  {stage}: {create_visual_bar(pct*100, 'blue')}")
    
    # --- Summary ---
    cascade_vs_baseline = (baseline_cascade_results['total_cost'] - cascade_results_greedy_test['total_cost']) / baseline_cascade_results['total_cost'] * 100
    cascade_vs_baseline_limit = (baseline_limit_results['total_cost'] - cascade_results_greedy_test['total_cost']) / baseline_limit_results['total_cost'] * 100
    
    cascade_selectivity_vs_baseline = (baseline_cascade_results['total_cost'] - cascade_results_selectivity_test['total_cost']) / baseline_cascade_results['total_cost'] * 100
    cascade_selectivity_vs_baseline_limit = (baseline_limit_results['total_cost'] - cascade_results_selectivity_test['total_cost']) / baseline_limit_results['total_cost'] * 100
    
    # Compare the two cascade methods
    greedy_vs_selectivity = (cascade_results_selectivity_test['total_cost'] - cascade_results_greedy_test['total_cost']) / cascade_results_selectivity_test['total_cost'] * 100
    
    console.print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[bold]SUMMARY:[/bold]")
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print(f"• 💰 Cascade model (Greedy) achieves [bold cyan]{cascade_reduction:.1f}%[/bold cyan] cost reduction compared to oracle-only approach")
    console.print(f"• 💰 Cascade model (Selectivity) achieves [bold purple]{cascade_selectivity_reduction:.1f}%[/bold purple] cost reduction compared to oracle-only approach")
    
    if greedy_vs_selectivity > 0:
        console.print(f"• 🏆 Greedy method is [bold green]{greedy_vs_selectivity:.1f}% more cost-effective[/bold green] than the selectivity method")
    elif greedy_vs_selectivity < 0:
        console.print(f"• 🏆 Selectivity method is [bold purple]{-greedy_vs_selectivity:.1f}% more cost-effective[/bold purple] than the greedy method")
    else:
        console.print(f"• 🤝 Both cascade methods achieve similar cost effectiveness")
    
    if cascade_vs_baseline > 0:
        console.print(f"• ✅ Cascade model (Greedy) is [bold green]{cascade_vs_baseline:.1f}% more cost-effective[/bold green] than the baseline model")
    else:
        console.print(f"• ⚠️ Baseline model is [bold yellow]{-cascade_vs_baseline:.1f}% more cost-effective[/bold yellow] than the cascade model (Greedy)")
    
    if cascade_selectivity_vs_baseline > 0:
        console.print(f"• ✅ Cascade model (Selectivity) is [bold purple]{cascade_selectivity_vs_baseline:.1f}% more cost-effective[/bold purple] than the baseline model")
    else:
        console.print(f"• ⚠️ Baseline model is [bold yellow]{-cascade_selectivity_vs_baseline:.1f}% more cost-effective[/bold yellow] than the cascade model (Selectivity)")
    
    if cascade_vs_baseline_limit > 0:
        console.print(f"• ✅ Cascade model (Greedy) is [bold green]{cascade_vs_baseline_limit:.1f}% more cost-effective[/bold green] than the baseline limit model")
    else:
        console.print(f"• ⚠️ Baseline limit model is [bold blue]{-cascade_vs_baseline_limit:.1f}% more cost-effective[/bold blue] than the cascade model (Greedy)")
        
    if cascade_selectivity_vs_baseline_limit > 0:
        console.print(f"• ✅ Cascade model (Selectivity) is [bold purple]{cascade_selectivity_vs_baseline_limit:.1f}% more cost-effective[/bold purple] than the baseline limit model")
    else:
        console.print(f"• ⚠️ Baseline limit model is [bold blue]{-cascade_selectivity_vs_baseline_limit:.1f}% more cost-effective[/bold blue] than the cascade model (Selectivity)")
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    
    # Save experiment results to file for later comparison across tasks
    results_path = save_experiment_results(
        args.task,
        args.sample_size,
        args.seed,
        oracle_only_cost,
        baseline_cascade_results,
        cascade_results_greedy_test,
        cascade_results_selectivity_test,
        baseline_limit_results,
        surrogate_to_prompt
    )
    console.print(f"[bold blue]💾 Results saved to:[/bold blue] [bold underline]{results_path}[/bold underline]")