#!/usr/bin/env python3
"""
Example script to run Task Cascades on a single task.

This script demonstrates how to use Task Cascades with minimal setup.
It runs only the core method (with and without guarantees) to generate
and evaluate cascades, making it suitable for testing and understanding
the system before running full experiments.

Usage:
    python task_cascades/experiments/example_run.py --task=game_review
    python task_cascades/experiments/example_run.py --task=legal_doc --sample_size=500
"""

import argparse
import os
import sys
import time
from pathlib import Path
from os.path import dirname, abspath

# Add the parent directory to sys.path to allow task_cascades imports
root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)

from rich.console import Console
from rich.table import Table

from task_cascades.config.config import ExperimentConfig
from task_cascades.utils.cache_manager import CacheManager
from task_cascades.data.create_dfs import prepare_data, load_dataset, apply_filtering_calibrator_to_dataframe
from task_cascades.filtering.train_classifier_for_filtering import train_data_filtering
from task_cascades.filtering.data_filtering_utils import chunk_and_get_confidences
from task_cascades.filtering.calibrators import train_filtering_calibrator
from task_cascades.cascade.find_surrogates import find_surrogates
from task_cascades.cascade.apply_trained_cascade import apply_cascade
from task_cascades.predictors.predictors import PROMPT_TO_TASK_TYPE_DICT

def main():
    parser = argparse.ArgumentParser(description='Run Task Cascades example on a single task')
    parser.add_argument(
        '--task', 
        type=str, 
        required=True,
        choices=['game_review', 'legal_doc', 'enron', 'wiki_talk', 
                'court_opinion', 'fever', 'ag_news', 'pubmed'],
        help='Task to evaluate'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=200,  # Smaller default for example
        help='Number of documents to sample (default: 200 for cost efficiency)'
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
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--target_accuracy',
        type=float,
        default=0.9,
        help='Target accuracy for the cascade'
    )
    parser.add_argument(
        '--skip_cache',
        action='store_true',
        help='Skip cache and recompute everything'
    )
    
    args = parser.parse_args()
    
    # Initialize components
    console = Console()
    config = ExperimentConfig()
    cache_manager = CacheManager(config.CACHE_DIR)
    config.ensure_directories()
    
    console.print("=" * 80)
    console.print(f"[bold cyan]Task Cascades Example Run[/bold cyan]")
    console.print("=" * 80)
    console.print(f"[bold]Task:[/bold] {args.task}")
    console.print(f"[bold]Sample Size:[/bold] {args.sample_size}")
    console.print(f"[bold]Target Accuracy:[/bold] {args.target_accuracy}")
    console.print(f"[bold]Configuration:[/bold] {config.NUM_ITERATIONS} iterations, {config.SURROGATES_PER_ITERATION} surrogates per iteration")
    console.print("=" * 80)
    
    # Step 1: Load and prepare data
    console.print("\n[bold blue]Step 1: Loading and preparing data...[/bold blue]")
    
    cache_path = cache_manager.get_cache_path(args.task, args.sample_size, args.seed)
    if not args.skip_cache and cache_manager.cache_exists(cache_path):
        console.print("Loading data from cache...")
        cached_data = cache_manager.load_from_cache(cache_path)
        train_df = cached_data['train_df']
        test_df = cached_data['test_df']
        documents = cached_data['documents']
        train_indices = cached_data['train_indices']
    else:
        console.print("Loading dataset and preparing train/test split...")
        df, documents = load_dataset(args.task)
        train_df, test_df, documents, train_indices = prepare_data(
            args.task, df, documents, args.sample_size, args.train_split, random_seed=args.seed
        )
        
        cache_manager.save_to_cache({
            'train_df': train_df,
            'test_df': test_df,
            'documents': documents,
            'train_indices': train_indices
        }, cache_path)
    
    console.print(f"✓ Loaded {len(train_df)} training samples, {len(test_df)} test samples")
    
    # Step 2: Train data filtering classifier
    console.print("\n[bold blue]Step 2: Training data filtering classifier...[/bold blue]")
    
    classifier_path = cache_manager.get_classifier_path(args.task, args.seed)
    if not args.skip_cache and cache_manager.cache_exists(classifier_path):
        console.print("Loading classifier from cache...")
        classifier, chunk_size = cache_manager.load_classifier(args.task, args.seed)
    else:
        console.print("Training new data filtering classifier...")
        classifier, chunk_size = train_data_filtering(args.task, train_df)
        cache_manager.save_classifier(args.task, args.seed, classifier, chunk_size)
    
    console.print(f"✓ Classifier ready (chunk size: {chunk_size})")
    
    # Step 3: Process data with classifier
    console.print("\n[bold blue]Step 3: Processing data with classifier...[/bold blue]")
    
    chunked_cache_path = cache_manager.get_cache_path(args.task, args.sample_size, args.seed, suffix="_chunked")
    if not args.skip_cache and cache_manager.cache_exists(chunked_cache_path):
        console.print("Loading chunked data from cache...")
        cached_chunked_data = cache_manager.load_from_cache(chunked_cache_path)
        train_df_chunked = cached_chunked_data['train_df']
        test_df_chunked = cached_chunked_data['test_df']
    else:
        console.print("Chunking documents and calculating confidences...")
        train_df_chunked = chunk_and_get_confidences(train_df, chunk_size, classifier)
        test_df_chunked = chunk_and_get_confidences(test_df, chunk_size, classifier)
        cache_manager.save_to_cache({
            'train_df': train_df_chunked,
            'test_df': test_df_chunked
        }, chunked_cache_path)
    
    console.print("✓ Data chunked and processed")
    
    # Step 4: Train filtering calibrator and apply filtering
    console.print("\n[bold blue]Step 4: Training filtering calibrator...[/bold blue]")
    
    filtering_calibrator_path = cache_manager.get_filtering_calibrator_path(args.task, args.seed, args.target_accuracy)
    if not args.skip_cache and cache_manager.cache_exists(filtering_calibrator_path):
        console.print("Loading filtering calibrator from cache...")
        filtering_calibrator = cache_manager.load_filtering_calibrator(args.task, args.seed, args.target_accuracy)
    else:
        console.print("Training filtering calibrator...")
        filtering_calibrator = train_filtering_calibrator(train_df_chunked, args.task)
        cache_manager.save_filtering_calibrator(args.task, args.seed, args.target_accuracy, filtering_calibrator)
    
    console.print("Applying filtering calibrator...")
    train_df_filtered = apply_filtering_calibrator_to_dataframe(train_df_chunked, filtering_calibrator)
    test_df_filtered = apply_filtering_calibrator_to_dataframe(test_df_chunked, filtering_calibrator)
    
    console.print("✓ Data filtering applied")
    
    # Step 5: Generate Task Cascades
    console.print("\n[bold blue]Step 5: Generating Task Cascades...[/bold blue]")
    
    cascade_results_path = cache_manager.get_cascade_results_path(args.task, args.seed, args.target_accuracy)
    if not args.skip_cache and cache_manager.cache_exists(cascade_results_path):
        console.print("Loading cascade results from cache...")
        cascade_results = cache_manager.load_cascade_results(args.task, args.seed, args.target_accuracy)
    else:
        console.print("Finding surrogates and designing cascades...")
        console.print(f"Using {config.NUM_ITERATIONS} iterations with {config.SURROGATES_PER_ITERATION} surrogates per iteration")
        
        start_time = time.perf_counter()
        cascade_results = find_surrogates(
            train_df_filtered, 
            args.task, 
            args.target_accuracy, 
            guarantee_accuracy=True,  # Generate both regular and guaranteed versions
            num_iterations=config.NUM_ITERATIONS,
            num_surrogate_requests=config.SURROGATES_PER_ITERATION
        )
        generation_time = time.perf_counter() - start_time
        
        cache_manager.save_cascade_results(args.task, args.seed, args.target_accuracy, cascade_results)
        console.print(f"✓ Cascades generated in {generation_time:.2f}s")
    
    # Step 6: Evaluate cascades
    console.print("\n[bold blue]Step 6: Evaluating Task Cascades...[/bold blue]")
    
    # Calculate oracle cost for comparison
    oracle_only_cost = test_df.drop_duplicates(subset=["uuid"])["oracle_cost"].sum()
    
    # Evaluate greedy cascade
    console.print("Evaluating greedy cascade...")
    start_time = time.perf_counter()
    greedy_results = apply_cascade(
        test_df_filtered,
        cascade_results["greedy"]["ordering"],
        cascade_results["surrogate_to_prompt"],
        cascade_results["greedy"]["thresholds"],
        PROMPT_TO_TASK_TYPE_DICT[args.task]
    )
    greedy_eval_time = time.perf_counter() - start_time
    
    # Evaluate guaranteed cascade
    console.print("Evaluating guaranteed cascade...")
    start_time = time.perf_counter()
    guaranteed_results = apply_cascade(
        test_df_filtered,
        cascade_results["greedy_guaranteed"]["ordering"],
        cascade_results["surrogate_to_prompt"],
        cascade_results["greedy_guaranteed"]["thresholds"],
        PROMPT_TO_TASK_TYPE_DICT[args.task]
    )
    guaranteed_eval_time = time.perf_counter() - start_time
    
    # Step 7: Display results
    console.print("\n[bold green]Step 7: Results[/bold green]")
    console.print("=" * 80)
    
    # Create results table
    table = Table(title=f"Task Cascades Results: {args.task} (Target Accuracy: {args.target_accuracy})")
    table.add_column("Method", style="bold white")
    table.add_column("Accuracy", justify="right")
    table.add_column("Total Cost", justify="right")
    table.add_column("Cost Reduction", justify="right")
    table.add_column("Eval Time (s)", justify="right")
    
    # Calculate metrics
    greedy_cost_reduction = (1 - greedy_results['total_cost'] / oracle_only_cost) * 100
    guaranteed_cost_reduction = (1 - guaranteed_results['total_cost'] / oracle_only_cost) * 100
    
    # Add rows
    table.add_row(
        "[bold green]Task Cascades (Greedy)[/bold green]",
        f"{greedy_results['overall_accuracy']:.4f}",
        f"{greedy_results['total_cost']:.4f}",
        f"{greedy_cost_reduction:.2f}%",
        f"{greedy_eval_time:.2f}"
    )
    
    table.add_row(
        "[bold green]Task Cascades (Guaranteed)[/bold green]",
        f"{guaranteed_results['overall_accuracy']:.4f}",
        f"{guaranteed_results['total_cost']:.4f}",
        f"{guaranteed_cost_reduction:.2f}%",
        f"{guaranteed_eval_time:.2f}"
    )
    
    table.add_row(
        "[bold red]Oracle Only[/bold red]",
        "1.0000",
        f"{oracle_only_cost:.4f}",
        "0.00%",
        "-"
    )
    
    console.print(table)
    
    # Additional insights
    console.print("\n[bold cyan]Key Insights:[/bold cyan]")
    accuracy_meets_target_greedy = "✓" if round(greedy_results['overall_accuracy'], 2) >= round(args.target_accuracy, 2) else "✗"
    accuracy_meets_target_guaranteed = "✓" if round(guaranteed_results['overall_accuracy'], 2) >= round(args.target_accuracy, 2) else "✗"
    
    console.print(f"• Greedy method meets target accuracy: {accuracy_meets_target_greedy}")
    console.print(f"• Guaranteed method meets target accuracy: {accuracy_meets_target_guaranteed}")
    console.print(f"• Greedy vs Guaranteed cost difference: {abs(greedy_results['total_cost'] - guaranteed_results['total_cost']):.4f}")
    
    if len(cascade_results["surrogate_to_prompt"]) > 1:
        num_surrogates = len(cascade_results["surrogate_to_prompt"]) - 1  # Subtract 1 for main task
        console.print(f"• Generated {num_surrogates} surrogate tasks")
    
    console.print("\n[bold blue]Cascade Structure:[/bold blue]")
    console.print("Greedy cascade ordering:", cascade_results["greedy"]["ordering"])
    console.print("Guaranteed cascade ordering:", cascade_results["greedy_guaranteed"]["ordering"])
    
    console.print("\n[bold blue]Surrogate Tasks:[/bold blue]")
    for surrogate_id, prompt in cascade_results["surrogate_to_prompt"].items():
        if surrogate_id != "s1":  # Skip main task
            console.print(f"• {surrogate_id}: {prompt[:100]}...")
    
    console.print("\n[bold green]✓ Task Cascades example run completed successfully![/bold green]")
    console.print("=" * 80)

if __name__ == "__main__":
    main()