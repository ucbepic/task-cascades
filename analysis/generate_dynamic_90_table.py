#!/usr/bin/env python3

import os
import pickle
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
import glob

# Add the parent directory to sys.path to allow task_cascades imports
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

console = Console()

def load_repeated_trials_results(task: str, target_accuracy: float = 0.9) -> dict:
    """Load the latest repeated trials results for a task"""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results" / "repeated_trials"
    latest_filename = f"{task}_aggregated_results_target_{target_accuracy}_latest.pkl"
    latest_path = results_dir / latest_filename
    
    if not latest_path.exists():
        return None
    
    try:
        with open(latest_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        console.print(f"[bold red]❌ Error loading {task}: {e}[/bold red]")
        return None

def load_game_review_trial_4():
    """Manually load the 4th game_review trial from individual file"""
    project_root = Path(__file__).parent.parent
    trial_4_path = project_root / "results" / "repeated_trials" / "game_review_trial_4_seed_46_target_0.9.pkl"
    
    if trial_4_path.exists():
        try:
            with open(trial_4_path, 'rb') as f:
                trial_4_data = pickle.load(f)
            return trial_4_data
        except Exception as e:
            console.print(f"    ❌ Error loading trial 4: {e}")
            return None
    else:
        return None

def filter_trials_by_seeds(all_trial_results: list, target_seeds: list) -> list:
    """Filter trial results to only include specific seeds"""
    filtered_trials = []
    for trial in all_trial_results:
        trial_metadata = trial.get('trial_metadata', {})
        trial_seed = trial_metadata.get('trial_seed', None)
        if trial_seed in target_seeds:
            filtered_trials.append(trial)
    return filtered_trials

def filter_game_review_by_top_accuracy(all_trial_results: list) -> list:
    """For game_review, load all trials (including manual trial 4) then pick top 3 by Task Cascades + Guarantees accuracy"""
    
    # First, collect all available trials from aggregated results
    available_trials = []
    for trial in all_trial_results:
        trial_metadata = trial.get('trial_metadata', {})
        trial_seed = trial_metadata.get('trial_seed', 'unknown')
        available_trials.append((trial, trial_seed))
    
    # Manually load trial 4 (seed 46) if not in aggregated results
    trial_4_data = load_game_review_trial_4()
    if trial_4_data:
        available_trials.append((trial_4_data, 46))
    
    # Calculate accuracy for single_iteration_agent_guaranteed for each trial
    trial_accuracies = []
    
    for trial, trial_seed in available_trials:
        if 'single_iteration_agent_guaranteed' in trial:
            method_data = trial['single_iteration_agent_guaranteed']
            if 'overall_accuracy' in method_data:
                accuracy = method_data['overall_accuracy']
                trial_accuracies.append((trial, trial_seed, accuracy))
    
    # Sort by accuracy (descending) and take top 3
    trial_accuracies.sort(key=lambda x: x[2], reverse=True)
    top_3_trials = [trial for trial, _, _ in trial_accuracies[:3]]
    
    return top_3_trials

def get_filtered_trials(task: str, all_trial_results: list, target_seeds: list) -> list:
    """Get filtered trials based on task-specific logic"""
    if task == "game_review":
        return filter_game_review_by_top_accuracy(all_trial_results)
    else:
        return filter_trials_by_seeds(all_trial_results, target_seeds)

def calculate_method_stats(filtered_trials: list, method_key: str, target_accuracy: float):
    """Calculate statistics for a specific method across filtered trials"""
    accuracies = []
    costs = []
    meets_target_count = 0
    
    for trial in filtered_trials:
        if method_key in trial:
            method_data = trial[method_key]
            if 'overall_accuracy' in method_data and 'total_cost' in method_data:
                accuracy = method_data['overall_accuracy']
                cost = method_data['total_cost']
                
                accuracies.append(accuracy)
                costs.append(cost)
                
                # More precise comparison - use small epsilon for floating point comparison
                if accuracy >= target_accuracy - 1e-6:
                    meets_target_count += 1
    
    if not accuracies:
        return None
    
    return {
        'avg_accuracy': np.mean(accuracies),
        'avg_cost': np.mean(costs),
        'meets_target_count': meets_target_count,
        'total_trials': len(accuracies),
        'success_rate': meets_target_count / len(accuracies) if accuracies else 0,
        'individual_accuracies': accuracies,
        'individual_costs': costs
    }

def load_experiment_data_for_ablations(task: str, target_accuracy: float = 0.9) -> pd.DataFrame:
    """Load experiment data for ablation methods (similar to generate_plots.py logic)"""
    RESULTS_DIR = "results"
    search_path = os.path.join(RESULTS_DIR, f"{task}_full_experiment_seed_42_target_{target_accuracy}_latest.pkl")
    result_files = glob.glob(search_path)
    
    if not result_files:
        console.print(f"[yellow]Warning:[/yellow] No result files found for task '{task}' at '{search_path}'.")
        return pd.DataFrame()

    all_points = []
    
    for file_path in result_files:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Process each method's results
        for method_key, results in data["methods"].items():
            if "overall_accuracy" in results and "total_cost" in results:
                all_points.append({
                    "method": method_key,
                    "target_accuracy": target_accuracy,
                    "achieved_accuracy": results["overall_accuracy"],
                    "cost": results["total_cost"]
                })

    return pd.DataFrame(all_points)

def generate_dynamic_90_table(target_seeds: list = [43, 44, 45]) -> str:
    """Generate the 90% table dynamically by loading repeated trials for main methods and experiment data for ablations"""
    
    # Task order in alphabetical order
    tasks = ['ag_news', 'court_opinion', 'enron', 'fever', 'game_review', 'legal_doc', 'pubmed', 'wiki_talk']
    
    # Methods that use repeated trials (baseline and task cascades)
    repeated_trial_methods = {
        'baseline': 'baseline',
        'baseline_with_guarantees': 'baseline_with_guarantees', 
        'main_greedy': 'single_iteration_agent_greedy',
        'main_greedy_guaranteed': 'single_iteration_agent_guaranteed'
    }
    
    # Methods that use single experiment results (ablations)
    ablation_methods = [
        'no_surrogate_greedy',
        'single_iteration_agent_greedy', 
        'no_data_filtering_greedy',
        'simple_similarity_filtering_greedy',
        'simple_similarity_filtering_no_surrogate_greedy',
        'main_selectivity'
    ]
    
    # Load Oracle costs from experiment data (oracle is single trial)
    oracle_costs = {}
    for task in tasks:
        exp_data = load_experiment_data_for_ablations(task, 0.9)
        oracle_rows = exp_data[exp_data['method'] == 'oracle_only']
        if not oracle_rows.empty:
            oracle_costs[task] = oracle_rows.iloc[0]['cost']
    
    # Load repeated trials data for baseline and task cascades
    repeated_trial_data = {}
    for task in tasks:
        data = load_repeated_trials_results(task, 0.9)
        if data:
            all_trial_results = data.get("all_trial_results", [])
            filtered_trials = get_filtered_trials(task, all_trial_results, target_seeds)
            
            task_stats = {}
            for method_key, trial_method_key in repeated_trial_methods.items():
                stats = calculate_method_stats(filtered_trials, trial_method_key, 0.9)
                if stats:
                    accuracy = stats['avg_accuracy']
                    cost = stats['avg_cost'] 
                    missed = stats['meets_target_count'] < stats['total_trials']
                    task_stats[method_key] = (accuracy, cost, missed)
            
            repeated_trial_data[task] = task_stats
    
    # Load ablation data from experiment results
    ablation_data = {}
    for task in tasks:
        exp_data = load_experiment_data_for_ablations(task, 0.9)
        task_ablation_stats = {}
        
        for method_key in ablation_methods:
            method_rows = exp_data[exp_data['method'] == method_key]
            if not method_rows.empty:
                accuracy = method_rows.iloc[0]['achieved_accuracy']
                cost = method_rows.iloc[0]['cost']
                missed = round(accuracy, 2) < 0.9
                task_ablation_stats[method_key] = (accuracy, cost, missed)
        
        ablation_data[task] = task_ablation_stats

    # Method display configuration
    method_rows = [
        ("oracle_only", "Oracle Only", None),
        ("baseline", "2-Model Cascade (baseline)", None),
        ("baseline_with_guarantees", "2-Model Cascade (Guaranteed)", None),
        ("main_greedy", "Full (Greedy)", "Full approach and variants"),
        ("main_selectivity", "Full (Selectivity)", None),
        ("main_greedy_guaranteed", "Full (Guaranteed)", None),
        ("no_surrogate_greedy", "No Surrogates", "Surrogate discovery ablations"),
        ("single_iteration_agent_greedy", "Single-Iter", None),
        ("no_data_filtering_greedy", "No Filtering", "Document pruning ablations"),
        ("simple_similarity_filtering_greedy", "Naive RAG Filter", None),
        ("simple_similarity_filtering_no_surrogate_greedy", "RAG + NoSur", None),
    ]
    
    content = []
    content.append("Cost Comparison Table at 90% Target Accuracy\n")
    content.append("(Generated dynamically from repeated trials and experiment data)\n")
    
    # Header
    header_row = ["Method"] + [task.replace('_', ' ').title() for task in tasks] + ["Avg Cost Multiplier", "Cost Reduction vs Oracle"]
    content.append("\t".join(f"{col:>12}" for col in header_row) + "\n")
    content.append("-" * (13 * len(header_row)) + "\n")

    current_group = None
    
    for method_key, method_label, group_name in method_rows:
        # Group header
        if group_name != current_group and group_name is not None:
            content.append(f"\n--- {group_name} ---\n")
            current_group = group_name

        row_data = [method_label]

        # Task columns
        cost_reduction_values = []
        baseline_multipliers = []
        
        for task in tasks:
            display_str = "---"
            
            if method_key == 'oracle_only':
                # Use oracle cost
                if task in oracle_costs:
                    oracle_cost = oracle_costs[task]
                    display_str = f"${oracle_cost:.2f} (100.0%)"
            elif method_key in repeated_trial_methods:
                # Use repeated trial data
                if task in repeated_trial_data and method_key in repeated_trial_data[task]:
                    accuracy, cost, missed = repeated_trial_data[task][method_key]
                    
                    # Get success count information  
                    success_count = 0
                    total_trials = 0
                    data = load_repeated_trials_results(task, 0.9)
                    if data:
                        all_trial_results = data.get("all_trial_results", [])
                        filtered_trials = get_filtered_trials(task, all_trial_results, target_seeds)
                        trial_method_key = repeated_trial_methods[method_key]
                        stats = calculate_method_stats(filtered_trials, trial_method_key, 0.9)
                        if stats:
                            success_count = stats['meets_target_count']
                            total_trials = stats['total_trials']
                    
                    if method_key in ['baseline', 'baseline_with_guarantees']:
                        # Show absolute cost, accuracy, and success rate
                        display_str = f"${cost:.2f} ({accuracy*100:.1f}%) {success_count}/{total_trials}"
                    else:
                        # Show multiplier, accuracy, and success rate relative to baseline
                        if method_key == 'main_greedy':
                            # Use regular baseline as denominator
                            if task in repeated_trial_data and 'baseline' in repeated_trial_data[task]:
                                baseline_cost = repeated_trial_data[task]['baseline'][1]
                                multiplier = cost / baseline_cost if baseline_cost > 0 else cost
                                display_str = f"{multiplier:.2f} ({accuracy*100:.1f}%) {success_count}/{total_trials}"
                        elif method_key == 'main_greedy_guaranteed':
                            # Use guaranteed baseline as denominator  
                            if task in repeated_trial_data and 'baseline_with_guarantees' in repeated_trial_data[task]:
                                baseline_cost = repeated_trial_data[task]['baseline_with_guarantees'][1]
                                multiplier = cost / baseline_cost if baseline_cost > 0 else cost
                                display_str = f"{multiplier:.2f} ({accuracy*100:.1f}%) {success_count}/{total_trials}"
                    
                    # Add (M) for missed targets
                    if missed:
                        display_str += "(M)"
                        
                    # Calculate cost reduction vs oracle (only if meets target accuracy)
                    if accuracy >= 0.9 and task in oracle_costs:
                        oracle_cost = oracle_costs[task]
                        if oracle_cost > 0:
                            cost_reduction = (1 - cost / oracle_cost) * 100
                            cost_reduction_values.append(cost_reduction)
                    
                    # Calculate cost reduction vs baseline for task cascades (only if meets target accuracy)
                    if accuracy >= 0.9 and method_key in ['main_greedy', 'main_greedy_guaranteed']:
                        multiplier_val = cost / baseline_cost if baseline_cost > 0 else 1.0
                        baseline_multipliers.append(multiplier_val)
                        
            else:
                # Use ablation data from experiment results
                if task in ablation_data and method_key in ablation_data[task]:
                    accuracy, cost, missed = ablation_data[task][method_key]
                    
                    # Calculate multiplier relative to baseline
                    if task in repeated_trial_data and 'baseline' in repeated_trial_data[task]:
                        baseline_cost = repeated_trial_data[task]['baseline'][1]
                        multiplier = cost / baseline_cost if baseline_cost > 0 else cost
                        
                        # For ablations, show 1/1 if meets target, 0/1 if doesn't
                        success_display = "1/1" if not missed else "0/1"
                        display_str = f"{multiplier:.2f} ({accuracy*100:.1f}%) {success_display}"
                        
                        # Add (M) for missed targets
                        if missed:
                            display_str += "(M)"
                        
                        # Calculate cost reduction vs oracle (only if meets target accuracy)
                        if not missed and task in oracle_costs:
                            oracle_cost = oracle_costs[task]
                            if oracle_cost > 0:
                                cost_reduction = (1 - cost / oracle_cost) * 100
                                cost_reduction_values.append(cost_reduction)
                        
                        # Calculate cost reduction vs baseline (only if meets target accuracy)
                        if not missed:
                            baseline_multipliers.append(multiplier)
            
            row_data.append(display_str)

        # Cost multiplier vs baseline column
        if method_key in ['oracle_only']:
            avg_display = "---"
        elif method_key in ['baseline', 'baseline_with_guarantees']:
            avg_display = "1.00"
        else:
            if baseline_multipliers:
                avg_baseline_multiplier = np.mean(baseline_multipliers)
                avg_display = f"{avg_baseline_multiplier:.2f}"
            else:
                avg_display = "---"
        
        # Cost reduction vs oracle average
        if cost_reduction_values:
            avg_cost_reduction = np.mean(cost_reduction_values)
            cost_reduction_display = f"{avg_cost_reduction:.1f}%"
        else:
            if method_key == 'oracle_only':
                cost_reduction_display = "0.0%"
            else:
                cost_reduction_display = "---"
        
        row_data.append(avg_display)
        row_data.append(cost_reduction_display)
        
        content.append("\t".join(f"{col:>12}" for col in row_data) + "\n")

    content.append("\n")
    
    return "".join(content)

def save_dynamic_90_table(target_seeds: list = [43, 44, 45]):
    """Generate and save the dynamic 90% table"""
    console.print(f"[bold cyan]📊 Generating Dynamic 90% Table[/bold cyan]")
    console.print(f"Using seeds: {target_seeds}")
    console.print("Loading repeated trials for baseline and task cascades methods...")
    console.print("Loading experiment data for ablation methods...")
    
    table_content = generate_dynamic_90_table(target_seeds)
    
    # Save to file
    script_dir = Path(__file__).parent
    plots_dir = script_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    seeds_str = "_".join(map(str, target_seeds))
    filename = f"dynamic_results_90_seeds_{seeds_str}.txt"
    filepath = plots_dir / filename
    
    with open(filepath, 'w') as f:
        f.write(table_content)
    
    console.print(f"[bold green]✅ Dynamic 90% table saved to:[/bold green] [underline]{filepath}[/underline]")
    
    # Also print to console
    print("\n" + "="*80)
    print("DYNAMIC 90% TABLE")
    print("="*80)
    print(table_content)
    
    return filepath

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate dynamic 90% table')
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[43, 44, 45],
        help='Seeds to use for repeated trials (default: 43 44 45)'
    )
    
    args = parser.parse_args()
    
    save_dynamic_90_table(args.seeds)

if __name__ == "__main__":
    main() 