#!/usr/bin/env python3

import os
import pickle
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add the parent directory to sys.path to allow task_cascades imports
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

console = Console()

def load_repeated_trials_results(task: str, target_accuracy: float = 0.9) -> dict:
    """Load the latest repeated trials results for a task"""
    # Results are in the project root's results directory
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results" / "repeated_trials"
    latest_filename = f"{task}_aggregated_results_target_{target_accuracy}_latest.pkl"
    latest_path = results_dir / latest_filename
    
    if not latest_path.exists():
        console.print(f"[bold red]❌ No results found for {task} at {latest_path}[/bold red]")
        return None
    
    with open(latest_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def print_task_results(task: str, data: dict):
    """Print results for a single task"""
    if not data:
        return
    
    aggregated_stats = data.get("aggregated_stats", {})
    all_trial_results = data.get("all_trial_results", [])
    
    # Get basic info
    num_trials = aggregated_stats.get("num_trials", len(all_trial_results))
    target_accuracy = aggregated_stats.get("target_accuracy", 0.9)
    oracle_cost = aggregated_stats.get("oracle_cost", 0)
    
    console.print(f"\n[bold cyan]📊 REPEATED TRIALS RESULTS: {task.upper()}[/bold cyan]")
    console.print(f"Target Accuracy: {target_accuracy} | Number of Trials: {num_trials}")
    console.print("═" * 120)
    
    # Create main results table
    table = Table(title=f"{task.upper()} - Repeated Trials Summary")
    table.add_column("Method", style="bold white", width=35)
    table.add_column("Accuracy\n(Mean ± Std)", justify="center", width=15)
    table.add_column("Cost\n(Mean ± Std)", justify="center", width=15)
    table.add_column("Min/Max\nAccuracy", justify="center", width=12)
    table.add_column("Min/Max\nCost", justify="center", width=12)
    table.add_column("Success\nRate", justify="center", width=10)
    table.add_column("Avg Cost\nReduction", justify="center", width=12)
    
    # Method display names
    method_display_names = {
        "single_iteration_agent_greedy": "🤖 Single-Shot 15 (Greedy)",
        "single_iteration_agent_guaranteed": "🤖✅ Single-Shot 15 (Guaranteed)",
        "baseline": "📏 2-Model Baseline",
        "baseline_with_guarantees": "📏✅ 2-Model Baseline (Guaranteed)"
    }
    
    # Sort methods for consistent display
    method_order = [
        "single_iteration_agent_greedy",
        "single_iteration_agent_guaranteed", 
        "baseline",
        "baseline_with_guarantees"
    ]
    
    methods_data = aggregated_stats.get("methods", {})
    
    for method in method_order:
        if method in methods_data:
            stats = methods_data[method]
            display_name = method_display_names.get(method, method)
            
            # Format statistics
            acc_mean = stats["accuracy_mean"]
            acc_std = stats["accuracy_std"]
            acc_min = stats["accuracy_min"]
            acc_max = stats["accuracy_max"]
            
            cost_mean = stats["cost_mean"]
            cost_std = stats["cost_std"]
            cost_min = stats["cost_min"]
            cost_max = stats["cost_max"]
            
            success_rate = stats["meets_target_rate"]
            cost_reduction = (1 - cost_mean / oracle_cost) * 100 if oracle_cost > 0 else 0
            
            # Add row to table
            table.add_row(
                display_name,
                f"{acc_mean:.3f} ± {acc_std:.3f}",
                f"{cost_mean:.3f} ± {cost_std:.3f}",
                f"{acc_min:.3f}/{acc_max:.3f}",
                f"{cost_min:.3f}/{cost_max:.3f}",
                f"{success_rate:.1%}",
                f"{cost_reduction:.1f}%"
            )
    
    console.print(table)
    
    # Print additional insights
    console.print(f"\n[bold]📈 INSIGHTS FOR {task.upper()}:[/bold]")
    
    if len(methods_data) >= 2:
        # Compare our method vs baseline
        if "single_iteration_agent_greedy" in methods_data and "baseline" in methods_data:
            our_cost = methods_data["single_iteration_agent_greedy"]["cost_mean"]
            baseline_cost = methods_data["baseline"]["cost_mean"]
            improvement = (baseline_cost - our_cost) / baseline_cost * 100
            
            if improvement > 0:
                console.print(f"• 🏆 Our method (Single-Shot 15) outperforms baseline by [bold green]{improvement:.1f}%[/bold green] cost reduction")
            else:
                console.print(f"• ⚠️  Baseline outperforms our method by [bold red]{-improvement:.1f}%[/bold red]")
        
        # Compare guaranteed vs non-guaranteed
        if "single_iteration_agent_greedy" in methods_data and "single_iteration_agent_guaranteed" in methods_data:
            greedy_cost = methods_data["single_iteration_agent_greedy"]["cost_mean"]
            guaranteed_cost = methods_data["single_iteration_agent_guaranteed"]["cost_mean"]
            overhead = (guaranteed_cost - greedy_cost) / greedy_cost * 100
            
            greedy_success = methods_data["single_iteration_agent_greedy"]["meets_target_rate"]
            guaranteed_success = methods_data["single_iteration_agent_guaranteed"]["meets_target_rate"]
            
            console.print(f"• 🛡️  Guaranteed version has [bold cyan]{overhead:.1f}%[/bold cyan] cost overhead")
            console.print(f"• 🎯 Success rate: Greedy {greedy_success:.1%} vs Guaranteed {guaranteed_success:.1%}")

def print_comparison_across_tasks(all_results: dict):
    """Print a comparison table across all tasks"""
    console.print(f"\n[bold cyan]🌍 CROSS-TASK COMPARISON[/bold cyan]")
    console.print("═" * 140)
    
    # Create cross-task comparison table
    table = Table(title="Method Performance Across All Tasks")
    table.add_column("Method", style="bold white", width=30)
    
    # Add columns for each task
    tasks = list(all_results.keys())
    for task in tasks:
        table.add_column(f"{task.replace('_', ' ').title()}\n(Cost Reduction)", justify="center", width=15)
    
    table.add_column("Average\nCost Reduction", justify="center", width=15)
    
    # Method display names
    method_display_names = {
        "single_iteration_agent_greedy": "🤖 Single-Shot 15 (Greedy)",
        "single_iteration_agent_guaranteed": "🤖✅ Single-Shot 15 (Guaranteed)",
        "baseline": "📏 2-Model Baseline", 
        "baseline_with_guarantees": "📏✅ 2-Model Baseline (Guaranteed)"
    }
    
    method_order = [
        "single_iteration_agent_greedy",
        "single_iteration_agent_guaranteed",
        "baseline", 
        "baseline_with_guarantees"
    ]
    
    for method in method_order:
        row_data = [method_display_names.get(method, method)]
        cost_reductions = []
        
        for task in tasks:
            if task in all_results and all_results[task]:
                aggregated_stats = all_results[task].get("aggregated_stats", {})
                methods_data = aggregated_stats.get("methods", {})
                oracle_cost = aggregated_stats.get("oracle_cost", 1)
                
                if method in methods_data:
                    cost_mean = methods_data[method]["cost_mean"]
                    cost_reduction = (1 - cost_mean / oracle_cost) * 100 if oracle_cost > 0 else 0
                    cost_reductions.append(cost_reduction)
                    row_data.append(f"{cost_reduction:.1f}%")
                else:
                    row_data.append("-")
            else:
                row_data.append("-")
        
        # Add average
        if cost_reductions:
            avg_reduction = np.mean(cost_reductions)
            row_data.append(f"{avg_reduction:.1f}%")
        else:
            row_data.append("-")
        
        table.add_row(*row_data)
    
    console.print(table)

def create_stability_box_plots(all_results: dict, output_path: str = None):
    """Create box plots showing absolute cost stability across repeated trials"""
    tasks = list(all_results.keys())
    
    if not tasks:
        console.print("[bold red]❌ No results available for plotting[/bold red]")
        return
    
    # Set up the plot - single row for paper presentation
    n_tasks = len(tasks)
    
    # Configure matplotlib for better paper presentation
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'DejaVu Sans',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    fig, axes = plt.subplots(1, n_tasks, figsize=(3.5 * n_tasks + 2, 4))
    
    # Ensure axes is always a 1D array
    if n_tasks == 1:
        axes = [axes]
    
    # Method configurations: (method_key, color, label)
    # Focus on core comparison without guaranteed methods
    method_configs = [
        ("baseline", "blue", "2-Model Cascade"),
        ("single_iteration_agent_greedy", "green", "Task Cascade")
    ]
    
    for i, task in enumerate(tasks):
        ax = axes[i]
        
        data = all_results[task]
        if not data:
            # Add group labels to titles even when no data
            if task in ["enron", "legal_doc"]:
                group_label = "(A)"
            elif task in ["game_review", "court_opinion"]:
                group_label = "(B)"
            else:  # ag_news
                group_label = "(C)"
            
            ax.text(0.5, 0.5, f'No data for {task}', ha='center', va='center', transform=ax.transAxes)
            title = f"{task.replace('_', ' ').title()} {group_label}"
            ax.set_title(title)
            continue
        
        aggregated_stats = data.get("aggregated_stats", {})
        all_trial_results = data.get("all_trial_results", [])
        oracle_cost = aggregated_stats.get("oracle_cost", 1.0)
        methods_data = aggregated_stats.get("methods", {})
        
        # Prepare data for box plots
        box_data = []
        box_labels = []
        box_colors = []
        success_rates = []
        
        target_accuracy = aggregated_stats.get("target_accuracy", 0.9)
        
        for method_key, color, label in method_configs:
            if method_key in methods_data:
                # Extract absolute costs from individual trials
                absolute_costs = []
                meets_target_count = 0
                total_trials = 0
                
                for trial_result in all_trial_results:
                    if method_key in trial_result and 'total_cost' in trial_result[method_key]:
                        trial_cost = trial_result[method_key]['total_cost']
                        absolute_costs.append(trial_cost)
                        
                        # Check if this trial met the target accuracy
                        if 'overall_accuracy' in trial_result[method_key]:
                            trial_accuracy = trial_result[method_key]['overall_accuracy']
                            total_trials += 1
                            if round(trial_accuracy, 2) >= round(target_accuracy, 2):
                                meets_target_count += 1
                
                if absolute_costs:
                    box_data.append(absolute_costs)
                    box_labels.append(label)
                    box_colors.append(color)
                    success_rates.append((meets_target_count, total_trials))
        
        if box_data:
            # Create box plot without labels (we'll use legend instead) - wider boxes
            bp = ax.boxplot(box_data, patch_artist=True, 
                           showmeans=True, meanline=True, widths=0.6)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Style the box plot elements - thicker lines for better visibility
            for element in ['whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bp[element], color='black', linewidth=1.5)
            plt.setp(bp['means'], color='red', linewidth=2.5)
            
            # Add annotations for each method
            for idx, (meets_target, total_trials) in enumerate(success_rates):
                x_pos = idx + 1  # Box plot positions start at 1
                method_key = [k for k, _, _ in method_configs][idx]
                
                # Calculate average accuracy across all trials
                accuracies = []
                for trial_result in all_trial_results:
                    if method_key in trial_result and 'overall_accuracy' in trial_result[method_key]:
                        trial_accuracy = trial_result[method_key]['overall_accuracy']
                        accuracies.append(trial_accuracy)
                
                if accuracies:
                    avg_accuracy = np.mean(accuracies)
                    avg_text = f"avg acc: {avg_accuracy:.2f}"
                    ax.text(x_pos, -0.08, avg_text, 
                           ha='center', va='top', fontsize=11,
                           transform=ax.get_xaxis_transform())
                
                # Add p50 miss annotation for methods with < 100% success, or "no misses" for 100% success
                if meets_target < total_trials:
                    # Calculate median deviation from target accuracy for failed trials
                    deviations = []
                    for trial_result in all_trial_results:
                        if method_key in trial_result and 'overall_accuracy' in trial_result[method_key]:
                            trial_accuracy = trial_result[method_key]['overall_accuracy']
                            if round(trial_accuracy, 2) < round(target_accuracy, 2):
                                deviation = target_accuracy - trial_accuracy
                                deviations.append(deviation)
                    
                    if deviations:
                        median_deviation = np.median(deviations)
                        deviation_text = f"p50 miss: -{median_deviation:.3f}"
                        ax.text(x_pos, -0.18, deviation_text, 
                               ha='center', va='top', fontsize=11, style='italic',
                               transform=ax.get_xaxis_transform())
                else:
                    # 100% success rate
                    ax.text(x_pos, -0.18, "no misses", 
                           ha='center', va='top', fontsize=11, style='italic',
                           transform=ax.get_xaxis_transform())
        
        # Customize subplot with group labels
        ax.set_ylabel('Total Cost')
        
        # Add group labels to titles
        if task in ["enron", "legal_doc"]:
            group_label = "(A)"
        elif task in ["game_review", "court_opinion"]:
            group_label = "(B)"
        else:  # ag_news
            group_label = "(C)"
        
        title = f"{task.replace('_', ' ').title()} {group_label}"
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Remove x-axis tick labels to save space
        ax.set_xticklabels([])
        
        # Set y-axis to start exactly at 0
        if box_data:
            all_values = [val for sublist in box_data for val in sublist]
            if all_values:
                max_val = max(all_values)
                margin = max_val * 0.1 if max_val > 0 else 0.1
                # Y-axis starts exactly at 0
                ax.set_ylim(0, max_val + margin)
    
    # Create legend - collect all methods that appear in any task
    legend_elements = []
    used_methods = set()
    
    # First, find all methods that actually appear in the data
    for task in tasks:
        if all_results[task]:
            methods_data = all_results[task].get("aggregated_stats", {}).get("methods", {})
            used_methods.update(methods_data.keys())
    
    # Create legend elements for methods that exist in data
    for method_key, color, label in method_configs:
        if method_key in used_methods:
            legend_elements.append(
                mpatches.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7, 
                                 edgecolor='black', linewidth=0.5, label=label)
            )
    
    # Add title with proper spacing and centering
    fig.suptitle('Cost Stability Across Repeated Trials', fontsize=16, y=0.95)
    
    # Place legend next to the title at the top
    if legend_elements:
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.95), 
                   ncol=2, frameon=False, fontsize=11)
        console.print(f"[bold blue]📋 Created legend with {len(legend_elements)} methods[/bold blue]")
    else:
        console.print(f"[bold yellow]⚠️  No legend elements created. Used methods: {used_methods}[/bold yellow]")
    
    # Adjust layout with space for bottom annotations and top legend
    plt.tight_layout(rect=[0, 0.15, 1.0, 0.92])
    plt.subplots_adjust(wspace=0.3)
    
    # Save plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        console.print(f"[bold green]📊 Plot saved to: {output_path}[/bold green]")
    else:
        # Default save location
        script_dir = Path(__file__).parent
        plots_dir = script_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        default_path = plots_dir / "repeated_trials_cost_stability_box_plots.pdf"
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        console.print(f"[bold green]📊 Plot saved to: {default_path}[/bold green]")

def main():
    parser = argparse.ArgumentParser(description='Print repeated trials results in table format')
    parser.add_argument(
        '--task',
        type=str,
        help='Specific task to display (if not provided, shows all available tasks)'
    )
    parser.add_argument(
        '--target_accuracy',
        type=float,
        default=0.9,
        help='Target accuracy for results (default: 0.9)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate stability box plots for all tasks'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for the plot (if not provided, saves to plots directory)'
    )
    
    args = parser.parse_args()
    
    # Available tasks (same order as varying target results: AA BB C)
    available_tasks = ["enron", "legal_doc", "game_review", "court_opinion", "ag_news"]
    
    if args.task:
        # Show results for specific task
        if args.task not in available_tasks:
            console.print(f"[bold red]❌ Task '{args.task}' not in available tasks: {available_tasks}[/bold red]")
            return
        
        data = load_repeated_trials_results(args.task, args.target_accuracy)
        print_task_results(args.task, data)
        
        if args.plot:
            console.print("[bold yellow]⚠️  Plot option requires multiple tasks. Please run without --task to generate plots.[/bold yellow]")
    else:
        # Show results for all tasks
        all_results = {}
        
        console.print("[bold green]🔍 Loading repeated trials results for all tasks...[/bold green]")
        
        for task in available_tasks:
            console.print(f"  📂 Loading {task}...")
            data = load_repeated_trials_results(task, args.target_accuracy)
            if data:
                all_results[task] = data
                if not args.plot:  # Only print tables if not plotting
                    print_task_results(task, data)
            else:
                console.print(f"  ❌ No results found for {task}")
        
        # Print cross-task comparison (if not plotting)
        if len(all_results) > 1 and not args.plot:
            print_comparison_across_tasks(all_results)
        
        # Generate plot if requested
        if args.plot:
            if all_results:
                console.print("[bold cyan]📊 Generating stability box plots...[/bold cyan]")
                create_stability_box_plots(all_results, args.output)
            else:
                console.print("[bold red]❌ No results available for plotting[/bold red]")
        
        # Final summary
        if not args.plot:
            console.print(f"\n[bold green]📋 SUMMARY[/bold green]")
            console.print(f"Successfully loaded results for {len(all_results)} tasks: {list(all_results.keys())}")
            console.print(f"Target accuracy: {args.target_accuracy}")
            console.print("\n[bold cyan]💡 TIP: Use --plot to generate stability visualization[/bold cyan]")

if __name__ == "__main__":
    main() 