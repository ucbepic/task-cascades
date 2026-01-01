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

def load_varying_target_results(task: str) -> dict:
    """Load the latest varying target accuracy results for a task"""
    # Results are in the project root's results directory
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results" / "varying_target"
    latest_filename = f"{task}_varying_target_results_latest.pkl"
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
    all_target_results = data.get("all_target_results", {})
    
    # Get basic info
    target_accuracies = aggregated_stats.get("target_accuracies", [])
    oracle_cost = aggregated_stats.get("oracle_cost", 0)
    
    console.print(f"\n[bold cyan]🎯 VARYING TARGET ACCURACY RESULTS: {task.upper()}[/bold cyan]")
    console.print(f"Target Accuracies: {target_accuracies}")
    console.print("═" * 140)
    
    # Create main results table
    table = Table(title=f"{task.upper()} - Varying Target Accuracy Summary")
    table.add_column("Method", style="bold white", width=35)
    
    # Add columns for each target accuracy
    for target_acc in target_accuracies:
        table.add_column(f"Target {target_acc}\n(Acc/Cost)", justify="center", width=12)
    
    table.add_column("Avg\nAccuracy", justify="center", width=10)
    table.add_column("Avg Cost\nReduction", justify="center", width=12)
    table.add_column("Success\nRate", justify="center", width=10)
    
    # Method display names
    method_display_names = {
        "single_iteration": "🤖 Task Cascade",
        "single_iteration_agent_guaranteed": "🤖✅ Task Cascade (With Guarantees)",
        "baseline": "📏 2-Model Cascade",
        "baseline_guaranteed": "📏✅ 2-Model Cascade (with Guarantees)",
        "oracle": "🔮 Oracle Only"
    }
    
    # Sort methods for consistent display
    method_order = [
        "single_iteration",
        "single_iteration_agent_guaranteed", 
        "baseline",
        "baseline_guaranteed",
        "oracle"
    ]
    
    methods_data = aggregated_stats.get("methods", {})
    
    for method in method_order:
        if method in methods_data:
            stats = methods_data[method]
            display_name = method_display_names.get(method, method)
            
            row_data = [display_name]
            
            # Add accuracy/cost for each target
            achieved_accuracies = stats["achieved_accuracies"]
            costs = stats["costs"]
            meets_target_flags = stats["meets_target_flags"]
            
            for i, target_acc in enumerate(target_accuracies):
                if i < len(achieved_accuracies):
                    acc = achieved_accuracies[i]
                    cost = costs[i]
                    meets_target = "✓" if meets_target_flags[i] else "✗"
                    row_data.append(f"{acc:.3f}{meets_target}\n{cost:.3f}")
                else:
                    row_data.append("-")
            
            # Add average statistics
            avg_accuracy = np.mean(achieved_accuracies) if achieved_accuracies else 0
            avg_cost_reduction = np.mean(stats["cost_reductions"]) if stats["cost_reductions"] else 0
            success_rate = np.mean(meets_target_flags) if meets_target_flags else 0
            
            row_data.extend([
                f"{avg_accuracy:.3f}",
                f"{avg_cost_reduction:.1f}%",
                f"{success_rate:.1%}"
            ])
            
            table.add_row(*row_data)
    
    console.print(table)
    
    # Print additional insights
    console.print(f"\n[bold]📈 INSIGHTS FOR {task.upper()}:[/bold]")
    
    if len(methods_data) >= 2:
        # Performance trends across target accuracies
        console.print("• 📊 Performance trends:")
        
        for method in ["single_iteration", "baseline"]:
            if method in methods_data:
                costs = methods_data[method]["costs"]
                cost_reductions = methods_data[method]["cost_reductions"]
                
                display_name = method_display_names.get(method, method)
                
                if len(costs) > 1:
                    cost_trend = "increasing" if costs[-1] > costs[0] else "decreasing"
                    min_reduction = min(cost_reductions)
                    max_reduction = max(cost_reductions)
                    
                    console.print(f"  - {display_name}: {cost_trend} cost trend, {min_reduction:.1f}%-{max_reduction:.1f}% cost reduction range")
        
        # Compare methods at highest target accuracy
        if target_accuracies:
            highest_target = max(target_accuracies)
            console.print(f"\n• 🏆 At highest target accuracy ({highest_target}):")
            
            highest_idx = target_accuracies.index(highest_target)
            
            for method in ["single_iteration", "baseline"]:
                if method in methods_data:
                    stats = methods_data[method]
                    if highest_idx < len(stats["costs"]):
                        cost = stats["costs"][highest_idx]
                        cost_reduction = stats["cost_reductions"][highest_idx]
                        meets_target = stats["meets_target_flags"][highest_idx]
                        
                        display_name = method_display_names.get(method, method)
                        status = "✓ meets target" if meets_target else "✗ misses target"
                        
                        console.print(f"  - {display_name}: {cost_reduction:.1f}% cost reduction, {status}")

def print_comparison_across_tasks(all_results: dict):
    """Print a comparison table across all tasks"""
    console.print(f"\n[bold cyan]🌍 CROSS-TASK COMPARISON (Average Performance)[/bold cyan]")
    console.print("═" * 120)
    
    # Create cross-task comparison table
    table = Table(title="Average Method Performance Across All Tasks")
    table.add_column("Method", style="bold white", width=35)
    
    # Add columns for each task
    tasks = list(all_results.keys())
    for task in tasks:
        table.add_column(f"{task.replace('_', ' ').title()}\n(Avg Cost Red.)", justify="center", width=15)
    
    table.add_column("Overall\nAverage", justify="center", width=12)
    
    # Method display names
    method_display_names = {
        "single_iteration": "🤖 Task Cascade",
        "single_iteration_agent_guaranteed": "🤖✅ Task Cascade (With Guarantees)",
        "baseline": "📏 2-Model Cascade", 
        "baseline_guaranteed": "📏✅ 2-Model Cascade (with Guarantees)",
        "oracle": "🔮 Oracle Only"
    }
    
    method_order = [
        "single_iteration",
        "single_iteration_agent_guaranteed",
        "baseline", 
        "baseline_guaranteed",
        "oracle"
    ]
    
    for method in method_order:
        row_data = [method_display_names.get(method, method)]
        all_avg_reductions = []
        
        for task in tasks:
            if task in all_results and all_results[task]:
                aggregated_stats = all_results[task].get("aggregated_stats", {})
                methods_data = aggregated_stats.get("methods", {})
                
                if method in methods_data:
                    cost_reductions = methods_data[method]["cost_reductions"]
                    avg_reduction = np.mean(cost_reductions) if cost_reductions else 0
                    all_avg_reductions.append(avg_reduction)
                    row_data.append(f"{avg_reduction:.1f}%")
                else:
                    row_data.append("-")
            else:
                row_data.append("-")
        
        # Add overall average
        if all_avg_reductions:
            overall_avg = np.mean(all_avg_reductions)
            row_data.append(f"{overall_avg:.1f}%")
        else:
            row_data.append("-")
        
        table.add_row(*row_data)
    
    console.print(table)

def create_accuracy_vs_cost_plot(all_results: dict, output_path: str = None):
    """Create a subplot for each task showing accuracy vs cost"""
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
    
    # Method configurations: (method_key, color, marker, label)
    method_configs = [
        ("baseline", "blue", "o", "2-Model Cascade"),
        ("baseline_guaranteed", "blue", "^", "2-Model Cascade (with Guarantees)"),
        ("single_iteration", "green", "o", "Task Cascade"),
        ("single_iteration_agent_guaranteed", "green", "^", "Task Cascade (With Guarantees)"),
        ("oracle", "gold", "*", "Oracle")
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
        methods_data = aggregated_stats.get("methods", {})
        oracle_cost = aggregated_stats.get("oracle_cost", 1.0)
        
        # Plot each method
        for method_key, color, marker, label in method_configs:
            if method_key in methods_data:
                stats = methods_data[method_key]
                accuracies = stats["achieved_accuracies"]
                costs = stats["costs"]
                
                # Normalize costs by oracle cost for better comparison
                normalized_costs = [c / oracle_cost for c in costs]
                
                # Plot with specified styling - bigger points for better visibility
                marker_size = 120 if marker == "*" else 80
                ax.scatter(normalized_costs, accuracies, 
                          color=color, marker=marker, s=marker_size, 
                          alpha=0.8, label=label, edgecolors='black', linewidths=0.5)
        
        # Customize subplot with group labels
        ax.set_xlabel('Cost')
        ax.set_ylabel('Accuracy')
        
        # Add group labels to titles
        if task in ["enron", "legal_doc"]:
            group_label = "(A)"
        elif task in ["game_review", "court_opinion"]:
            group_label = "(B)"
        else:  # ag_news
            group_label = "(C)"
        
        title = f"{task.replace('_', ' ').title()} {group_label}"
        ax.set_title(title)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limits for log scale (can't start at 0)
        if methods_data:
            all_costs = []
            for method_stats in methods_data.values():
                normalized_method_costs = [c / oracle_cost for c in method_stats["costs"]]
                all_costs.extend(normalized_method_costs)
            
            if all_costs:
                min_cost = min(all_costs)
                max_cost = max(all_costs)
                # Add some padding for log scale
                ax.set_xlim(left=min_cost * 0.8, right=max_cost * 1.2)
        
        # Set y-axis to show meaningful range
        if methods_data:
            all_accuracies = []
            for method_stats in methods_data.values():
                all_accuracies.extend(method_stats["achieved_accuracies"])
            
            if all_accuracies:
                min_acc = min(all_accuracies)
                max_acc = max(all_accuracies)
                margin = (max_acc - min_acc) * 0.1 if max_acc > min_acc else 0.05
                ax.set_ylim(max(0, min_acc - margin), min(1, max_acc + margin))
    
    # No need to hide subplots since we have exactly n_tasks subplots
    
    # Create legend
    legend_elements = []
    for method_key, color, marker, label in method_configs:
        # Check if this method appears in any task
        method_exists = any(
            method_key in all_results[task].get("aggregated_stats", {}).get("methods", {})
            for task in tasks if all_results[task]
        )
        if method_exists:
            marker_size = 10 if marker == "*" else 8
            legend_elements.append(
                plt.Line2D([0], [0], marker=marker, color='w', 
                          markerfacecolor=color, markersize=marker_size, 
                          markeredgecolor='black', markeredgewidth=0.5, label=label)
            )
    


    # Add title with proper spacing
    fig.suptitle('Accuracy vs Cost Trade-offs', fontsize=16, y=0.92)
    
    # Place legend very close to the 5th subplot
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.87, 0.5), 
               ncol=1, frameon=False, fontsize=11)
    
    # Adjust layout with more space between plots
    plt.tight_layout(rect=[0, 0, 0.88, 0.94])
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        console.print(f"[bold green]📊 Plot saved to: {output_path}[/bold green]")
    else:
        # Default save location
        script_dir = Path(__file__).parent
        plots_dir = script_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        default_path = plots_dir / "varying_target_accuracy_vs_cost_all_tasks.pdf"
        plt.savefig(default_path, bbox_inches='tight')
        console.print(f"[bold green]📊 Plot saved to: {default_path}[/bold green]")
    
    # plt.show()

def main():
    parser = argparse.ArgumentParser(description='Print varying target accuracy results in table format')
    parser.add_argument(
        '--task',
        type=str,
        help='Specific task to display (if not provided, shows all available tasks)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate accuracy vs cost plot for all tasks'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for the plot (if not provided, saves to plots directory)'
    )
    
    args = parser.parse_args()
    
    # Available tasks (same as in the shell script)
    available_tasks = ["enron", "legal_doc", "game_review", "court_opinion", "ag_news"]
    
    if args.task:
        # Show results for specific task
        if args.task not in available_tasks:
            console.print(f"[bold red]❌ Task '{args.task}' not in available tasks: {available_tasks}[/bold red]")
            return
        
        data = load_varying_target_results(args.task)
        print_task_results(args.task, data)
        
        if args.plot:
            console.print("[bold yellow]⚠️  Plot option requires multiple tasks. Please run without --task to generate plots.[/bold yellow]")
    else:
        # Show results for all tasks
        all_results = {}
        
        console.print("[bold green]🔍 Loading varying target accuracy results for all tasks...[/bold green]")
        
        for task in available_tasks:
            console.print(f"  📂 Loading {task}...")
            data = load_varying_target_results(task)
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
                console.print("[bold cyan]📊 Generating accuracy vs cost plot...[/bold cyan]")
                create_accuracy_vs_cost_plot(all_results, args.output)
            else:
                console.print("[bold red]❌ No results available for plotting[/bold red]")
        
        # Final summary
        if not args.plot:
            console.print(f"\n[bold green]📋 SUMMARY[/bold green]")
            console.print(f"Successfully loaded results for {len(all_results)} tasks: {list(all_results.keys())}")
            console.print("\n[bold cyan]💡 TIP: Use --plot to generate accuracy vs cost visualization[/bold cyan]")

if __name__ == "__main__":
    main() 