import os
import glob
import pickle
import argparse
from rich.console import Console
from rich.table import Table
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
CACHE_DIR = "cache"
OUTPUT_DIR = "results/plots"
CONSOLE = Console()

# Standard tasks and target accuracies
STANDARD_TASKS = ["ag_news", "fever", "game_review", "pubmed", "wiki_talk", "court_opinion", "enron", "legal_doc"]
STANDARD_TARGET_ACCURACIES = [0.8, 0.85, 0.9, 0.95]
SEED = 42

def load_cascade_results(task: str, target_accuracy: float) -> Dict[str, Any]:
    """Load cascade results for a given task and target accuracy."""
    file_path = os.path.join(CACHE_DIR, f"{task}_seed_{SEED}_target_{target_accuracy}_cascade_results.pkl")
    
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        CONSOLE.print(f"[red]Error loading {file_path}: {e}[/red]")
        return None

def extract_surrogate_instructions(cascade_results: Dict[str, Any]) -> Dict[str, str]:
    """Extract surrogate instructions from cascade results."""
    if not cascade_results or "surrogate_to_prompt" not in cascade_results:
        return {}
    
    return cascade_results["surrogate_to_prompt"]

def format_instruction_text(instruction: str, max_width: int = 80) -> str:
    """Format instruction text with proper wrapping and cleaning."""
    # Remove any prefix/suffix patterns commonly used
    import re
    
    # Clean up common patterns
    instruction = instruction.strip()
    
    # Remove common wrapper text patterns
    patterns_to_remove = [
        r"^.*?Given the following document.*?:",
        r"^.*?Please classify.*?:",
        r"^.*?Classify.*?:",
        r"Output only.*?$",
        r"Your answer should be.*?$",
        r"^.*?Answer with.*?:",
    ]
    
    for pattern in patterns_to_remove:
        instruction = re.sub(pattern, "", instruction, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean up extra whitespace
    instruction = " ".join(instruction.split())
    
    # Simple word wrapping
    words = instruction.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= max_width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return "\n".join(lines)

def get_cascade_method_info(cascade_results: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Get information about a specific cascade method."""
    if method not in cascade_results:
        return {}
    
    method_data = cascade_results[method]
    
    return {
        "ordering": method_data.get("ordering", []),
        "accuracy": method_data.get("accuracy", 0.0),
        "total_cost": method_data.get("total_cost", 0.0),
        "num_surrogates_used": len([item for item in method_data.get("ordering", []) if item[0].startswith("s")])
    }

def generate_surrogate_instructions_report(output_file: str = None):
    """Generate a comprehensive report of all surrogate instructions across tasks and targets."""
    
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "cascade_surrogate_instructions.txt")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    CONSOLE.print(f"[bold cyan]🔍 Extracting Surrogate Instructions from Cascade Results[/bold cyan]")
    
    all_data = {}
    
    # Collect data for all tasks and targets
    for task in STANDARD_TASKS:
        CONSOLE.print(f"[cyan]Processing task: {task}[/cyan]")
        all_data[task] = {}
        
        for target_accuracy in STANDARD_TARGET_ACCURACIES:
            cascade_results = load_cascade_results(task, target_accuracy)
            
            if cascade_results:
                surrogate_instructions = extract_surrogate_instructions(cascade_results)
                
                # Get method info for the main methods
                method_info = {}
                for method in ["greedy", "selectivity", "greedy_guaranteed"]:
                    method_info[method] = get_cascade_method_info(cascade_results, method)
                
                all_data[task][target_accuracy] = {
                    "surrogate_instructions": surrogate_instructions,
                    "method_info": method_info
                }
                
                CONSOLE.print(f"  ✓ {target_accuracy*100:.0f}%: Found {len(surrogate_instructions)} surrogates")
            else:
                CONSOLE.print(f"  ✗ {target_accuracy*100:.0f}%: No data found")
    
    # Generate the report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("CASCADE SURROGATE INSTRUCTIONS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write("This report contains all surrogate instructions discovered across tasks and target accuracies.\n")
        f.write("Generated from cascade results files in cache/\n\n")
        
        for task in STANDARD_TASKS:
            if task not in all_data or not all_data[task]:
                continue
                
            f.write("=" * 80 + "\n")
            f.write(f"TASK: {task.upper().replace('_', ' ')}\n")
            f.write("=" * 80 + "\n\n")
            
            for target_accuracy in STANDARD_TARGET_ACCURACIES:
                if target_accuracy not in all_data[task]:
                    continue
                    
                data = all_data[task][target_accuracy]
                surrogate_instructions = data["surrogate_instructions"]
                method_info = data["method_info"]
                
                f.write(f"TARGET ACCURACY: {target_accuracy*100:.0f}%\n")
                f.write("-" * 40 + "\n\n")
                
                # Summary of cascade methods
                f.write("CASCADE METHOD SUMMARY:\n")
                for method, info in method_info.items():
                    if info:
                        f.write(f"  {method.title()}: {info['num_surrogates_used']} surrogates, "
                               f"accuracy={info['accuracy']:.3f}, cost=${info['total_cost']:.4f}\n")
                
                f.write("\n")
                
                # Show cascade orderings with full surrogate instructions
                for method, info in method_info.items():
                    if not info or not info.get("ordering"):
                        continue
                        
                    f.write(f"{method.upper()} METHOD CASCADE ORDERING:\n")
                    f.write("-" * 50 + "\n\n")
                    
                    for j, (surrogate_name, predictor, doc_fraction) in enumerate(info["ordering"], 1):
                        if surrogate_name in surrogate_instructions:
                            instruction = surrogate_instructions[surrogate_name]
                            
                            if surrogate_name == "s1":
                                f.write(f"{j}. MAIN TASK (s1) - {predictor}, {doc_fraction:.1f} doc fraction:\n")
                            else:
                                f.write(f"{j}. SURROGATE {surrogate_name.upper()} - {predictor}, {doc_fraction:.1f} doc fraction:\n")
                            
                            f.write("   " + "=" * 60 + "\n")
                            
                            # Format and indent the instruction
                            formatted_instruction = format_instruction_text(instruction, max_width=76)
                            for line in formatted_instruction.split('\n'):
                                f.write(f"   {line}\n")
                            
                            f.write("\n")
                    
                    f.write("\n")
                
                f.write("\n" + "=" * 60 + "\n\n")
    
    CONSOLE.print(f"[bold green]📁 Surrogate instructions report saved to:[/bold green] [underline]{output_file}[/underline]")
    
    # Print summary statistics
    CONSOLE.print(f"\n[bold cyan]📊 Summary Statistics[/bold cyan]")
    
    total_tasks = len([task for task in all_data if all_data[task]])
    total_combinations = sum(len(all_data[task]) for task in all_data if all_data[task])
    
    # Count unique surrogates across all tasks/targets
    all_surrogates = set()
    all_instructions = []
    
    for task in all_data:
        for target_accuracy in all_data[task]:
            surrogate_instructions = all_data[task][target_accuracy]["surrogate_instructions"]
            for surrogate_name, instruction in surrogate_instructions.items():
                if surrogate_name != "s1":  # Skip main task
                    all_surrogates.add(surrogate_name)
                    all_instructions.append(instruction)
    
    unique_instructions = len(set(all_instructions))
    
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Value", justify="center")
    
    summary_table.add_row("Tasks with data", str(total_tasks))
    summary_table.add_row("Task-target combinations", str(total_combinations))
    summary_table.add_row("Total surrogate instances", str(len(all_instructions)))
    summary_table.add_row("Unique surrogate instructions", str(unique_instructions))
    summary_table.add_row("Avg surrogates per combination", f"{len(all_instructions)/total_combinations:.1f}" if total_combinations > 0 else "0")
    
    CONSOLE.print(summary_table)

def generate_surrogate_count_table(target_accuracy: float = 0.9):
    """Generate a table showing the number of surrogates used by each method at a specific target accuracy."""
    
    CONSOLE.print(f"[bold cyan]📊 Generating Surrogate Count Table for Target Accuracy {target_accuracy*100:.0f}%[/bold cyan]")
    
    # Collect data for all tasks at the specified target accuracy
    table_data = {}
    methods = ["greedy", "selectivity", "greedy_guaranteed"]
    
    for task in STANDARD_TASKS:
        CONSOLE.print(f"[cyan]Processing task: {task}[/cyan]")
        cascade_results = load_cascade_results(task, target_accuracy)
        
        if cascade_results:
            table_data[task] = {}
            for method in methods:
                method_info = get_cascade_method_info(cascade_results, method)
                table_data[task][method] = method_info.get("num_surrogates_used", 0) if method_info else 0
            CONSOLE.print(f"  ✓ Found data for {task}")
        else:
            CONSOLE.print(f"  ✗ No data found for {task}")
    
    # Create and display the table
    table = Table(show_header=True, header_style="bold magenta", title=f"Number of Surrogates by Method (Target Accuracy: {target_accuracy*100:.0f}%)")
    
    # Add columns
    table.add_column("Task", style="cyan", no_wrap=True)
    for method in methods:
        table.add_column(method.title(), justify="center", style="yellow")
    table.add_column("Total", justify="center", style="bold green")
    
    # Add rows
    totals_by_method = {method: 0 for method in methods}
    
    for task in STANDARD_TASKS:
        if task in table_data:
            row_data = [task.replace('_', ' ').title()]
            task_total = 0
            
            for method in methods:
                count = table_data[task][method]
                row_data.append(str(count))
                totals_by_method[method] += count
                task_total += count
            
            row_data.append(str(task_total))
            table.add_row(*row_data)
    
    # Add totals row
    totals_row = ["[bold]TOTAL[/bold]"]
    grand_total = 0
    for method in methods:
        total = totals_by_method[method]
        totals_row.append(f"[bold]{total}[/bold]")
        grand_total += total
    totals_row.append(f"[bold]{grand_total}[/bold]")
    table.add_row(*totals_row)
    
    CONSOLE.print("\n")
    CONSOLE.print(table)
    
    # Print additional statistics
    CONSOLE.print(f"\n[bold cyan]📈 Summary Statistics[/bold cyan]")
    stats_table = Table(show_header=True, header_style="bold magenta")
    stats_table.add_column("Method", style="cyan")
    stats_table.add_column("Total Surrogates", justify="center")
    stats_table.add_column("Avg per Task", justify="center")
    stats_table.add_column("Max per Task", justify="center")
    stats_table.add_column("Min per Task", justify="center")
    
    for method in methods:
        method_counts = [table_data[task][method] for task in table_data if method in table_data[task]]
        if method_counts:
            total = sum(method_counts)
            avg = total / len(method_counts)
            max_count = max(method_counts)
            min_count = min(method_counts)
            
            stats_table.add_row(
                method.title(),
                str(total),
                f"{avg:.1f}",
                str(max_count),
                str(min_count)
            )
    
    CONSOLE.print(stats_table)
    
    return table_data

def create_surrogate_count_plot(target_accuracy: float = 0.9, save_path: str = None):
    """Create a compact clustered bar chart showing surrogate counts by method and task for SIGMOD paper."""
    
    CONSOLE.print(f"[bold cyan]📊 Creating surrogate count plot for {target_accuracy*100:.0f}% target accuracy[/bold cyan]")
    
    # Get the data
    table_data = {}
    methods = ["greedy", "greedy_guaranteed"]  # Remove selectivity
    method_labels = ["Task Cascades", "Task Cascades (+ Guarantees)"]  # Remove selectivity ordering
    
    for task in STANDARD_TASKS:
        cascade_results = load_cascade_results(task, target_accuracy)
        if cascade_results:
            table_data[task] = {}
            for method in methods:
                method_info = get_cascade_method_info(cascade_results, method)
                table_data[task][method] = method_info.get("num_surrogates_used", 0) if method_info else 0
    
    # Prepare data for clustered bar chart
    tasks = list(table_data.keys())
    # Use full task names
    task_labels = [task.replace('_', ' ').title() for task in tasks]
    
    # Reorder methods to match the new labels order
    methods_reordered = ["greedy", "greedy_guaranteed"]
    
    # Get values for each method across all tasks
    method_values = []
    for method in methods_reordered:
        values = [table_data[task][method] for task in tasks]
        method_values.append(values)
    
    # Set up the plot with compact styling for SIGMOD
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(7, 2.8))  # Slightly smaller since only 2 methods
    
    # Create clustered bar chart
    n_tasks = len(tasks)
    n_methods = len(methods)
    bar_width = 0.3  # Wider bars since only 2 methods
    x_pos = np.arange(n_tasks)
    
    # Colors for the two methods - dark green and orange
    colors = ['#2F7D32', '#F18F01']  # Dark green, Orange
    
    bars = []
    for i, (values, color, label) in enumerate(zip(method_values, colors, method_labels)):
        offset = (i - 0.5) * bar_width  # Center the bars around x_pos
        bar = ax.bar(x_pos + offset, values, bar_width, 
                    label=label, color=color, 
                    edgecolor='black', linewidth=0.4, alpha=0.8)
        bars.append(bar)
        
        # Add value labels on top of bars
        for j, (b, value) in enumerate(zip(bar, values)):
            if value > 0:  # Only show label if value > 0
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                       str(value), ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Customize the plot
    ax.set_xlabel('Workload', fontsize=10)
    ax.set_ylabel('# of Tasks', fontsize=10)
    ax.set_title(f'Number of Tasks in Cascades (Target Accuracy: {target_accuracy*100:.0f}%)', 
                fontsize=10, pad=12)
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(task_labels, fontsize=9, rotation=0, ha='center')
    
    # Set y-axis
    max_value = max(max(values) for values in method_values)
    ax.set_ylim(0, max_value * 1.15)
    ax.set_yticks(range(0, max_value + 2, 1))
    ax.tick_params(axis='y', labelsize=8)
    
    # Grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Legend with smaller font
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    # Tight layout for compact appearance
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, f"surrogate_count_clustered_target_{int(target_accuracy*100)}.pdf")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # plt.show()
    
    CONSOLE.print(f"[bold green]📁 Plot saved to:[/bold green] [underline]{save_path}[/underline]")
    
    return save_path

def print_cascade_for_task(task: str, target_accuracy: float):
    """Print cascade information for a specific task and target accuracy directly to terminal."""
    CONSOLE.print(f"[bold cyan]🔍 Loading cascade for {task} with target accuracy {target_accuracy*100:.0f}%[/bold cyan]")
    
    cascade_results = load_cascade_results(task, target_accuracy)
    
    if not cascade_results:
        CONSOLE.print(f"[red]❌ No cascade results found for {task} with target {target_accuracy*100:.0f}%[/red]")
        return
    
    surrogate_instructions = extract_surrogate_instructions(cascade_results)
    
    if not surrogate_instructions:
        CONSOLE.print(f"[red]❌ No surrogate instructions found in cascade results[/red]")
        return
    
    CONSOLE.print(f"[green]✅ Found {len(surrogate_instructions)} surrogates[/green]\n")
    
    # Get method info for the main methods
    methods = ["greedy", "selectivity", "greedy_guaranteed"]
    
    for method in methods:
        method_info = get_cascade_method_info(cascade_results, method)
        
        if not method_info or not method_info.get("ordering"):
            CONSOLE.print(f"[yellow]⚠️  No data found for {method} method[/yellow]")
            continue
            
        CONSOLE.print(f"[bold magenta]{'='*80}[/bold magenta]")
        CONSOLE.print(f"[bold magenta]{method.upper()} METHOD CASCADE[/bold magenta]")
        CONSOLE.print(f"[bold magenta]{'='*80}[/bold magenta]")
        CONSOLE.print(f"[cyan]Accuracy: {method_info['accuracy']:.3f} | Cost: ${method_info['total_cost']:.4f} | Surrogates: {method_info['num_surrogates_used']}[/cyan]\n")
        
        for j, (surrogate_name, predictor, doc_fraction) in enumerate(method_info["ordering"], 1):
            if surrogate_name in surrogate_instructions:
                instruction = surrogate_instructions[surrogate_name]
                
                if surrogate_name == "s1":
                    CONSOLE.print(f"[bold yellow]{j}. MAIN TASK (s1) - {predictor}[/bold yellow]")
                else:
                    CONSOLE.print(f"[bold yellow]{j}. SURROGATE {surrogate_name.upper()} - {predictor}[/bold yellow]")
                
                CONSOLE.print(f"[dim]   Document fraction: {doc_fraction:.3f}[/dim]")
                CONSOLE.print(f"[dim]   {'='*60}[/dim]")
                
                # Format and print the instruction
                formatted_instruction = format_instruction_text(instruction, max_width=76)
                for line in formatted_instruction.split('\n'):
                    CONSOLE.print(f"   {line}")
                
                CONSOLE.print()
        
        CONSOLE.print()

def main():
    """Main function to run the surrogate instruction extraction."""
    global STANDARD_TASKS, STANDARD_TARGET_ACCURACIES
    
    parser = argparse.ArgumentParser(description="Extract and list all surrogate instructions from cascade results")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output file path (default: results/plots/cascade_surrogate_instructions.txt)")
    parser.add_argument("--task", "-t", type=str, default=None,
                       help="Process specific task only (default: all tasks)")
    parser.add_argument("--target", type=float, default=None,
                       help="Process specific target accuracy only (default: all targets)")
    parser.add_argument("--print", "-p", action="store_true",
                       help="Print cascade to terminal instead of writing to file")
    parser.add_argument("--table", action="store_true",
                       help="Generate surrogate count table for target 90% accuracy")
    parser.add_argument("--plot", action="store_true",
                       help="Generate bar chart of surrogate counts by method")
    
    args = parser.parse_args()
    
    # If --plot flag is used, generate the bar chart
    if args.plot:
        target_acc = args.target if args.target else 0.9
        create_surrogate_count_plot(target_acc)
        return
    
    # If --table flag is used, generate the surrogate count table
    if args.table:
        target_acc = args.target if args.target else 0.9
        generate_surrogate_count_table(target_acc)
        return
    
    # If --print flag is used with specific task and target, print to terminal
    if args.print and args.task and args.target:
        if args.task not in STANDARD_TASKS:
            CONSOLE.print(f"[red]Error: Task '{args.task}' not found. Available: {', '.join(STANDARD_TASKS)}[/red]")
            return
        if args.target not in STANDARD_TARGET_ACCURACIES:
            CONSOLE.print(f"[red]Error: Target '{args.target}' not found. Available: {STANDARD_TARGET_ACCURACIES}[/red]")
            return
        
        print_cascade_for_task(args.task, args.target)
        return
    
    # Override global settings if specific task/target requested
    
    if args.task:
        if args.task in STANDARD_TASKS:
            STANDARD_TASKS = [args.task]
        else:
            CONSOLE.print(f"[red]Error: Task '{args.task}' not found. Available: {', '.join(STANDARD_TASKS)}[/red]")
            return
    
    if args.target:
        if args.target in STANDARD_TARGET_ACCURACIES:
            STANDARD_TARGET_ACCURACIES = [args.target]
        else:
            CONSOLE.print(f"[red]Error: Target '{args.target}' not found. Available: {STANDARD_TARGET_ACCURACIES}[/red]")
            return
    
    generate_surrogate_instructions_report(args.output)

if __name__ == "__main__":
    main() 