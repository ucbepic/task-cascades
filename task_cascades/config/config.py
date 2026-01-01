"""Configuration settings for Task Cascades experiments."""

from dataclasses import dataclass
from typing import List, Dict
import os


@dataclass
class ExperimentConfig:
    """Configuration for experiment parameters."""

    # Default experiment settings
    SAMPLE_SIZE: int = 1000
    TARGET_ACCURACIES: List[float] = None
    CACHE_DIR: str = "cache"
    RESULTS_DIR: str = "results"

    # Cascade settings (3 iterations x 5 surrogates = 15 total)
    NUM_ITERATIONS: int = 3
    SURROGATES_PER_ITERATION: int = 5

    # Default methods to run
    DEFAULT_METHODS: List[str] = None

    def __post_init__(self):
        if self.TARGET_ACCURACIES is None:
            self.TARGET_ACCURACIES = [0.9]

        if self.DEFAULT_METHODS is None:
            self.DEFAULT_METHODS = [
                # Main methods
                "task_cascades",
                "task_cascades_guaranteed",
                "task_cascades_lite",
                # Variants
                "no_filtering",
                "no_surrogates",
                "filtering_only",
                # Baselines
                "baseline",
                "baseline_guaranteed",
                "oracle",
            ]

    def ensure_directories(self):
        """Ensure cache and results directories exist."""
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)


# Method name mappings (new paper names -> old internal names)
# For backwards compatibility with existing code
METHOD_NAME_MAP = {
    "task_cascades": "main_greedy",
    "task_cascades_guaranteed": "main_greedy_guaranteed",
    "task_cascades_lite": "tc_lite",
    "no_filtering": "no_data_filtering_greedy",
    "no_surrogates": "no_surrogate_greedy",
    "filtering_only": "learned_restructuring_no_surrogate_greedy",
    "single_iteration": "single_iteration_agent_greedy",
    "similarity_filtering": "simple_similarity_filtering_greedy",
    "baseline": "baseline",
    "baseline_guaranteed": "baseline_with_guarantees",
    "oracle": "oracle_only",
    "lotus": "lotus",
}


@dataclass
class MethodConfig:
    """Display configuration for experiment methods."""

    METHOD_STYLES: Dict[str, str] = None
    METHOD_DESCRIPTIONS: Dict[str, str] = None

    def __post_init__(self):
        if self.METHOD_STYLES is None:
            self.METHOD_STYLES = {
                # Main methods (Task Cascades)
                "task_cascades": "[bold green]Task Cascades[/bold green]",
                "task_cascades_guaranteed": "[bold green]Task Cascades (Guaranteed)[/bold green]",
                "task_cascades_lite": "[bold cyan]Task Cascades Lite[/bold cyan]",
                # Variants
                "no_filtering": "[bold yellow]No Filtering[/bold yellow]",
                "no_surrogates": "[bold yellow]No Surrogates[/bold yellow]",
                "filtering_only": "[bold yellow]Filtering Only[/bold yellow]",
                "single_iteration": "[bold yellow]Single Iteration[/bold yellow]",
                "similarity_filtering": "[bold yellow]Similarity Filtering[/bold yellow]",
                # Baselines
                "baseline": "[bold magenta]2-Model Baseline[/bold magenta]",
                "baseline_guaranteed": "[bold magenta]2-Model Baseline (Guaranteed)[/bold magenta]",
                "oracle": "[bold red]Oracle[/bold red]",
                "lotus": "[bold blue]LOTUS[/bold blue]",
            }

        if self.METHOD_DESCRIPTIONS is None:
            self.METHOD_DESCRIPTIONS = {
                # Main methods
                "task_cascades": "Full Task Cascades: surrogate discovery + learned filtering",
                "task_cascades_guaranteed": "Task Cascades with statistical accuracy guarantees",
                "task_cascades_lite": "Lightweight variant: 1 iteration, 8 surrogates",
                # Variants
                "no_filtering": "Surrogates only (no document filtering)",
                "no_surrogates": "Filtering only (no surrogate tasks)",
                "filtering_only": "Learned filtering without surrogate discovery",
                "single_iteration": "Single iteration with 15 surrogates",
                "similarity_filtering": "Cosine similarity filtering instead of learned",
                # Baselines
                "baseline": "Simple 2-model cascade (GPT-4o-mini -> GPT-4o)",
                "baseline_guaranteed": "2-model cascade with accuracy guarantees",
                "oracle": "All documents to GPT-4o (cost upper bound)",
                "lotus": "LOTUS semantic filtering baseline",
            }
