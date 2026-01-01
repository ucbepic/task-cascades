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
                "no_surrogates",
                "single_iteration",
                "no_filtering",
                "naive_rag_filter",
                "selectivity_ordering",
                "restructure_top25",
                "rag_no_surrogates",
                # Baselines
                "baseline",
                "baseline_guaranteed",
                "oracle",
                # External (optional)
                "lotus",
            ]

    def ensure_directories(self):
        """Ensure cache and results directories exist."""
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)


@dataclass
class MethodConfig:
    """Display configuration for experiment methods."""

    METHOD_STYLES: Dict[str, str] = None
    METHOD_DESCRIPTIONS: Dict[str, str] = None

    def __post_init__(self):
        if self.METHOD_STYLES is None:
            self.METHOD_STYLES = {
                # Baselines
                "oracle": "[bold red]Oracle Only[/bold red]",
                "baseline": "[bold magenta]2-Model Cascade[/bold magenta]",
                "baseline_guaranteed": "[bold magenta]2-Model Cascade (+G)[/bold magenta]",
                # Main methods
                "task_cascades": "[bold green]Task Cascades[/bold green]",
                "task_cascades_guaranteed": "[bold green]Task Cascades (+G)[/bold green]",
                "task_cascades_lite": "[bold cyan]Task Cascades (Lite)[/bold cyan]",
                # Variants
                "no_surrogates": "[bold yellow]No Surrogates[/bold yellow]",
                "single_iteration": "[bold yellow]Single-Iteration[/bold yellow]",
                "no_filtering": "[bold yellow]No Filtering[/bold yellow]",
                "naive_rag_filter": "[bold yellow]Naive RAG Filter[/bold yellow]",
                "selectivity_ordering": "[bold yellow]Selectivity Ordering[/bold yellow]",
                "restructure_top25": "[bold yellow]Restructure (Top-25%)[/bold yellow]",
                "rag_no_surrogates": "[bold yellow]RAG + NoSur[/bold yellow]",
                # External baselines
                "lotus": "[bold blue]LOTUS[/bold blue]",
            }

        if self.METHOD_DESCRIPTIONS is None:
            self.METHOD_DESCRIPTIONS = {
                # Baselines
                "oracle": "All documents sent to GPT-4o",
                "baseline": "2-model cascade: GPT-4o-mini → GPT-4o",
                "baseline_guaranteed": "2-model cascade with statistical accuracy guarantees",
                # Main methods
                "task_cascades": "Full pipeline: surrogate discovery + learned filtering (3 iter × 5 surrogates)",
                "task_cascades_guaranteed": "Task Cascades with statistical accuracy guarantees",
                "task_cascades_lite": "Lightweight: 1 iteration, 8 surrogates",
                # Variants
                "no_surrogates": "Learned filtering only, no surrogate task discovery",
                "single_iteration": "Single iteration generating all 15 surrogates at once",
                "no_filtering": "Surrogate discovery only, no document filtering",
                "naive_rag_filter": "Cosine similarity filtering instead of learned filtering",
                "selectivity_ordering": "Selectivity-based cascade ordering instead of greedy",
                "restructure_top25": "Keep top-25% most relevant chunks per document",
                "rag_no_surrogates": "Cosine similarity filtering without surrogate discovery",
                # External baselines
                "lotus": "LOTUS semantic filtering baseline",
            }
