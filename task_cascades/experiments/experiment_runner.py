"""Main experiment runner class with improved structure."""

import time
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from rich.console import Console
from rich.table import Table

from .config import ExperimentConfig, MethodConfig
from .cache_manager import CacheManager
from .create_dfs import prepare_data, load_dataset, apply_filtering_calibrator_to_dataframe
from .train_classifier_for_filtering import train_data_filtering, simple_similarity_data_filtering, position_based_data_filtering
from .data_filtering_utils import chunk_and_get_confidences
from .calibrators import train_filtering_calibrator
from .find_surrogates import find_surrogates
from .apply_trained_cascade import apply_cascade, train_and_apply_baseline_cascade, train_and_apply_lotus_cascade
from .predictors import PROMPT_TO_TASK_TYPE_DICT, TASK_PROMPT_DICT

class ExperimentRunner:
    """Main class for running cascade experiments."""
    
    def __init__(self, config: ExperimentConfig = None, method_config: MethodConfig = None):
        self.config = config or ExperimentConfig()
        self.method_config = method_config or MethodConfig()
        self.cache_manager = CacheManager(self.config.CACHE_DIR)
        self.console = Console()
        self.config.ensure_directories()
    
    def setup_data(self, task: str, sample_size: int, train_split: float, seed: int, skip_cache: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, List, List]:
        """Set up and load experiment data."""
        cache_path = self.cache_manager.get_cache_path(task, sample_size, seed)
        
        if not skip_cache and self.cache_manager.cache_exists(cache_path):
            self.console.print("[bold cyan]📂 Loading[/bold cyan]: Data from cache")
            cached_data = self.cache_manager.load_from_cache(cache_path)
            return cached_data['train_df'], cached_data['test_df'], cached_data['documents'], cached_data['train_indices']
        else:
            self.console.print("[bold magenta]🔮 Preparing[/bold magenta]: Loading dataset and preparing data...")
            df, documents = load_dataset(task)
            train_df, test_df, documents, train_indices = prepare_data(
                task, df, documents, sample_size, train_split, random_seed=seed
            )
            
            self.cache_manager.save_to_cache({
                'train_df': train_df,
                'test_df': test_df,
                'documents': documents,
                'train_indices': train_indices
            }, cache_path)
            
            return train_df, test_df, documents, train_indices
    
    def setup_classifier(self, task: str, train_df: pd.DataFrame, seed: int, skip_cache: bool = False) -> Tuple[Any, int]:
        """Set up data filtering classifier."""
        classifier_path = self.cache_manager.get_classifier_path(task, seed)
        
        if not skip_cache and self.cache_manager.cache_exists(classifier_path):
            self.console.print("[bold cyan]📂 Loading[/bold cyan]: Data filtering classifier from cache")
            return self.cache_manager.load_classifier(task, seed)
        else:
            self.console.print("[bold magenta]🔮 Training[/bold magenta]: Training data filtering classifier...")
            classifier, chunk_size = train_data_filtering(task, train_df)
            self.cache_manager.save_classifier(task, seed, classifier, chunk_size)
            return classifier, chunk_size
    
    def setup_chunked_data(self, task: str, sample_size: int, seed: int, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                          classifier: Any, chunk_size: int, skip_cache: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Set up chunked data with confidences."""
        chunked_cache_path = self.cache_manager.get_cache_path(task, sample_size, seed, suffix="_chunked")
        
        if not skip_cache and self.cache_manager.cache_exists(chunked_cache_path):
            self.console.print("[bold cyan]📂 Loading[/bold cyan]: Chunked documents with confidences from cache")
            cached_chunked_data = self.cache_manager.load_from_cache(chunked_cache_path)
            return cached_chunked_data['train_df'], cached_chunked_data['test_df']
        else:
            self.console.print("[bold blue]🔍 Processing[/bold blue]: Chunking documents and calculating confidences...")
            train_df_chunked = chunk_and_get_confidences(train_df, chunk_size, classifier)
            test_df_chunked = chunk_and_get_confidences(test_df, chunk_size, classifier)
            self.cache_manager.save_to_cache({
                'train_df': train_df_chunked,
                'test_df': test_df_chunked
            }, chunked_cache_path)
            return train_df_chunked, test_df_chunked
    
    def setup_filtered_data(self, task: str, sample_size: int, seed: int, train_df_chunked: pd.DataFrame, 
                           test_df_chunked: pd.DataFrame, target_accuracy: float, skip_cache: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
        """Set up filtered data with calibrator."""
        # Use first target accuracy for backward compatibility
        filtering_calibrator_path = self.cache_manager.get_filtering_calibrator_path(task, seed, target_accuracy)
        
        if not skip_cache and self.cache_manager.cache_exists(filtering_calibrator_path):
            self.console.print("[bold cyan]📂 Loading[/bold cyan]: Data filtering calibrator from cache")
            filtering_calibrator = self.cache_manager.load_filtering_calibrator(task, seed, target_accuracy)
        else:
            self.console.print("[bold magenta]🔮 Training[/bold magenta]: Data filtering calibrator...")
            filtering_calibrator = train_filtering_calibrator(train_df_chunked, task)
            self.cache_manager.save_filtering_calibrator(task, seed, target_accuracy, filtering_calibrator)
        
        filtered_cache_path = self.cache_manager.get_cache_path(task, sample_size, seed, suffix="_filtered")
        if not skip_cache and self.cache_manager.cache_exists(filtered_cache_path):
            self.console.print("[bold cyan]📂 Loading[/bold cyan]: Filtered dataframes from cache")
            cached = self.cache_manager.load_from_cache(filtered_cache_path)
            train_df_filtered = cached['train_df']
            test_df_filtered = cached['test_df']
        else:
            self.console.print("[bold blue]🔍 Processing[/bold blue]: Applying filtering calibrator...")
            train_df_filtered = apply_filtering_calibrator_to_dataframe(train_df_chunked, filtering_calibrator)
            test_df_filtered = apply_filtering_calibrator_to_dataframe(test_df_chunked, filtering_calibrator)
            self.cache_manager.save_to_cache({'train_df': train_df_filtered, 'test_df': test_df_filtered}, filtered_cache_path)
        
        return train_df_filtered, test_df_filtered, filtering_calibrator
    
    def create_no_data_filtering_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a version of the dataframe with no data filtering applied."""
        df_no_filtering = df.copy()
        df_no_filtering['filtered_text'] = df_no_filtering['text']
        df_no_filtering['fraction'] = 1.0
        return df_no_filtering
    
    def run_baseline_methods(self, train_df_filtered: pd.DataFrame, test_df_filtered: pd.DataFrame, 
                           target_accuracy: float, task: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Run baseline and lotus cascade methods."""
        self.console.print("\n[bold yellow]📏 BASELINE METHODS: Traditional Approaches[/bold yellow]")
        start_time = time.perf_counter()
        
        # Compute baseline cascade
        self.console.print("[bold magenta]🔮 Computing[/bold magenta]: Baseline cascade...")
        baseline_results, baseline_results_guaranteed = train_and_apply_baseline_cascade(
            train_df_filtered, test_df_filtered, target_accuracy, task
        )
        baseline_results["runtime"] = time.perf_counter() - start_time
        baseline_results_guaranteed["runtime"] = baseline_results["runtime"]
        
        # Compute lotus cascade
        start_time = time.perf_counter()
        self.console.print("[bold magenta]🔮 Computing[/bold magenta]: Lotus cascade...")
        lotus_results = train_and_apply_lotus_cascade(
            train_df_filtered, test_df_filtered, target_accuracy, task
        )
        lotus_results["runtime"] = time.perf_counter() - start_time
        
        # Print results
        self.console.print(
            f"[bold cyan]Baseline cost:[/bold cyan] [green]{baseline_results['total_cost']:.4f}[/green]    "
            f"[bold cyan]Baseline accuracy:[/bold cyan] [yellow]{baseline_results['overall_accuracy']:.4f}[/yellow]"
        )
        self.console.print(
            f"[bold cyan]Baseline guaranteed cost:[/bold cyan] [green]{baseline_results_guaranteed['total_cost']:.4f}[/green]    "
            f"[bold cyan]Baseline guaranteed accuracy:[/bold cyan] [yellow]{baseline_results_guaranteed['overall_accuracy']:.4f}[/yellow]"
        )
        self.console.print(
            f"[bold cyan]Lotus cost:[/bold cyan] [green]{lotus_results['total_cost']:.4f}[/green]    "
            f"[bold cyan]Lotus accuracy:[/bold cyan] [yellow]{lotus_results['overall_accuracy']:.4f}[/yellow]"
        )
        
        return baseline_results, baseline_results_guaranteed, lotus_results
    
    def run_main_cascade_methods(self, train_df_filtered: pd.DataFrame, test_df_filtered: pd.DataFrame,
                               task: str, target_accuracy: float, seed: int, skip_cache: bool = False) -> Dict[str, Any]:
        """Run main cascade methods with the updated configuration."""
        self.console.print(f"\n[bold green]🔄 MAIN METHODS: Full Pipeline with Surrogates + Data Filtering (Target: {target_accuracy})[/bold green]")
        
        # Load or find surrogates for main methods
        cascade_results_path = self.cache_manager.get_cascade_results_path(task, seed, target_accuracy)
        if not skip_cache and self.cache_manager.cache_exists(cascade_results_path):
            self.console.print("[bold cyan]📂 Loading[/bold cyan]: Cascade results from cache")
            cascade_results = self.cache_manager.load_cascade_results(task, seed, target_accuracy)
        else:
            self.console.print("[bold magenta]🔮 Discovering[/bold magenta]: Finding surrogates for cascade...")
            # Updated to use 3 iterations with 5 surrogates per iteration
            cascade_results = find_surrogates(
                train_df_filtered, 
                task, 
                target_accuracy, 
                guarantee_accuracy=True, 
                num_iterations=self.config.NUM_ITERATIONS,  # 3 iterations
                num_surrogate_requests=self.config.SURROGATES_PER_ITERATION  # 5 surrogates per iteration
            )
            self.cache_manager.save_cascade_results(task, seed, target_accuracy, cascade_results)
        
        return cascade_results
    
    def run_no_data_filtering_method(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                   task: str, target_accuracy: float) -> Dict[str, Any]:
        """Run ablation without data filtering."""
        self.console.print("\n[bold orange1]📄 ABLATION 1: No Data Filtering (Surrogates on Full Documents)[/bold orange1]")
        start_time = time.perf_counter()
        
        # Create no-filtering dataframes
        train_df_no_filtering = self.create_no_data_filtering_df(train_df)
        test_df_no_filtering = self.create_no_data_filtering_df(test_df)
        
        self.console.print("[bold magenta]🔮 Discovering[/bold magenta]: Finding surrogates without data filtering...")
        no_filtering_cascade_results = find_surrogates(
            train_df_no_filtering, 
            task, 
            target_accuracy, 
            include_selectivity=False, 
            num_iterations=self.config.NUM_ITERATIONS,
            num_surrogate_requests=self.config.SURROGATES_PER_ITERATION
        )
        
        self.console.print("[bold magenta]🔮 Testing[/bold magenta]: Applying no-filtering cascade to test set...")
        no_filtering_results = apply_cascade(
            test_df_no_filtering, 
            no_filtering_cascade_results["greedy"]["ordering"], 
            no_filtering_cascade_results["surrogate_to_prompt"], 
            no_filtering_cascade_results["greedy"]["thresholds"], 
            PROMPT_TO_TASK_TYPE_DICT[task]
        )
        no_filtering_results["runtime"] = time.perf_counter() - start_time
        
        return no_filtering_results