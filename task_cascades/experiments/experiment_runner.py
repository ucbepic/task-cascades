"""Reusable experiment runner for Task Cascades experiments."""

import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
from rich.console import Console

from task_cascades.config.config import ExperimentConfig, MethodConfig
from task_cascades.data.create_dfs import (
    prepare_data, load_dataset, apply_filtering_calibrator_to_dataframe
)
from task_cascades.filtering.train_classifier_for_filtering import (
    train_data_filtering, simple_similarity_data_filtering, position_based_data_filtering
)
from task_cascades.filtering.data_filtering_utils import chunk_and_get_confidences
from task_cascades.filtering.calibrators import train_filtering_calibrator
from task_cascades.cascade.find_surrogates import find_surrogates
from task_cascades.cascade.apply_trained_cascade import apply_cascade, train_and_apply_baseline_cascade
from task_cascades.predictors.predictors import (
    PROMPT_TO_TASK_TYPE_DICT, TASK_PROMPT_DICT, BASELINE_PREDICTOR, ORACLE_PREDICTOR, PREDICTORS
)
from task_cascades.config.consts import CANDIDATE_FRACTIONS

console = Console()


@dataclass
class ExperimentRunner:
    """Reusable experiment runner that handles data prep, method execution, and caching."""

    task: str
    sample_size: int = 1000
    train_split: float = 0.2
    seed: int = 42
    cache_dir: str = "cache"
    results_dir: str = "results"
    skip_cache: bool = False

    # Processed data (populated by prepare())
    train_df: pd.DataFrame = field(default=None, repr=False)
    test_df: pd.DataFrame = field(default=None, repr=False)
    train_df_filtered: pd.DataFrame = field(default=None, repr=False)
    test_df_filtered: pd.DataFrame = field(default=None, repr=False)
    train_df_no_filtering: pd.DataFrame = field(default=None, repr=False)
    test_df_no_filtering: pd.DataFrame = field(default=None, repr=False)
    oracle_cost: float = field(default=0.0)

    # Simple similarity filtered data (lazy loaded)
    train_df_simple_filtered: pd.DataFrame = field(default=None, repr=False)
    test_df_simple_filtered: pd.DataFrame = field(default=None, repr=False)

    # Position filtered data (lazy loaded)
    train_df_position_filtered: pd.DataFrame = field(default=None, repr=False)
    test_df_position_filtered: pd.DataFrame = field(default=None, repr=False)

    # Config
    config: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def _cache_path(self, suffix: str = "") -> str:
        return os.path.join(self.cache_dir, f"{self.task}_{self.sample_size}_seed_{self.seed}{suffix}_cache.pkl")

    def _load_cache(self, path: str) -> Optional[dict]:
        if not self.skip_cache and os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_cache(self, data: dict, path: str):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def prepare(self) -> "ExperimentRunner":
        """Load and preprocess data. Returns self for chaining."""

        # Load raw data
        cache_path = self._cache_path()
        cached = self._load_cache(cache_path)

        if cached:
            console.print("[bold cyan]📂 Loading[/bold cyan]: Data from cache")
            self.train_df = cached['train_df']
            self.test_df = cached['test_df']
        else:
            console.print("[bold magenta]🔮 Preparing[/bold magenta]: Loading dataset...")
            df, documents = load_dataset(self.task)
            self.train_df, self.test_df, documents, _ = prepare_data(
                self.task, df, documents, self.sample_size, self.train_split, random_seed=self.seed
            )
            self._save_cache({'train_df': self.train_df, 'test_df': self.test_df, 'documents': documents}, cache_path)

        # Train classifier
        classifier_path = os.path.join(self.cache_dir, f"{self.task}_seed_{self.seed}_classifier.pkl")
        cached_classifier = self._load_cache(classifier_path)

        if cached_classifier:
            console.print("[bold cyan]📂 Loading[/bold cyan]: Classifier from cache")
            classifier, chunk_size = cached_classifier['classifier'], cached_classifier['chunk_size']
        else:
            console.print("[bold magenta]🔮 Training[/bold magenta]: Data filtering classifier...")
            classifier, chunk_size = train_data_filtering(self.task, self.train_df)
            self._save_cache({'classifier': classifier, 'chunk_size': chunk_size}, classifier_path)

        # Chunk data
        chunked_cache_path = self._cache_path("_chunked")
        cached_chunked = self._load_cache(chunked_cache_path)

        if cached_chunked:
            console.print("[bold cyan]📂 Loading[/bold cyan]: Chunked data from cache")
            train_df_chunked = cached_chunked['train_df']
            test_df_chunked = cached_chunked['test_df']
        else:
            console.print("[bold blue]🔍 Processing[/bold blue]: Chunking documents...")
            train_df_chunked = chunk_and_get_confidences(self.train_df, chunk_size, classifier)
            test_df_chunked = chunk_and_get_confidences(self.test_df, chunk_size, classifier)
            self._save_cache({'train_df': train_df_chunked, 'test_df': test_df_chunked}, chunked_cache_path)

        # Train calibrator
        calibrator_path = os.path.join(self.cache_dir, f"{self.task}_seed_{self.seed}_filtering_calibrator.pkl")
        cached_calibrator = self._load_cache(calibrator_path)

        if cached_calibrator:
            console.print("[bold cyan]📂 Loading[/bold cyan]: Calibrator from cache")
            filtering_calibrator = cached_calibrator
        else:
            console.print("[bold magenta]🔮 Training[/bold magenta]: Filtering calibrator...")
            filtering_calibrator = train_filtering_calibrator(train_df_chunked, self.task)
            self._save_cache(filtering_calibrator, calibrator_path)

        # Apply calibrator
        filtered_cache_path = self._cache_path("_filtered")
        cached_filtered = self._load_cache(filtered_cache_path)

        if cached_filtered:
            console.print("[bold cyan]📂 Loading[/bold cyan]: Filtered data from cache")
            self.train_df_filtered = cached_filtered['train_df']
            self.test_df_filtered = cached_filtered['test_df']
        else:
            console.print("[bold blue]🔍 Processing[/bold blue]: Applying calibrator...")
            self.train_df_filtered = apply_filtering_calibrator_to_dataframe(train_df_chunked, filtering_calibrator)
            self.test_df_filtered = apply_filtering_calibrator_to_dataframe(test_df_chunked, filtering_calibrator)
            self._save_cache({'train_df': self.train_df_filtered, 'test_df': self.test_df_filtered}, filtered_cache_path)

        # No-filtering versions
        self.train_df_no_filtering = self.train_df.copy()
        self.train_df_no_filtering['filtered_text'] = self.train_df_no_filtering['text']
        self.train_df_no_filtering['fraction'] = 1.0

        self.test_df_no_filtering = self.test_df.copy()
        self.test_df_no_filtering['filtered_text'] = self.test_df_no_filtering['text']
        self.test_df_no_filtering['fraction'] = 1.0

        # Oracle cost
        self.oracle_cost = self.test_df.drop_duplicates(subset=["uuid"])["oracle_cost"].sum()

        return self

    def prepare_simple_similarity(self) -> "ExperimentRunner":
        """Prepare simple similarity filtered data (lazy)."""
        if self.train_df_simple_filtered is not None:
            return self

        console.print("[bold yellow]📐 Preparing[/bold yellow]: Simple similarity filtering...")
        classifier, chunk_size = simple_similarity_data_filtering(self.task, self.train_df)

        cache_path = self._cache_path("_simple_filtered")
        cached = self._load_cache(cache_path)

        if cached:
            self.train_df_simple_filtered = cached['train_df']
            self.test_df_simple_filtered = cached['test_df']
        else:
            train_chunked = chunk_and_get_confidences(self.train_df, chunk_size, classifier)
            test_chunked = chunk_and_get_confidences(self.test_df, chunk_size, classifier)
            calibrator = train_filtering_calibrator(train_chunked, self.task)
            self.train_df_simple_filtered = apply_filtering_calibrator_to_dataframe(train_chunked, calibrator)
            self.test_df_simple_filtered = apply_filtering_calibrator_to_dataframe(test_chunked, calibrator)
            self._save_cache({'train_df': self.train_df_simple_filtered, 'test_df': self.test_df_simple_filtered}, cache_path)

        return self

    def prepare_position_based(self) -> "ExperimentRunner":
        """Prepare position-based filtered data (lazy)."""
        if self.train_df_position_filtered is not None:
            return self

        console.print("[bold red]📍 Preparing[/bold red]: Position-based filtering...")
        classifier, chunk_size = position_based_data_filtering(self.task, self.train_df)

        cache_path = self._cache_path("_position_filtered")
        cached = self._load_cache(cache_path)

        if cached:
            self.train_df_position_filtered = cached['train_df']
            self.test_df_position_filtered = cached['test_df']
        else:
            train_chunked = chunk_and_get_confidences(self.train_df, chunk_size, classifier)
            test_chunked = chunk_and_get_confidences(self.test_df, chunk_size, classifier)
            calibrator = train_filtering_calibrator(train_chunked, self.task)
            self.train_df_position_filtered = apply_filtering_calibrator_to_dataframe(train_chunked, calibrator)
            self.test_df_position_filtered = apply_filtering_calibrator_to_dataframe(test_chunked, calibrator)
            self._save_cache({'train_df': self.train_df_position_filtered, 'test_df': self.test_df_position_filtered}, cache_path)

        return self

    def run_method(self, method: str, target_accuracy: float) -> Dict[str, Any]:
        """Run a single method and return results."""

        task_type = PROMPT_TO_TASK_TYPE_DICT[self.task]
        start_time = time.perf_counter()

        if method == "oracle":
            return {
                "overall_accuracy": 1.0,
                "total_cost": self.oracle_cost,
                "stage_usage": {"oracle": 1.0},
                "runtime": 0.0
            }

        elif method == "baseline":
            baseline, _ = train_and_apply_baseline_cascade(
                self.train_df_filtered, self.test_df_filtered, target_accuracy, self.task
            )
            baseline["runtime"] = time.perf_counter() - start_time
            return baseline

        elif method == "baseline_guaranteed":
            _, baseline_g = train_and_apply_baseline_cascade(
                self.train_df_filtered, self.test_df_filtered, target_accuracy, self.task
            )
            baseline_g["runtime"] = time.perf_counter() - start_time
            return baseline_g

        elif method == "task_cascades":
            cascade = find_surrogates(
                self.train_df_filtered, self.task, target_accuracy,
                num_iterations=self.config.NUM_ITERATIONS,
                num_surrogate_requests=self.config.SURROGATES_PER_ITERATION
            )
            result = apply_cascade(
                self.test_df_filtered, cascade["greedy"]["ordering"],
                cascade["surrogate_to_prompt"], cascade["greedy"]["thresholds"], task_type
            )
            result["runtime"] = time.perf_counter() - start_time
            return result

        elif method == "task_cascades_guaranteed":
            cascade = find_surrogates(
                self.train_df_filtered, self.task, target_accuracy,
                num_iterations=self.config.NUM_ITERATIONS,
                num_surrogate_requests=self.config.SURROGATES_PER_ITERATION,
                guarantee_accuracy=True
            )
            result = apply_cascade(
                self.test_df_filtered, cascade["greedy_guaranteed"]["ordering"],
                cascade["surrogate_to_prompt"], cascade["greedy_guaranteed"]["thresholds"], task_type
            )
            result["runtime"] = time.perf_counter() - start_time
            return result

        elif method == "task_cascades_lite":
            cascade = find_surrogates(
                self.train_df_filtered, self.task, target_accuracy,
                num_iterations=1, num_surrogate_requests=8,
                provide_feedback=True, include_selectivity=False,
                proxy_predictor_only=True  # TC Lite: surrogates use only gpt-4o-mini
            )
            result = apply_cascade(
                self.test_df_filtered, cascade["greedy"]["ordering"],
                cascade["surrogate_to_prompt"], cascade["greedy"]["thresholds"], task_type
            )
            result["runtime"] = time.perf_counter() - start_time
            return result

        elif method == "selectivity_ordering":
            cascade = find_surrogates(
                self.train_df_filtered, self.task, target_accuracy,
                num_iterations=self.config.NUM_ITERATIONS,
                num_surrogate_requests=self.config.SURROGATES_PER_ITERATION,
                include_selectivity=True
            )
            result = apply_cascade(
                self.test_df_filtered, cascade["selectivity"]["ordering"],
                cascade["surrogate_to_prompt"], cascade["selectivity"]["thresholds"], task_type
            )
            result["runtime"] = time.perf_counter() - start_time
            return result

        elif method == "no_filtering":
            cascade = find_surrogates(
                self.train_df_no_filtering, self.task, target_accuracy,
                num_iterations=self.config.NUM_ITERATIONS,
                num_surrogate_requests=self.config.SURROGATES_PER_ITERATION,
                include_selectivity=False
            )
            result = apply_cascade(
                self.test_df_no_filtering, cascade["greedy"]["ordering"],
                cascade["surrogate_to_prompt"], cascade["greedy"]["thresholds"], task_type
            )
            result["runtime"] = time.perf_counter() - start_time
            return result

        elif method == "no_surrogates":
            cascade = self._create_no_surrogate_cascade(self.train_df_filtered, target_accuracy)
            result = apply_cascade(
                self.test_df_filtered, cascade["greedy"]["ordering"],
                cascade["surrogate_to_prompt"], cascade["greedy"]["thresholds"], task_type
            )
            result["runtime"] = time.perf_counter() - start_time
            return result

        elif method == "single_iteration":
            cascade = find_surrogates(
                self.train_df_filtered, self.task, target_accuracy,
                num_iterations=1, provide_feedback=False, include_selectivity=False
            )
            result = apply_cascade(
                self.test_df_filtered, cascade["greedy"]["ordering"],
                cascade["surrogate_to_prompt"], cascade["greedy"]["thresholds"], task_type
            )
            result["runtime"] = time.perf_counter() - start_time
            return result

        elif method == "naive_rag_filter":
            self.prepare_simple_similarity()
            cascade = find_surrogates(
                self.train_df_simple_filtered, self.task, target_accuracy,
                num_iterations=self.config.NUM_ITERATIONS,
                num_surrogate_requests=self.config.SURROGATES_PER_ITERATION,
                include_selectivity=False
            )
            result = apply_cascade(
                self.test_df_simple_filtered, cascade["greedy"]["ordering"],
                cascade["surrogate_to_prompt"], cascade["greedy"]["thresholds"], task_type
            )
            result["runtime"] = time.perf_counter() - start_time
            return result

        elif method == "restructure_top25":
            cascade = self._create_no_surrogate_cascade(self.train_df_filtered, target_accuracy)
            result = apply_cascade(
                self.test_df_filtered, cascade["greedy"]["ordering"],
                cascade["surrogate_to_prompt"], cascade["greedy"]["thresholds"], task_type
            )
            result["runtime"] = time.perf_counter() - start_time
            return result

        elif method == "rag_no_surrogates":
            self.prepare_simple_similarity()
            cascade = self._create_no_surrogate_cascade(self.train_df_simple_filtered, target_accuracy)
            result = apply_cascade(
                self.test_df_simple_filtered, cascade["greedy"]["ordering"],
                cascade["surrogate_to_prompt"], cascade["greedy"]["thresholds"], task_type
            )
            result["runtime"] = time.perf_counter() - start_time
            return result

        elif method == "lotus":
            return self._run_lotus(target_accuracy)

        else:
            raise ValueError(f"Unknown method: {method}")

    def _create_no_surrogate_cascade(self, train_df: pd.DataFrame, target_accuracy: float) -> dict:
        """Create a cascade with only the baseline task (no surrogates)."""
        from task_cascades.cascade.cascade_utils import design_cascade_optimal_greedy
        from task_cascades.predictors.predictors import run_predictor_and_get_row_copies

        all_executions = []
        task_prompt = TASK_PROMPT_DICT[self.task]
        task_type = PROMPT_TO_TASK_TYPE_DICT[self.task]

        baseline_results = run_predictor_and_get_row_copies(
            BASELINE_PREDICTOR, task_prompt, train_df, "s1", task_type=task_type
        )
        oracle_results = run_predictor_and_get_row_copies(
            ORACLE_PREDICTOR, task_prompt, train_df, "s1", task_type=task_type
        )

        all_executions.extend(baseline_results)
        all_executions.extend(oracle_results)
        all_executions_df = pd.DataFrame(all_executions)

        all_candidates = []
        for doc_fraction in CANDIDATE_FRACTIONS:
            for predictor in PREDICTORS:
                if predictor == ORACLE_PREDICTOR and doc_fraction == 1.0:
                    continue
                all_candidates.append(("s1", predictor, doc_fraction))

        cascade_greedy = design_cascade_optimal_greedy(all_executions_df, all_candidates, target_accuracy, self.task)
        surrogate_to_prompt = {"s1": task_prompt}

        return {
            "greedy": {**cascade_greedy, "surrogate_to_prompt": surrogate_to_prompt},
            "surrogate_to_prompt": surrogate_to_prompt
        }

    def _run_lotus(self, target_accuracy: float) -> Dict[str, Any]:
        """Run LOTUS baseline."""
        try:
            from task_cascades.baselines.lotus import (
                configure_lotus_models, label_documents_with_oracle,
                run_lotus_binary, run_lotus_ag_news_ovr,
                monkey_patch_lotus_logprobs_assert, BINARY_TASKS, MULTICLASS_TASKS
            )

            monkey_patch_lotus_logprobs_assert()
            configure_lotus_models()

            test_df_lotus = self.test_df.drop_duplicates(subset=["uuid"]).copy()
            labeled_df, _ = label_documents_with_oracle(test_df_lotus, self.task)

            if self.task in BINARY_TASKS:
                results = run_lotus_binary(
                    labeled_df, self.task,
                    recall_targets=[target_accuracy],
                    precision_targets=[target_accuracy],
                    oracle_total_cost=self.oracle_cost
                )
            elif self.task in MULTICLASS_TASKS:
                results = run_lotus_ag_news_ovr(
                    labeled_df,
                    recall_targets=[target_accuracy],
                    precision_targets=[target_accuracy],
                    oracle_total_cost=self.oracle_cost
                )
            else:
                return {"error": f"LOTUS not supported for task {self.task}"}

            if (target_accuracy, target_accuracy) in results:
                point = results[(target_accuracy, target_accuracy)]
                return {
                    "overall_accuracy": point["accuracy"],
                    "total_cost": point["cost"],
                    "stage_usage": {"lotus": 1.0}
                }
            return {"error": "No results for target accuracy"}

        except ImportError as e:
            return {"error": f"LOTUS not available: {e}"}
        except Exception as e:
            return {"error": f"LOTUS failed: {e}"}

    def run_all(self, methods: List[str], target_accuracy: float) -> Dict[str, Any]:
        """Run all specified methods and return results dict."""
        results = {}
        for method in methods:
            console.print(f"[bold]Running {method}...[/bold]")
            try:
                results[method] = self.run_method(method, target_accuracy)
                if "overall_accuracy" in results[method]:
                    console.print(f"  ✓ {method}: accuracy={results[method]['overall_accuracy']:.4f}, cost={results[method]['total_cost']:.4f}")
            except Exception as e:
                console.print(f"  ✗ {method} failed: {e}")
                results[method] = {"error": str(e)}
        return results

    def save_results(self, results: Dict[str, Any], target_accuracy: float) -> str:
        """Save results to file and return path."""
        import pandas as pd

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(
            self.results_dir,
            f"{self.task}_seed_{self.seed}_target_{target_accuracy}_{timestamp}.pkl"
        )

        full_results = {
            "task": self.task,
            "sample_size": self.sample_size,
            "seed": self.seed,
            "target_accuracy": target_accuracy,
            "oracle_cost": self.oracle_cost,
            "methods": results
        }

        with open(results_path, 'wb') as f:
            pickle.dump(full_results, f)

        # Also save latest
        latest_path = os.path.join(
            self.results_dir,
            f"{self.task}_seed_{self.seed}_target_{target_accuracy}_latest.pkl"
        )
        with open(latest_path, 'wb') as f:
            pickle.dump(full_results, f)

        return results_path
