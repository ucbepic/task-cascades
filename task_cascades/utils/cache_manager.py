"""Cache management utilities for experiments."""

import os
import pickle
from typing import Any, Dict, Optional
from pathlib import Path

class CacheManager:
    """Manages caching for experiment data."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.ensure_cache_dir()
    
    def ensure_cache_dir(self):
        """Ensure the cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_cache_path(self, task: str, sample_size: int, seed: int, suffix: str = "") -> str:
        """Get the cache file path for a given task, sample size, and seed."""
        return os.path.join(self.cache_dir, f"{task}_{sample_size}_seed_{seed}{suffix}_cache.pkl")
    
    def get_filtering_calibrator_path(self, task: str, seed: int, target_accuracy: float) -> str:
        """Get path to the filtering calibrator cache file."""
        return os.path.join(self.cache_dir, f"{task}_seed_{seed}_target_{target_accuracy}_filtering_calibrator.pkl")
    
    def get_classifier_path(self, task: str, seed: int) -> str:
        """Get path to the classifier cache file."""
        return os.path.join(self.cache_dir, f"{task}_seed_{seed}_classifier.pkl")
    
    def get_cascade_results_path(self, task: str, seed: int, target_accuracy: float) -> str:
        """Get path to the cascade results cache file."""
        return os.path.join(self.cache_dir, f"{task}_seed_{seed}_target_{target_accuracy}_cascade_results.pkl")
    
    def save_to_cache(self, data: Dict[str, Any], cache_path: str):
        """Save data to cache."""
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_from_cache(self, cache_path: str) -> Dict[str, Any]:
        """Load data from cache."""
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    def cache_exists(self, cache_path: str) -> bool:
        """Check if cache file exists."""
        return os.path.exists(cache_path)
    
    def save_filtering_calibrator(self, task: str, seed: int, target_accuracy: float, filtering_calibrator):
        """Save filtering calibrator to cache."""
        path = self.get_filtering_calibrator_path(task, seed, target_accuracy)
        with open(path, 'wb') as f:
            pickle.dump(filtering_calibrator, f)
    
    def load_filtering_calibrator(self, task: str, seed: int, target_accuracy: float):
        """Load filtering calibrator from cache."""
        path = self.get_filtering_calibrator_path(task, seed, target_accuracy)
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def save_classifier(self, task: str, seed: int, classifier, chunk_size: int):
        """Save classifier and metadata to cache."""
        path = self.get_classifier_path(task, seed)
        with open(path, 'wb') as f:
            pickle.dump({
                'classifier': classifier,
                'chunk_size': chunk_size
            }, f)
    
    def load_classifier(self, task: str, seed: int):
        """Load classifier from cache."""
        path = self.get_classifier_path(task, seed)
        with open(path, 'rb') as f:
            d = pickle.load(f)
            return d['classifier'], d['chunk_size']
    
    def save_cascade_results(self, task: str, seed: int, target_accuracy: float, cascade_results: Dict[str, Any]):
        """Save cascade results to cache."""
        path = self.get_cascade_results_path(task, seed, target_accuracy)
        with open(path, 'wb') as f:
            pickle.dump(cascade_results, f)
    
    def load_cascade_results(self, task: str, seed: int, target_accuracy: float) -> Dict[str, Any]:
        """Load cascade results from cache."""
        path = self.get_cascade_results_path(task, seed, target_accuracy)
        with open(path, 'rb') as f:
            return pickle.load(f)