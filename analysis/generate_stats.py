import os
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
import re
from os.path import dirname, abspath

# Add the parent directory to sys.path to allow task_cascades imports
root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)

from task_cascades.data.create_dfs import prepare_data, load_dataset
from task_cascades.predictors.predictors import TASK_PROMPT_DICT

# Constants
RESULTS_DIR = "results"
SEED = 42
SAMPLE_SIZE = 1000
TRAIN_SPLIT = 0.2

def read_experiment_results() -> List[Dict[str, Any]]:
    """Read all experiment result files from the results directory."""
    results = []
    
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory {RESULTS_DIR} does not exist.")
        return results
    
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith("_latest.pkl") and f"seed_{SEED}" in filename:
            file_path = os.path.join(RESULTS_DIR, filename)
            try:
                with open(file_path, 'rb') as f:
                    result = pickle.load(f)
                    results.append(result)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return results

def count_words(text: str) -> int:
    """Count words in a text string."""
    if pd.isna(text) or text == "":
        return 0
    # Simple word count using whitespace splitting
    return len(str(text).split())

def analyze_task_dataset(task: str) -> Dict[str, Any]:
    """Analyze a single task dataset and return statistics."""
    print(f"\n📊 Analyzing task: {task}")
    
    try:
        # Load dataset and prepare data (same as run_experiments.py)
        df, documents = load_dataset(task)
        train_df, test_df, documents, train_indices = prepare_data(
            task, df, documents, SAMPLE_SIZE, TRAIN_SPLIT, random_seed=SEED
        )
        
        # Calculate statistics for test dataset
        test_stats = {}
        
        # Get unique documents in test set
        test_docs = test_df.drop_duplicates(subset=["uuid"])
        
        # Average words per document
        test_docs['word_count'] = test_docs['text'].apply(count_words)
        avg_words = test_docs['word_count'].mean()
        median_words = test_docs['word_count'].median()
        min_words = test_docs['word_count'].min()
        max_words = test_docs['word_count'].max()
        
        # Label distribution
        label_counts = test_docs['label'].value_counts().to_dict()
        total_docs = len(test_docs)
        label_percentages = {label: (count / total_docs) * 100 
                           for label, count in label_counts.items()}
        
        test_stats = {
            'task': task,
            'total_test_documents': total_docs,
            'total_train_documents': len(train_df.drop_duplicates(subset=["uuid"])),
            'avg_words_per_doc': avg_words,
            'median_words_per_doc': median_words,
            'min_words_per_doc': min_words,
            'max_words_per_doc': max_words,
            'label_counts': label_counts,
            'label_percentages': label_percentages,
            'unique_labels': list(label_counts.keys())
        }
        
        print(f"✅ Successfully analyzed {task} from cache")
        return test_stats
        
    except Exception as e:
        print(f"❌ Error analyzing task {task}: {e}")
        return None

def print_dataset_statistics(stats_list: List[Dict[str, Any]]):
    """Print comprehensive dataset statistics."""
    print("\n" + "="*100)
    print("DATASET STATISTICS SUMMARY (Test Sets, Seed 42)")
    print("="*100)
    
    # Summary table
    print(f"\n{'Task':<15} | {'Test Docs':<10} | {'Train Docs':<11} | {'Avg Words':<10} | {'Med Words':<10} | {'Labels':<8} | {'Label Distribution'}")
    print("-" * 100)
    
    for stats in stats_list:
        if stats is None:
            continue
            
        task = stats['task']
        test_docs = stats['total_test_documents']
        train_docs = stats['total_train_documents']
        avg_words = stats['avg_words_per_doc']
        median_words = stats['median_words_per_doc']
        num_labels = len(stats['unique_labels'])
        
        # Create a concise label distribution string
        label_dist_str = ", ".join([f"{label}: {pct:.1f}%" 
                                   for label, pct in sorted(stats['label_percentages'].items())])
        if len(label_dist_str) > 40:
            label_dist_str = label_dist_str[:37] + "..."
        
        print(f"{task:<15} | {test_docs:<10} | {train_docs:<11} | {avg_words:<10.1f} | {median_words:<10.1f} | {num_labels:<8} | {label_dist_str}")
    
    # Detailed statistics for each task
    print("\n" + "="*100)
    print("DETAILED STATISTICS BY TASK")
    print("="*100)
    
    for stats in stats_list:
        if stats is None:
            continue
            
        print(f"\n🎯 Task: {stats['task'].upper()}")
        print("-" * 60)
        print(f"  📚 Documents:")
        print(f"    - Test set: {stats['total_test_documents']:,} documents")
        print(f"    - Train set: {stats['total_train_documents']:,} documents")
        print(f"    - Total: {stats['total_test_documents'] + stats['total_train_documents']:,} documents")
        
        print(f"  📝 Word Statistics (Test Set):")
        print(f"    - Average: {stats['avg_words_per_doc']:.1f} words")
        print(f"    - Median: {stats['median_words_per_doc']:.1f} words")
        print(f"    - Range: {stats['min_words_per_doc']:,} - {stats['max_words_per_doc']:,} words")
        
        print(f"  🏷️  Label Distribution (Test Set):")
        for label in sorted(stats['unique_labels']):
            count = stats['label_counts'][label]
            percentage = stats['label_percentages'][label]
            print(f"    - {label}: {count:,} documents ({percentage:.2f}%)")
    
    # Overall summary across all tasks
    print("\n" + "="*100)
    print("OVERALL SUMMARY ACROSS ALL TASKS")
    print("="*100)
    
    if stats_list:
        total_test_docs = sum(stats['total_test_documents'] for stats in stats_list if stats)
        total_train_docs = sum(stats['total_train_documents'] for stats in stats_list if stats)
        avg_words_overall = np.mean([stats['avg_words_per_doc'] for stats in stats_list if stats])
        median_words_overall = np.median([stats['median_words_per_doc'] for stats in stats_list if stats])
        num_tasks = len([stats for stats in stats_list if stats])
        
        print(f"📊 Dataset Overview:")
        print(f"  - Total tasks analyzed: {num_tasks}")
        print(f"  - Total test documents: {total_test_docs:,}")
        print(f"  - Total train documents: {total_train_docs:,}")
        print(f"  - Average words per document (across tasks): {avg_words_overall:.1f}")
        print(f"  - Median words per document (across tasks): {median_words_overall:.1f}")

def main():
    """Main function to generate dataset statistics."""
    # Specific tasks to analyze
    tasks = ['ag_news', 'court_opinion', 'enron', 'fever', 'game_review', 'legal_doc', 'wiki_talk', 'pubmed']
    
    print(f"🔍 Analyzing datasets for {len(tasks)} tasks with seed {SEED}...")
    print(f"📋 Tasks: {', '.join(tasks)}")
    
    # Analyze each task
    all_stats = []
    for task in tasks:
        if task in TASK_PROMPT_DICT:  # Only analyze tasks we have prompts for
            stats = analyze_task_dataset(task)
            if stats:
                all_stats.append(stats)
        else:
            print(f"⚠️  Skipping {task}: No task prompt defined")
    
    # Print comprehensive statistics
    if all_stats:
        print_dataset_statistics(all_stats)
    else:
        print("❌ No statistics could be generated.")

if __name__ == "__main__":
    main()
