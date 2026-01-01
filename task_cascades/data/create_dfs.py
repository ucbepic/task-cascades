import pandas as pd
import numpy as np
from typing import List, Any
import sklearn.base
from task_cascades.predictors.predictors import PREDICTORS, BASELINE_PREDICTOR, ORACLE_PREDICTOR, TASK_PROMPT_DICT, PROMPT_TO_TASK_TYPE_DICT
from task_cascades.config.consts import CANDIDATE_FRACTIONS, MIN_TRAINING_SAMPLES
from task_cascades.filtering.data_filtering_utils import chunk_and_get_confidences, construct_filtered_doc

import concurrent.futures
from tqdm import tqdm
import uuid

def prepare_data(
    task: str,
    df: pd.DataFrame, 
    documents: List[Any], 
    sample_size: int,
    train_split: float,
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, List[Any], np.ndarray]:
    """Prepare train/test splits of the data."""
    np.random.seed(random_seed)
    sample_indices = np.random.permutation(len(df))[:sample_size]
    documents = [documents[i] for i in sample_indices]
    df = df.iloc[sample_indices].reset_index(drop=True)
    
    # Add uuid to the dataframe
    df["uuid"] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    num_training_samples = int(len(df) * train_split)
    if num_training_samples < MIN_TRAINING_SAMPLES:
        print(f"Not enough training samples for {task}. Need {MIN_TRAINING_SAMPLES} but got {num_training_samples}. Using {MIN_TRAINING_SAMPLES} training samples.")
        num_training_samples = MIN_TRAINING_SAMPLES
    
    train_indices = np.random.permutation(len(df))[:num_training_samples]
    
    prompt = TASK_PROMPT_DICT[task]
    
    # Label the documents
    def process_doc(doc, predictor: str):
        label, confidence, cost = PREDICTORS[predictor](prompt, text=doc, task_type=PROMPT_TO_TASK_TYPE_DICT[task])
        return label, confidence, cost
    
    # Create labels and costs using ThreadPoolExecutor for parallel processing
    
    labels = []
    costs = []
    
    # Process documents in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        # Create a list of futures
        futures = [executor.submit(process_doc, doc, ORACLE_PREDICTOR) for doc in documents]
        
        # Process results in the order of submission
        for i, future in enumerate(tqdm(futures, total=len(documents), desc="Labeling documents with oracle")):
            label, _, cost = future.result()
            labels.append(label)
            costs.append(cost)
    
    # Add labels and costs to the dataframe
    df["label"] = labels
    df["oracle_cost"] = costs
    
    # Do it again for the baseline predictor
    labels = [] # Re-initialize for baseline
    costs = [] # Re-initialize for baseline
    confidences = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_doc, doc, BASELINE_PREDICTOR) for doc in documents]
        # Process results in the order of submission
        for future in tqdm(futures, total=len(documents), desc="Labeling documents with baseline"):
            label, confidence, cost = future.result()
            
            labels.append(label)
            costs.append(cost)
            confidences.append(confidence)
            
    df["baseline_prediction"] = labels
    df["baseline_cost"] = costs
    df["baseline_confidence"] = confidences
    
    # Update train and test dataframes with the labels and costs
    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.drop(train_df.index).reset_index(drop=True)
    
    return train_df, test_df, documents, train_indices


def load_dataset(task: str) -> tuple[pd.DataFrame, List[Any]]:
    """Load and return the appropriate dataset based on task type."""
    if task == "legal_doc":
        df = pd.read_csv("expt_data/legal_cuad.csv")
        df = df.drop_duplicates(subset=["document"])
        df["text"] = df["document"]
        documents = df["document"].tolist()
    elif task == "game_review":
        df = pd.read_csv("expt_data/review_per_row_top10_filtered.csv")
        df = df.drop_duplicates(subset=["review_text"])
        df["text"] = df["review_text"]
        documents = df["review_text"].tolist()
    elif task == "enron":
        df = pd.read_csv("expt_data/enron.csv")
        df = df.drop_duplicates(subset=["text"])
        documents = df["text"].tolist()
    elif task == "wiki_talk":
        df = pd.read_csv("expt_data/wiki_talk.csv")
        df = df.drop_duplicates(subset=["document"])
        df["text"] = df["document"]
        documents = df["document"].tolist()
    elif task == "court_opinion":
        df = pd.read_csv("expt_data/court_opinions.csv")
        df = df.drop_duplicates(subset=["opinion_text"])
        df["text"] = df["opinion_text"]
        documents = df["text"].tolist()
    elif task == "screenplay":
        df = pd.read_csv("expt_data/screenplays.csv")
        df = df.drop_duplicates(subset=["text"])
        documents = df["text"].tolist()
    elif task == "sms_spam":
        df = pd.read_csv("expt_data/sms_spam.csv")
        df = df.drop_duplicates(subset=["text"])
        documents = df["text"].tolist()
    elif task == "fever":
        df = pd.read_csv("expt_data/fever_processed.csv")
        df = df.drop_duplicates(subset=["text"])
        documents = df["text"].tolist()
    elif task == "ag_news":
        df = pd.read_csv("expt_data/agnews_test.csv")
        df = df.drop_duplicates(subset=["text"])
        documents = df["text"].tolist()
    elif task == "pubmed":
        df = pd.read_csv("expt_data/random-pubmed-articles.csv")
        df = df.drop_duplicates(subset=["article"])
        # Drop empty string articles
        df = df[df["article"].str.strip() != ""].reset_index(drop=True)
        df["text"] = df["article"]
        documents = df["article"].tolist()
    elif task == "sec":
        df = pd.read_csv("expt_data/sec10k.csv")
        df = df.drop_duplicates(subset=["text"])
        documents = df["text"].tolist()
    elif task == "biodex":
        df = pd.read_csv("expt_data/biodex_dataset.csv")
        df = df.drop_duplicates(subset=["text"])
        documents = df["text"].tolist()
    elif task == "longhealth":
        df = pd.read_csv("expt_data/longhealth.csv")
        df = df.drop_duplicates(subset=["text"])
        documents = df["text"].tolist()
    else:
        raise ValueError(f"Invalid task: {task}")
    
    return df, documents

def apply_filtering_calibrator_to_dataframe(
    chunked_df: pd.DataFrame,
    data_filtering_calibrator: sklearn.base.BaseEstimator,
) -> pd.DataFrame:
    """
    Apply a chunk classifier to documents in a dataframe and create filtered versions at different sizes.
    Explodes the dataframe to have one row per document per fraction.
    
    Args:
        df: DataFrame containing documents to process (must have a "text" column)
        chunk_classifier: Trained classifier that can predict relevance scores for document chunks
        data_filtering_calibrator: Calibrator to adjust confidence scores
        chunk_size: Size of chunks to split documents into
        
    Returns:
        Expanded DataFrame with one row per document per fraction
    """
    # Step 2: Prepare for exploding the dataframe
    expanded_rows = []
    
    for _, row in chunked_df.iterrows():
        base_row = row.to_dict()
        chunks = base_row.pop("chunks", [])  # Remove chunks from the base row
        
        if not chunks:
            # Handle documents with no chunks
            for fraction in CANDIDATE_FRACTIONS:
                new_row = base_row.copy()
                new_row["fraction"] = fraction
                new_row["filtered_text"] = ""
                new_row["raw_confidence"] = 0.0
                expanded_rows.append(new_row)
            continue
        
        # Process each fraction
        num_chunks_seen = set()
        for fraction in CANDIDATE_FRACTIONS:
            new_row = base_row.copy()
            new_row["fraction"] = fraction
            
            # Calculate number of chunks to keep based on fraction
            num_chunks = max(1, int(len(chunks) * fraction))
            if num_chunks in num_chunks_seen:
                continue
            num_chunks_seen.add(num_chunks)
            
            # Get filtered document and its raw confidence score
            filtered_doc, raw_confidence = construct_filtered_doc(
                row["text"], 
                chunks, 
                num_chunks
            )
            
            new_row["filtered_text"] = filtered_doc
            new_row["raw_confidence"] = raw_confidence
            
            expanded_rows.append(new_row)

        # # Ensure a single-chunk version exists (use minimal fraction value)
        # if 1 not in num_chunks_seen:
        #     single_row = base_row.copy()
        #     # Represent the fraction for 1 chunk as the reciprocal of total chunks (rounded)
        #     single_row["fraction"] = round(1.0 / len(chunks), 4)

        #     filtered_doc, raw_confidence = construct_filtered_doc(
        #         row["text"],
        #         chunks,
        #         1,
        #     )

        #     single_row["filtered_text"] = filtered_doc
        #     single_row["raw_confidence"] = raw_confidence
        #     expanded_rows.append(single_row)
    
    # Create expanded dataframe
    expanded_df = pd.DataFrame(expanded_rows)
    assert len(expanded_df) >= len(chunked_df), f"Expanded dataframe has {len(expanded_df)} rows, expected at least {len(chunked_df)}"
    assert len(expanded_df) <= len(chunked_df) * len(CANDIDATE_FRACTIONS), f"Expanded dataframe has {len(expanded_df)} rows, expected at most {len(chunked_df) * len(CANDIDATE_FRACTIONS)}"
    
    # Reshape confidences for the calibrator
    raw_confidences = expanded_df["raw_confidence"].values.reshape(-1, 1)
    
    # Apply calibrator to all confidences at once
    calibrated_confidences = data_filtering_calibrator.predict(raw_confidences)
    
    # Add calibrated confidences to the dataframe
    expanded_df["data_filtering_confidence"] = calibrated_confidences
    
    # Drop the raw confidence column
    expanded_df = expanded_df.drop(columns=["raw_confidence"])
    
    return expanded_df

