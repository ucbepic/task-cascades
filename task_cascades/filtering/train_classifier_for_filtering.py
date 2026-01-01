import random
import pandas as pd
from task_cascades.filtering.data_filtering_utils import split_into_lines, format_lines, embed_texts
from task_cascades.predictors.predictors import ORACLE_PREDICTOR, TASK_PROMPT_DICT, process_doc, PROMPT_TO_TASK_TYPE_DICT
from typing import List, Tuple
import json
from litellm import completion, Cache
import litellm
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from concurrent.futures import ThreadPoolExecutor

litellm.cache = Cache(type="disk")
load_dotenv()
MAX_CHUNK_SIZE = 50
MIN_CHUNK_SIZE = 1

def find_relevant_chunks(doc: str, task_prompt: str, ground_truth: bool) -> list[dict]:
    # Convert doc into lines
    lines = split_into_lines(doc)
    lines_str = format_lines(lines)

    
    prompt = f"""Given a document, identify the MINIMAL line ranges that are needed to do the following task: ```{task_prompt}```

The ground truth answer for this document is: {ground_truth}

Document Text:
```{lines_str}```

Output the relevant line ranges (start_line and end_line, inclusive) in this JSON format:
{{"ranges": [
    {{"start_line": <number>, "end_line": <number>}},
    ...
]}}

Only include ranges that contain information needed to determine the answer.
I will concatenate all the ranges you return to form a single document of text, which should be enough information to determine the answer.
Each range should represent a coherent chunk of text that provides evidence for or against the task.
Your line ranges should be the MINIMAL information needed to determine the answer.
Important: The ranges must not overlap - each line should only appear in one range."""

    try:
        response = completion(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            caching=True
        )
        return json.loads(response.choices[0].message.content)["ranges"]
    except Exception as e:
        print(f"Error in find_relevant_chunks: {e}")
        return []
    
def get_relevant_chunk_size(df: pd.DataFrame, task_prompt: str, task_type: str = "binary") -> Tuple[int, List[str]]:
    # Print accuracy
    print(f"Accuracy: {sum(df['baseline_prediction'] == df['label'])/len(df)}")
    
    # Subset df to get mistakes
    mistakes = df[df["baseline_prediction"] != df["label"]]
    print(f"There are {len(mistakes)} mistakes in the sample")
    
    # Use the full df
    
    # Get initial ranges once
    with ThreadPoolExecutor(max_workers=64) as executor:
        kwargs = {}
        
        futures = [
            executor.submit(find_relevant_chunks, row["text"], task_prompt, row["label"], **kwargs) 
            for _, row in df.iterrows()
        ]
        all_ranges = [future.result() for future in futures]
    
    max_iterations = 3
    target_accuracy = 0.80
    
    # Track best performance
    best_accuracy = 0
    best_ranges = None
    best_chunks = None

    
    for iteration in range(max_iterations):
        avg_lines = sum(
            sum(r["end_line"] - r["start_line"] + 1 for r in ranges)
            for ranges in all_ranges
        ) / sum(len(ranges) for ranges in all_ranges)
        print(f"\nIteration {iteration + 1}/{max_iterations}; current avg chunk size: {avg_lines:.2f}")
        
        # Format chunks with current ranges
        all_chunks = []
        for ranges, (_, row) in zip(all_ranges, df.iterrows()):
            lines = split_into_lines(row["text"])
            chunk_text = format_lines(lines, ranges)
            all_chunks.append(chunk_text)

            
        # Calculate accuracy with current chunks using parallel processing
        with ThreadPoolExecutor(max_workers=64) as executor: 
            futures = [
                executor.submit(process_doc, task_prompt=task_prompt, model=ORACLE_PREDICTOR, text=chunk, task_type=task_type) 
                for chunk in all_chunks
            ]
            predictions = [future.result()[0] for future in futures]
        
        # Calculate accuracy
        correct = sum(
            1 for pred, (_, row) in zip(predictions, df.iterrows())
            if pred == row["label"]
        )
        accuracy = correct / len(df)
        print(f"Current accuracy: {accuracy:.2%}")
        
        # Update best performance if current is significantly better
        if accuracy > best_accuracy + 0.05:
            best_accuracy = accuracy
            best_ranges = [[r.copy() for r in ranges] for ranges in all_ranges]  # Deep copy to preserve state
            best_chunks = all_chunks.copy()
            print(f"New best accuracy: {best_accuracy:.2%}")
        
        if round(accuracy, 2) >= round(target_accuracy, 2):
            print(f"Target accuracy reached at iteration {iteration + 1}")
            break
            
        # Expand chunks if needed and not at last iteration
        if iteration < max_iterations - 1:
            for ranges in all_ranges:
                for range_dict in ranges:
                    # range_dict["start_line"] = max(1, range_dict["start_line"] - 1)
                    range_dict["end_line"] += 1
            print("Expanded chunks for next iteration")
    
    # Use best performing ranges/chunks if we didn't hit target accuracy
    if best_accuracy < target_accuracy:
        print(f"\nUsing best performing ranges/chunks with accuracy: {best_accuracy:.2%}")
        all_ranges = best_ranges
        all_chunks = best_chunks
    
    # Calculate final stats
    avg_lines = sum(
        sum(r["end_line"] - r["start_line"] + 1 for r in ranges)
        for ranges in all_ranges
    ) / sum(len(ranges) for ranges in all_ranges)
    print(f"\nFinal average relevant lines per chunk: {avg_lines:.2f}")
    
    returned_chunk_size = int(avg_lines)
    if returned_chunk_size > MAX_CHUNK_SIZE:
        returned_chunk_size = MAX_CHUNK_SIZE
        print(f"Returning chunk size of {MAX_CHUNK_SIZE} because average chunk size was too large")
    elif returned_chunk_size < MIN_CHUNK_SIZE:
        returned_chunk_size = MIN_CHUNK_SIZE
        print(f"Returning chunk size of {MIN_CHUNK_SIZE} because average chunk size was too small")
        
    # If the chunk size is > 1/10 the total number of lines, set it to 1/10 the avg number
    # avg_num_lines = sum(len(split_into_lines(row["text"])) for _, row in df.iterrows()) / len(df)
    # if returned_chunk_size > avg_num_lines / 10:
    #     returned_chunk_size = max(1, int(avg_num_lines / 10))
    #     print(f"Returning chunk size of {returned_chunk_size} because it was too large")
        
    return all_ranges, all_chunks, returned_chunk_size



RELEVANT_CHUNKS_DOC_PROMPT = """Given this document, identify ALL the line ranges that are needed to perform the following task: {task_prompt}

Here is the document with line numbers:
{formatted_lines}

The ground truth answer for this document is: {ground_truth}

Your task is to identify the starting line numbers of chunks that are relevant for the task.
Each chunk will be {chunk_size} lines long.

Return ONLY the line numbers in a JSON array format like this:
{{"starting_lines": [1, 5, 10]}}

Important:
- Only include starting lines where the next {chunk_size} lines contain relevant information
- You must include at least one range that justifies the ground truth answer"""
    
def get_relevant_chunks_for_doc(document: str, chunk_size: int, ground_truth: bool, task_prompt: str, **kwargs) -> Tuple[List[str]]:
    # Query gpt-4o for the starting line for all relevant chunks with size chunk_size
    # Split review into lines
    lines = split_into_lines(document)
    
    # Format lines with line numbers
    formatted_lines = format_lines(lines)
    
    # Prompt to find relevant starting lines
    prompt = RELEVANT_CHUNKS_DOC_PROMPT.format(
        formatted_lines=formatted_lines,
        chunk_size=chunk_size,
        ground_truth=ground_truth,
        task_prompt=task_prompt,
        metadata=str(kwargs)
    )

    try:
        res = completion(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            caching=True
        )
        
        response_json = json.loads(res.choices[0].message.content)
        starting_lines = response_json.get("starting_lines", [])
        
        # Convert to list of dicts with start and end lines
        ranges = []
        for start in starting_lines:
            if start <= len(lines):
                end = min(start + chunk_size - 1, len(lines))
                ranges.append({
                    "start_line": start,
                    "end_line": end
                })
                
        return ranges
        
    except Exception as e:
        print(f"Error finding relevant chunks: {e}")
        return []

def get_chunks_from_ranges(review_text: str, ranges: List[dict], chunk_size: int) -> Tuple[List[str], List[str]]:
    """Extract relevant and irrelevant chunks from a review based on ranges.
    
    Args:
        review_text: The full review text
        ranges: List of dicts with start_line and end_line
        chunk_size: Size of chunks to extract
        
    Returns:
        Tuple of (relevant_chunks, irrelevant_chunks)
    """
    lines = split_into_lines(review_text)
    
    # Get relevant chunks from ranges
    relevant_chunks = []
    for r in ranges:
        chunk_lines = lines[r["start_line"]-1:r["end_line"]]
        if chunk_lines:
            relevant_chunks.append(format_lines(chunk_lines))
            
    # Get irrelevant chunks
    irrelevant_chunks = []
    current_line = 1
    used_ranges = sorted(ranges, key=lambda x: x["start_line"])
    
    for r in used_ranges:
        # Get chunks before this range
        while current_line < r["start_line"]:
            end_line = min(current_line + chunk_size - 1, r["start_line"] - 1)
            if end_line - current_line + 1 >= chunk_size:  # Only add if full chunk size
                chunk_lines = lines[current_line-1:end_line]
                if chunk_lines:
                    irrelevant_chunks.append(format_lines(chunk_lines))
            current_line = end_line + 1
        current_line = r["end_line"] + 1
    
    # Get remaining chunks after last range
    while current_line <= len(lines):
        end_line = min(current_line + chunk_size - 1, len(lines))
        if end_line - current_line + 1 >= chunk_size:  # Only add if full chunk size
            chunk_lines = lines[current_line-1:end_line]
            if chunk_lines:
                irrelevant_chunks.append(format_lines(chunk_lines))
        current_line = end_line + 1
        
    return relevant_chunks, irrelevant_chunks

def get_relevant_chunks_for_all_docs(df: pd.DataFrame, chunk_size: int, task_name: str) -> Tuple[List[str], List[str], float]:
    """Get relevant and irrelevant chunks for all documents.
    
    Args:
        df: DataFrame with documents
        chunk_size: Size of chunks to extract
        
    Returns:
        Tuple of (all_relevant_chunks, all_irrelevant_chunks, avg_relevant_chunks_per_doc)
    """
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = []
        
        task_prompt = TASK_PROMPT_DICT[task_name]
        task_type = PROMPT_TO_TASK_TYPE_DICT[task_name]
        
        # Submit tasks for each review
        for _, row in df.iterrows():
            future = executor.submit(
                get_relevant_chunks_for_doc,
                document=row["text"],
                chunk_size=chunk_size,
                ground_truth=row['label'],
                task_prompt=task_prompt
            )
            futures.append((future, row["text"]))
        
        # Collect results
        all_relevant_chunks = []
        all_irrelevant_chunks = []
        num_relevant_chunks_per_doc = []  # Track number of relevant chunks per document
        
        for future, review_text in futures:
            try:
                ranges = future.result()
                if ranges:
                    relevant, irrelevant = get_chunks_from_ranges(
                        review_text=review_text,
                        ranges=ranges,
                        chunk_size=chunk_size
                    )
                    all_relevant_chunks.extend(relevant)
                    all_irrelevant_chunks.extend(irrelevant)
                    num_relevant_chunks_per_doc.append(len(relevant))
                else:
                    num_relevant_chunks_per_doc.append(0)
            except Exception as e:
                print(f"Error processing review: {e}")
                num_relevant_chunks_per_doc.append(0)
                continue
        
        # Calculate average number of relevant chunks per document
        avg_relevant_chunks = np.mean(num_relevant_chunks_per_doc)
        print(f"\nAverage number of relevant chunks per document: {avg_relevant_chunks:.2f}")
                
        return all_relevant_chunks, all_irrelevant_chunks, avg_relevant_chunks

class CustomLogisticRegression(nn.Module):
    def __init__(self, input_dim: int, query_embedding: np.ndarray):
        super().__init__()
        # Initialize linear layer with query embedding
        self.linear = nn.Linear(input_dim, 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(query_embedding.reshape(1, -1)).float())
            
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def predict_proba(self, X):
        # For sklearn compatibility
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            probs = self.forward(X_tensor).numpy()
            return np.hstack([1 - probs, probs])
            
    def predict(self, X):
        # For sklearn compatibility
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

def train_classifier(query_embedding: list[float], relevant_chunks: List[str], irrelevant_chunks: List[str]):
    # Set random seeds for reproducibility
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Print a sample of relevant and irrelevant chunks
    print(f"Sample relevant chunk: {relevant_chunks[0]}")
    print(f"Sample irrelevant chunk: {irrelevant_chunks[0]}" if len(irrelevant_chunks) > 0 else "No irrelevant chunks")
    
    # Convert embeddings to numpy arrays in batches of 1000
    batch_size = 500
    relevant_chunks_embeddings = []
    print(f"Processing {len(relevant_chunks)} relevant chunks in batches of 1000...")
    for i in range(0, len(relevant_chunks), batch_size):
        batch = relevant_chunks[i:i+batch_size]
        print(f"Processing relevant chunks batch {i//batch_size + 1}/{(len(relevant_chunks)-1)//batch_size + 1}")
        batch_embeddings = embed_texts(batch)
        relevant_chunks_embeddings.extend(batch_embeddings)
        
    irrelevant_chunks_embeddings = []
    print(f"\nProcessing {len(irrelevant_chunks)} irrelevant chunks in batches of 1000...")
    for i in range(0, len(irrelevant_chunks), batch_size):
        batch = irrelevant_chunks[i:i+batch_size]
        print(f"Processing irrelevant chunks batch {i//batch_size + 1}/{(len(irrelevant_chunks)-1)//batch_size + 1}")
        batch_embeddings = embed_texts(batch)
        irrelevant_chunks_embeddings.extend(batch_embeddings)

    # Combine all embeddings and create labels
    all_embeddings = relevant_chunks_embeddings + irrelevant_chunks_embeddings
    all_labels = [1] * len(relevant_chunks_embeddings) + [0] * len(irrelevant_chunks_embeddings)
    
    # Convert to numpy arrays
    X = np.array(all_embeddings)
    y = np.array(all_labels)
    
    # Shuffle data
    shuffle_indices = np.random.permutation(len(X))
    X = X[shuffle_indices]
    y = y[shuffle_indices]
    
    # Split into train/val (50/50)
    train_size = int(0.5 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:]
    y_val = y[train_size:]
    
    # Upsample minority class in training set only
    train_relevant_mask = y_train == 1
    train_irrelevant_mask = y_train == 0
    
    X_train_relevant = X_train[train_relevant_mask]
    X_train_irrelevant = X_train[train_irrelevant_mask]
    y_train_relevant = y_train[train_relevant_mask]
    y_train_irrelevant = y_train[train_irrelevant_mask]
    
    # Determine which class is minority and upsample
    if len(X_train_relevant) < len(X_train_irrelevant):
        # Relevant is minority class
        num_samples = len(X_train_irrelevant)
        indices = np.random.choice(len(X_train_relevant), num_samples, replace=True)
        X_train_relevant = X_train_relevant[indices]
        y_train_relevant = y_train_relevant[indices]
    else:
        # Irrelevant is minority class
        num_samples = len(X_train_relevant)
        indices = np.random.choice(len(X_train_irrelevant), num_samples, replace=True)
        X_train_irrelevant = X_train_irrelevant[indices]
        y_train_irrelevant = y_train_irrelevant[indices]
    
    # Combine upsampled data
    X_train = np.vstack([X_train_relevant, X_train_irrelevant])
    y_train = np.concatenate([y_train_relevant, y_train_irrelevant])
    
    # Shuffle again after upsampling
    shuffle_indices = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    
    # Convert to PyTorch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    
    # Print class distribution before training
    print(f"Training set class distribution after upsampling - Relevant: {(y_train == 1).sum().item()}, Irrelevant: {(y_train == 0).sum().item()}")
    print(f"Validation set class distribution - Relevant: {(y_val == 1).sum().item()}, Irrelevant: {(y_val == 0).sum().item()}")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train.reshape(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = CustomLogisticRegression(input_dim, np.array(query_embedding)).to(device)
    
    # Training parameters
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Early stopping parameters
    patience = 5
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(100):  # Max epochs
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_preds = (val_outputs >= 0.5).float()
            val_f1 = f1_score(y_val.cpu().numpy(), val_preds.cpu().numpy())
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} with best validation F1: {best_val_f1:.4f}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_preds = (model(X_train) >= 0.5).cpu().numpy()
        train_acc = accuracy_score(y_train.cpu().numpy(), train_preds)
        train_f1 = f1_score(y_train.cpu().numpy(), train_preds)
        print(f"Training metrics - Accuracy: {train_acc:.4f}, F1 Score: {train_f1:.4f}")
        
        val_preds = (model(X_val) >= 0.5).cpu().numpy()
        val_acc = accuracy_score(y_val.cpu().numpy(), val_preds)
        val_f1 = f1_score(y_val.cpu().numpy(), val_preds)
        print(f"Validation metrics - Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")
    
    return model

def find_similarity_threshold(query_vector: list[float], relevant_chunks: List[str], irrelevant_chunks: List[str]) -> float:
    # Find the embedding similarity threshold for the TASK_PROMPT query vector
    relevant_chunks_embeddings = embed_texts(relevant_chunks)
    irrelevant_chunks_embeddings = embed_texts(irrelevant_chunks)
    # Reshape query vector to (1, -1) since it's a single sample
    query_vector = np.array(query_vector).reshape(1, -1)
    # Convert each embedding to numpy array before reshaping
    relevant_similarities = [cosine_similarity(query_vector, np.array(chunk_embedding).reshape(1, -1))[0][0] for chunk_embedding in relevant_chunks_embeddings]
    irrelevant_similarities = [cosine_similarity(query_vector, np.array(chunk_embedding).reshape(1, -1))[0][0] for chunk_embedding in irrelevant_chunks_embeddings]
    
    print(f"Average relevant similarity: {np.mean(relevant_similarities):.2f}")
    print(f"Average irrelevant similarity: {np.mean(irrelevant_similarities):.2f}")
    
    return np.mean(relevant_similarities)

def train_data_filtering(task_name: str, df: pd.DataFrame):
    main_task_prompt = TASK_PROMPT_DICT[task_name]
    
    # Have gpt-4o "scan" the documents that gpt-4o-mini got wrong and figure out the chunks that are relevant
    task_type = PROMPT_TO_TASK_TYPE_DICT[task_name]
    relevant_ranges, relevant_chunks, chunk_size = get_relevant_chunk_size(df, main_task_prompt, task_type)
    
    
    # First, we need to query gpt-4o for all the chunks that are relevant to the task
    relevant_chunks, irrelevant_chunks, avg_relevant_chunks_per_doc = get_relevant_chunks_for_all_docs(df, chunk_size, task_name)
    print(f"Found {len(relevant_chunks)} relevant chunks and {len(irrelevant_chunks)} irrelevant chunks. Average relevant chunks per doc: {avg_relevant_chunks_per_doc:.2f}")
    
    # Throw out empty chunks
    relevant_chunks = [chunk for chunk in relevant_chunks if chunk]
    irrelevant_chunks = [chunk for chunk in irrelevant_chunks if chunk]
    
    query_embeddings = embed_texts([main_task_prompt])
    query_embedding = np.mean(query_embeddings, axis=0) # Average the query embeddings
    
    # Train a classifier to distinguish relevant chunks from irrelevant chunks
    classifier = train_classifier(query_embedding, relevant_chunks, irrelevant_chunks)
    
    return classifier, max(1, int(chunk_size))

def simple_similarity_data_filtering(task_name: str, df: pd.DataFrame):
    """
    Simple baseline data filtering that just uses cosine similarity between chunks and task prompt.
    No classifier training, just chunk documents by avg_lines/10 and score by similarity.
    
    Args:
        task_name: The task name to get the prompt for
        df: DataFrame containing documents
        
    Returns:
        Tuple of (similarity_scorer, chunk_size)
    """
    main_task_prompt = TASK_PROMPT_DICT[task_name]
    
    # Calculate average number of lines across all documents
    total_lines = 0
    for _, row in df.iterrows():
        lines = split_into_lines(row['text'])
        total_lines += len(lines)
    
    avg_lines_per_doc = total_lines / len(df)
    
    # Chunk size is average lines divided by 10
    chunk_size = max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, int(avg_lines_per_doc / 10)))
    
    print(f"Average lines per document: {avg_lines_per_doc:.0f}")
    print(f"Chunk size (lines): {chunk_size}")
    
    # Get query embedding
    query_embeddings = embed_texts([main_task_prompt])
    query_embedding = np.array(query_embeddings[0])
    
    # Create a simple similarity-based scorer
    class SimpleSimilarityScorer:
        def __init__(self, query_embedding):
            self.query_embedding = query_embedding.reshape(1, -1)
            
        def predict_proba(self, chunk_embeddings):
            """
            Score chunks based on cosine similarity to query embedding.
            Returns probabilities in sklearn format: [[prob_irrelevant, prob_relevant], ...]
            """
            if len(chunk_embeddings) == 0:
                return np.array([]).reshape(0, 2)
                
            chunk_embeddings = np.array(chunk_embeddings)
            if len(chunk_embeddings.shape) == 1:
                chunk_embeddings = chunk_embeddings.reshape(1, -1)
                
            # Calculate cosine similarities
            similarities = cosine_similarity(chunk_embeddings, self.query_embedding).flatten()
            
            # Convert similarities to probabilities (similarity is already 0-1 range)
            # Higher similarity = higher probability of being relevant
            prob_relevant = similarities
            prob_irrelevant = 1 - prob_relevant
            
            return np.column_stack([prob_irrelevant, prob_relevant])
            
        def predict(self, chunk_embeddings):
            """Binary predictions based on threshold of 0.5 similarity"""
            probs = self.predict_proba(chunk_embeddings)
            return (probs[:, 1] >= 0.5).astype(int)
    
    similarity_scorer = SimpleSimilarityScorer(query_embedding)
    
    print(f"Created simple similarity scorer with chunk size: {chunk_size}")
    
    return similarity_scorer, chunk_size

def position_based_data_filtering(task_name: str, df: pd.DataFrame):
    """
    Even simpler baseline data filtering that just assigns scores based on chunk position.
    First chunk gets highest score, second chunk gets second highest score, etc.
    No embeddings or similarity calculations - just position-based relevance.
    
    Args:
        task_name: The task name (not actually used, just for consistency)
        df: DataFrame containing documents
        
    Returns:
        Tuple of (position_scorer, chunk_size)
    """
    # Calculate average number of lines across all documents
    total_lines = 0
    for _, row in df.iterrows():
        lines = split_into_lines(row['text'])
        total_lines += len(lines)
    
    avg_lines_per_doc = total_lines / len(df)
    
    # Chunk size is average lines divided by 10
    chunk_size = max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, int(avg_lines_per_doc / 10)))
    
    print(f"Average lines per document: {avg_lines_per_doc:.0f}")
    print(f"Chunk size (lines): {chunk_size}")
    
    # Create a position-based scorer
    class PositionBasedScorer:
        def __init__(self):
            pass
            
        def predict_proba(self, chunk_embeddings, chunk_positions=None):
            """
            Score chunks based on their position in the document.
            Earlier chunks get higher relevance scores.
            Returns probabilities in sklearn format: [[prob_irrelevant, prob_relevant], ...]
            """
            if len(chunk_embeddings) == 0:
                return np.array([]).reshape(0, 2)
            
            num_chunks = len(chunk_embeddings)
            
            # If chunk_positions is not provided, assume sequential ordering
            if chunk_positions is None:
                positions = np.arange(num_chunks)
            else:
                positions = np.array(chunk_positions)
            
            # Calculate relevance scores: earlier chunks get higher scores
            # Use exponential decay: score = exp(-position * decay_rate)
            decay_rate = 0.1
            relevance_scores = np.exp(-positions * decay_rate)
            
            # Normalize to 0-1 range
            if len(relevance_scores) > 1:
                min_score = relevance_scores.min()
                max_score = relevance_scores.max()
                if max_score > min_score:
                    relevance_scores = (relevance_scores - min_score) / (max_score - min_score)
                else:
                    relevance_scores = np.ones_like(relevance_scores) * 0.5
            else:
                relevance_scores = np.array([0.8])  # Single chunk gets high relevance
            
            # Convert to probability format
            prob_relevant = relevance_scores
            prob_irrelevant = 1 - prob_relevant
            
            return np.column_stack([prob_irrelevant, prob_relevant])
            
        def predict(self, chunk_embeddings, chunk_positions=None):
            """Binary predictions based on threshold of 0.5 relevance"""
            probs = self.predict_proba(chunk_embeddings, chunk_positions)
            return (probs[:, 1] >= 0.5).astype(int)
    
    position_scorer = PositionBasedScorer()
    
    print(f"Created position-based scorer with chunk size: {chunk_size}")
    
    return position_scorer, chunk_size