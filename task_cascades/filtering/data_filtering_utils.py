"""
This file contains the data filtering methods for the game reviews.

Each method is a function that takes a document and returns a filtered document and a confidence score.
"""

from litellm import completion, completion_cost, Cache, embedding
import numpy as np
import sklearn
import json
from typing import Tuple
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from typing import List
import tiktoken
import re
import litellm
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
from task_cascades.config.consts import SENTINEL_CONF


litellm.cache = Cache(type="disk")

# Initialize tokenizer
ENCODING = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 8192

def format_lines(lines: list[str], ranges: list[dict] = None, include_line_numbers: bool = False) -> str:
    if not ranges:
        if include_line_numbers:
            return '\n'.join(f"Line #{i+1}. {line}" for i, line in enumerate(lines))
        else:
            return '\n'.join(lines)
    else:
        lines_with_numbers = []
        for range_dict in ranges:
            start = range_dict["start_line"] - 1  # Convert to 0-based index
            end = range_dict["end_line"] - 1
            for i in range(start, end + 1):
                if i < len(lines):  # Ensure we don't go past end of lines
                    indicator = "..." if i > end else ""
                    if include_line_numbers:
                        lines_with_numbers.append(f"Line #{i+1}. {lines[i]}{indicator}")
                    else:
                        lines_with_numbers.append(f"{lines[i]}{indicator}")
        return '\n'.join(lines_with_numbers)

def truncate_text(text: str) -> str:
    """Truncate text to MAX_TOKENS tokens."""
    tokens = ENCODING.encode(text)
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
        return ENCODING.decode(tokens)
    return text if text else "N/A"

def embed_texts(texts: List[str], return_cost: bool = False) -> list[list[float]]:
    # Strip all non alphanumeric, space, newline, tab, carriage return, #, $, punctuation, and known special characters
    texts = [re.sub(r'[^a-zA-Z0-9 \n\t\r#$!@#$%^&*()_+-=[]{}|;:,.<>?~]', '', text) for text in texts]
    
    # Strip line numbers
    texts = [re.sub(r'Line #\d+', '', text) for text in texts]
    
    inputs = [truncate_text(text) for text in texts]
    assert all(text for text in inputs), "All texts must be non-empty"
    
    try:
        query_embeddings_response = embedding(
            # model="azure/text-embedding-3-small",
            model="text-embedding-3-small",
            input=inputs,
            # api_key=os.getenv("AZURE_API_KEY_EMBEDDING"),
            # api_base=os.getenv("AZURE_API_BASE_EMBEDDING"),
            # api_version=os.getenv("AZURE_API_VERSION_EMBEDDING"),
            caching=True,
            num_retries=3
        )
    except Exception as e:
        # Print traceback and raise error
        print(f"Error embedding texts: {inputs}")
        import traceback
        traceback.print_exc(e)
        raise e
    
    if return_cost:
        return [item['embedding'] for item in query_embeddings_response.data], completion_cost(query_embeddings_response)
    else:
        return [item['embedding'] for item in query_embeddings_response.data]

def split_into_lines(text: str, line_length: int = 50) -> list[str]:
    """Split text into lines of specified length, trying to break at word boundaries."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        # Add space before word except at start of line
        word_length = len(word)
        space_length = 1 if current_line else 0
        
        if current_length + space_length + word_length <= line_length:
            if current_line:  # Add space if not start of line
                current_line.append(' ')
                current_length += 1
            current_line.append(word)
            current_length += word_length
        else:
            # Complete current line
            if current_line:
                lines.append(''.join(current_line))
            # Start new line with current word
            current_line = [word]
            current_length = word_length
    
    # Add final line if any
    if current_line:
        lines.append(''.join(current_line))
    

    return lines

def split_into_chunks(text: str, chunk_size: int, chunk_overlap: int = 1) -> list[str]:
    lines = split_into_lines(text)
    chunks = []
    for i in range(0, len(lines), chunk_size - chunk_overlap):
        chunk = lines[i:i + chunk_size]
        if chunk:  # Only add non-empty chunks
            chunks.append("\n".join(chunk))
    return chunks

def split_into_chunks_return_map(text: str, chunk_size: int, chunk_overlap: int = 1) -> Tuple[list[str], dict[str, Tuple[int, int]]]:
    lines = split_into_lines(text)
    chunks = []
    chunk_map = []
    for i in range(0, len(lines), chunk_size - chunk_overlap):
        chunk = lines[i:i + chunk_size]
        if chunk:  # Only add non-empty chunks
            chunk_text = "\n".join(chunk)
            chunks.append(chunk_text)
            chunk_map.append((i, i + len(chunk)))
            
    return chunks, chunk_map




class DocChunk(BaseModel):
    chunk: str
    confidence_score: float
    start_line: int
    end_line: int

def chunk_doc(document: str, chunk_size: int, classifier: sklearn.base.BaseEstimator, **kwargs) -> Tuple[List[DocChunk], float]:
    # Get the classifier prediction
    # First get the chunks and embeddings
    chunks, chunk_map = split_into_chunks_return_map(document, chunk_size, chunk_overlap=0)
    assert len(chunks) > 0, f"No chunks found for document: {document}"
    
    # Check if classifier is position-based (doesn't need embeddings)
    is_position_based = hasattr(classifier, '__class__') and classifier.__class__.__name__ == 'PositionBasedScorer'
    
    if is_position_based:
        # For position-based scoring, no need for embeddings - just pass dummy embeddings
        # and use chunk positions
        chunk_positions = list(range(len(chunks)))
        confidence_scores = classifier.predict_proba(chunks, chunk_positions)[:, 1]
        cost = 0  # No embedding cost for position-based scoring
    else:
        chunks_embeddings, cost = embed_texts(chunks, return_cost=True)
        if not chunks_embeddings:
            return [], 0
        
        # Get the confidence scores
        confidence_scores = classifier.predict_proba(chunks_embeddings)[:, 1]
    
    # Replace nan with 1 since we want to be conservative and treat unknown confidence as high
    confidence_scores = np.nan_to_num(confidence_scores, nan=1)
    
    chunks_with_metadata = []
    for i in range(len(chunks)):
        chunks_with_metadata.append(DocChunk(chunk=chunks[i], confidence_score=confidence_scores[i], start_line=chunk_map[i][0], end_line=chunk_map[i][1]))
    
    assert len(chunks_with_metadata) > 0, "No chunks found"
    
    # Sort chunks by confidence score descending
    chunks_with_metadata.sort(key=lambda x: x.confidence_score, reverse=True)
    
    return chunks_with_metadata, cost

def chunk_and_get_confidences(df: pd.DataFrame, chunk_size: int, classifier: sklearn.base.BaseEstimator):    
    # Chunk the document
    def process_row(row):
        chunks, cost = chunk_doc(row["text"], chunk_size, classifier)
        result_row = row.to_dict()
        result_row.update({
            "chunks": chunks,
            "chunking_cost": cost
        })
        return result_row
        
    with ThreadPoolExecutor(max_workers=32) as executor:
        method_results = list(tqdm(
            executor.map(process_row, [row for _, row in df.iterrows()]),
            total=len(df),
            desc="Chunking and getting confidences..."
        ))
    return pd.DataFrame(method_results)

def construct_filtered_doc(document: str, chunks: List[DocChunk], num_chunks: int) -> Tuple[str, float]:            
    # Compute a combined confidence for the unselected chunks
    confidence_scores = [chunk.confidence_score for chunk in chunks[num_chunks:]]
    avg_confidence_score = np.prod([1 - score for score in confidence_scores]) if len(confidence_scores) > 0 else 1
    
    if len(confidence_scores) > 0 and np.mean(confidence_scores) == 0:
        print(f"All information is selected for document: {document}")
        
    if np.isnan(avg_confidence_score):
        raise ValueError(f"Avg confidence score is NaN. We tried to select {num_chunks} chunks and there are {len(chunks)} chunks. We got confidences {confidence_scores}")
    
    filtered_doc = "\n".join([chunk.chunk for chunk in chunks[:num_chunks]])
    return filtered_doc, avg_confidence_score * SENTINEL_CONF