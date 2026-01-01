
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from task_cascades.filtering.data_filtering_utils import construct_filtered_doc
from task_cascades.predictors.predictors import PREDICTORS, ORACLE_PREDICTOR, TASK_PROMPT_DICT
from task_cascades.config.consts import CANDIDATE_FRACTIONS


class Calibrator:
    """
    A calibrator that uses Platt scaling (logistic regression) to calibrate probabilities.
    """
    def __init__(self):
        self.platt = LogisticRegression(random_state=0)
        
    def fit(self, X, y):
        X = np.asarray(X).reshape(-1, 1)
        y = np.asarray(y)
        self.platt.fit(X, y)
        return self

    def predict(self, X):        
        X = np.asarray(X).reshape(-1, 1)
        predictions = self.platt.predict_proba(X)[:, 1]
        return predictions

def train_filtering_calibrator(train_df_with_chunks: pd.DataFrame, task_name: str) -> Tuple[Calibrator, pd.DataFrame]:
    """
    Train the data filtering calibrator which predicts if filtered documents preserve ground truth.
    
    Args:
        train_df_with_chunks: DataFrame with documents and their chunks
        task_name: Name of the task
        
    Returns:
        Tuple containing:
        - filtering_calibrator: Calibrator for data filtering confidence
        - train_set: DataFrame with training data and calibrated results
    """
    train_samples = []
    for _, row in train_df_with_chunks.iterrows():
        seen_num_chunks = set()
        for frac in CANDIDATE_FRACTIONS:
            num_chunks = max(1, int(round(frac * len(row["chunks"]))))
            if num_chunks in seen_num_chunks:
                continue
            seen_num_chunks.add(num_chunks)
            filtered_doc, avg_confidence_score = construct_filtered_doc(row["text"], row["chunks"], num_chunks)
            row_dict = row.to_dict()
            row_dict["filtered_text"] = filtered_doc
            row_dict["data_filter_conf"] = avg_confidence_score
            row_dict["selection_fraction"] = frac
            train_samples.append(row_dict)
    
    # Run the ORACLE_PREDICTOR on the train samples to obtain oracle outcomes
    def process_oracle(sample):
        pred, _, cost = PREDICTORS[ORACLE_PREDICTOR](TASK_PROMPT_DICT[task_name], text=sample["filtered_text"])
        sample["prediction"] = pred
        return sample
    
    with ThreadPoolExecutor(max_workers=32) as executor:
        oracle_results = list(tqdm(
            executor.map(process_oracle, train_samples),
            total=len(train_samples),
            desc=f"Running {ORACLE_PREDICTOR} on train samples for calibration"
        ))
    
    train_set = pd.DataFrame(oracle_results)
    train_set["data_filter_correct"] = train_set["prediction"] == train_set["label"]
    print(f"Accuracy for data filtering: {np.mean(train_set['data_filter_correct']):.5f}")

    # Train filtering calibrator on the training portion.
    filtering_calibrator = Calibrator()
    filtering_calibrator.fit(train_set["data_filter_conf"], train_set["data_filter_correct"])
    calibrated_filter = filtering_calibrator.predict(train_set["data_filter_conf"])
    print(f"Calibration error for data filtering: {np.mean(np.abs(calibrated_filter - train_set['data_filter_correct'])):.5f}")
    print(f"Number of distinct calibrated values: {len(np.unique(calibrated_filter))}")
    
    return filtering_calibrator