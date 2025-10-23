from sklearn.metrics import f1_score
import numpy as np

import pandas as pd
from datetime import datetime

def make_json_safe(d):
    safe = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            safe[k] = v.tolist()
        elif isinstance(v, (pd.Timestamp, datetime)):
            safe[k] = v.isoformat()
        elif isinstance(v, pd.Timedelta):
            safe[k] = str(v)
        elif isinstance(v, dict):
            safe[k] = make_json_safe(v)  # recursive
        elif isinstance(v, pd.DataFrame):
            safe[k] = make_json_safe(v.to_dict())
        else:
            safe[k] = v
    return safe


def best_threshold(y_true, y_proba):
    """ Find the best threshold for binary classification based on F1 score."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    best_score= -1
    best_t = 0.5  # fallback

    ts = np.unique(y_proba)
    # if dataset is large
    if len(ts) > 500:
        ts = np.linspace(ts.min(), ts.max(), 500)

    for t in ts:
        preds = (y_proba >= t).astype(int)
        score = f1_score(y_true, preds)
        if score >= best_score:
            best_score = score
            best_t = t

    return best_t

def train_test_by_time(df, split_date='2018-12-31'):
    """ Split data into training and testing sets based on a date.
    returns X_train, X_test, y_train, y_test
    """
    X_train, X_test = df.loc[:split_date].drop('Stress', axis=1), df.loc[split_date:].drop('Stress', axis=1)
    y_train, y_test = df.loc[:split_date, 'Stress'], df.loc[split_date:, 'Stress']

    return X_train, X_test, y_train, y_test


def process_optuna_df(optuna_df):

    fold_metrics = optuna_df['user_attrs_fold_metrics']
    fold_scores = optuna_df['user_attrs_fold_scores']

    fold_metrics_df = pd.DataFrame()

    for i in fold_metrics.index:
        fold_metrics_df[int(i)] = pd.DataFrame(fold_metrics[i]).T
    
    fold_scores_dict = {}

    for i in fold_scores.index:
        fold_scores_dict[int(i)] = pd.DataFrame(fold_scores[i]['model'])

    fold_metrics_agg = fold_metrics_df.apply(['mean', 'std'], axis=1)
    fold_metrics_agg['coeff of variation'] = fold_metrics_agg['std'] / fold_metrics_agg['mean']
    fold_metrics_agg

    cols = fold_scores_dict[0].columns
    by_metrics = {}


    for col in cols:
        d = {}
        for i, fold in fold_scores_dict.items():
            d[i] = fold[col]
        by_metrics[col] = pd.DataFrame(d)
    
    return by_metrics, fold_metrics_agg, fold_metrics_df