import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score)

def give_metrics(y_true, y_pred, y_pred_proba=None, df=None, model_name ='Model'):
    """
    Calculate and return a DataFrame of classification metrics.
    If a DataFrame is provided, append the new metrics as a new row.
    """

    metrics = pd.DataFrame({
        'Accuracy ': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0)
    }, index=[model_name])

    if y_pred_proba is not None:
        metrics['ROC AUC'] = roc_auc_score(y_true, y_pred_proba)
        metrics['Average Precision'] = average_precision_score(y_true, y_pred_proba)
    
    if df is not None:
        metrics = pd.concat([df, metrics], axis=0)
    
    return metrics