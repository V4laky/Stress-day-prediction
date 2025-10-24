import pandas as pd
import numpy as np

from src.utils import best_threshold
from sklearn.metrics import get_scorer

SCORING = [
    'accuracy',
    'precision', 
    'recall', 
    'f1',
    'roc_auc',
    'average_precision'
    ]



def cv_with_es_and_metrics(model, X, y, cv, scoring = SCORING, early_stopping=None, 
                            eval_metric='average_precision', optimize_threshold=False):

    metrics = {i:[] for i in scoring}
    scorers = {i:get_scorer(i) for i in scoring}

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if early_stopping is not None:
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)], 
                    #eval_metric=eval_metric,
                    #early_stopping_rounds=early_stopping 
                    verbose=False
                )
            except TypeError as e:
                print(f"Couldn't use early stopping for {type(model).__name__}: {e}")
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
            
        # predictions
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_val)[:, 1]
        else:
            proba = model.predict(X_val)

        if optimize_threshold:
            pred = proba >= best_threshold(y_val, proba) # based on f1
            pred = pred.astype(int)
        else:
            pred = (proba >= .5).astype(int)

        # Score each metric appropriately
        for metric_name, scorer in scorers.items():
            if metric_name in ["roc_auc", "average_precision"]:
                # pass probabilities
                metrics[metric_name].append(scorer._score_func(y_val, proba))
            else:
                # pass binary labels
                metrics[metric_name].append(scorer._score_func(y_val, pred))

    return metrics


def timeseries_multiple_cv_scoring(X_train, y_train, models_dict, tscv, early_stopping=None,
                                   scoring = SCORING, alpha=.8):
    """
    Perform time series cross-validation for multiple models and compute 
    weighted mean and standard deviation of multiple metrics.

    Parameters
    ----------
    X_train : array-like or DataFrame
        Feature matrix used for training.
    
    y_train : array-like
        Target vector corresponding to X_train.
    
    models_dict : dict
        Dictionary of models to evaluate, with structure:
        {
            "ModelName1": estimator1,
            "ModelName2": estimator2,
            ...
        }
    
    tscv : cross-validation generator
        A time series cross-validator (e.g. TimeSeriesSplit).
    
    early_stopping: int, deafault=None

    scoring : list or dict of str, default=SCORING
        Scoring metrics to evaluate. Should match valid metrics for 
        `sklearn.model_selection.cross_validate`.
    
    alpha : float, default=0.8
        Exponential decay factor for weighting CV folds. 
        More recent folds are given higher weight.

    Returns
    -------
    val_df : pandas.DataFrame
        A DataFrame containing the weighted mean and standard deviation 
        of each metric for every model. Columns are named as:
            - `val_<metric>_mean`
            - `val_<metric>_std`
        Rows correspond to model names.

    fold_scores_dict : dict
        Dictionary containing the raw fold scores for each metric and model.
        Structure:
        {
            "ModelName1": {
                "val_<metric>_fold_scores": array([...]),
                ...
            },
            ...
        }

    Notes
    -----
    - The function uses exponential weighting of folds so that more 
      recent folds contribute more strongly to the mean and standard deviation.
    - Useful for time series problems where performance on later periods
      is more relevant than earlier periods.
    """

    means_dict={}
    stds_dict={}
    fold_scores_dict={}

    n_splits = len(list(tscv.split(X_train)))
    weights = np.array([alpha**(n_splits - i) for i in range(n_splits)])
    weights = weights/ weights.sum()

    for name, model in models_dict.items():
        means = {}
        stds = {}
        fold_scores={}
        
        scores = cv_with_es_and_metrics(model, X_train, y_train, cv=tscv, scoring=scoring, 
                                        optimize_threshold=True, early_stopping=early_stopping)
        #scores = cross_validate(model, X_train, y_train, cv=tscv, scoring=scoring, n_jobs=-1, verbose=0)
        
        for metric in scoring:
            fold_vals = np.array(scores[metric])
            fold_scores[f"val_{metric}_fold_scores"] = fold_vals

            weighted_score = fold_vals @ weights
            weighted_mean = weighted_score

            means[f"val_{metric}_mean"] = weighted_mean
            
            res = fold_vals - weighted_mean
            weighted_var = (weights * res**2).sum()
            weighted_std = np.sqrt(weighted_var)

            stds[f"val_{metric}_std"] = weighted_std
        
        means_dict[name] = means
        stds_dict[name] = stds
        fold_scores_dict[name] = fold_scores

    val_df = pd.concat([pd.DataFrame(means_dict).T, pd.DataFrame(stds_dict).T], axis=1)
    return val_df, fold_scores_dict
