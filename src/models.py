import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from pathlib import Path
from joblib import dump

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



SCORING = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'average_precision': 'average_precision'
}

def timeseries_multiple_cv_scoring(X_train, y_train, models_dict, tscv, scoring = SCORING, alpha=.8):
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
        
        scores = cross_validate(model, X_train, y_train, cv=tscv, scoring=scoring, n_jobs=-1, verbose=0)
        
        for metric in scoring:
            fold_vals = np.array(scores['test_' + metric])
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



def build_rf(trial=None, params=None, **kwargs):
    if params is not None:
        params['class_weight'] = 'balanced'
        params['random_state'] = 42
        return RandomForestClassifier(**params)
    elif trial is not None:
        return RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 200, 1500),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 32),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 16),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError("Either trial or params must be provided")

def build_xgb(trial=None, params=None, scale_pos_weight=None):
    if params is not None:
        if scale_pos_weight is not None:
            params['scale_pos_weight'] = scale_pos_weight
        params['random_state'] = 42
        return XGBClassifier(**params)
    if trial is not None:
        return XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 200, 1500),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError("Either trial or params must be provided")
    
    
def build_lgbm(trial=None, params=None, scale_pos_weight=None):
    if params is not None:
        if scale_pos_weight is not None:
            params['scale_pos_weight'] = scale_pos_weight
        params['random_state'] = 42
        return LGBMClassifier(**params)
    elif trial is not None:
        return LGBMClassifier(
            n_estimators= trial.suggest_int('n_estimators', 200, 1500),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            num_leaves=trial.suggest_int('num_leaves', 31, 256),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )        
    else:
        raise ValueError("Either trial or params must be provided")

MODEL_BUILDERS = {
    'rf': build_rf,
    'xgb': build_xgb,
    'lgbm': build_lgbm
}

def objective(trial, model_type, X_train, y_train, alpha=0.8, scoring='average_precision', scale_pos_weight=None, n_splits=5):
    """
    Objective function for hyperparameter optimization (e.g., Optuna),
    using time series cross-validation with exponential weighting.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Current Optuna trial object.
    
    model_type : str
        Key to select which model builder to use from MODEL_BUILDERS.
    
    X_train : array-like or DataFrame
        Feature matrix.
    
    y_train : array-like
        Target vector.
    
    alpha : float, default=0.8
        Exponential decay factor for weighting CV folds.
    
    scoring : str, default='average_precision'
        Metric name to optimize (must match one of the metrics in SCORING).
    
    scale_pos_weight : float or None, optional
        Optional parameter passed to the model builder (useful for imbalanced data).
    
    n_splits : int, default=5
        number of splits made for cv.

    Returns
    -------
    float
        Weighted mean score of the selected metric across CV folds.
    """
    model = MODEL_BUILDERS[model_type](trial, scale_pos_weight=scale_pos_weight)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    val_df, fold_scores = timeseries_multiple_cv_scoring(
        X_train, y_train, {'model': model}, tscv, alpha=alpha
    )
    trial.set_user_attr('fold_scores', fold_scores)
    trial.set_user_attr('fold_metrics', val_df)
    
    score_col = f'val_{scoring}_mean'
    if score_col not in val_df.columns:
        raise ValueError(f"Scoring metric '{scoring}' not found in CV results. Available: {val_df.columns.tolist()}")

    return val_df.loc['model', score_col]


def train_top_models(X_train, y_train, study, model_type, top_n=5, scale_pos_weight=None, save_dir="models/"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:top_n]
    
    trained_models = []
    for rank, trial in enumerate(top_trials, 1):
        save_path = save_dir / f"{model_type}_trial_{rank}.joblib"
        model = MODEL_BUILDERS[model_type](params=trial.params, scale_pos_weight=scale_pos_weight)

        # attach user_attrs to model
        if hasattr(trial, "user_attrs"):
            model.user_attrs = trial.user_attrs

        model.fit(X_train, y_train)

        trained_models.append(model)
        dump(model, save_path)
        print(f"Trained and saved {model_type} model from trial {rank} -> {save_path}")
            
    return trained_models
    

# Ensemble / Aggregation

def ensemble_predict(models, X_test, weights=None, threshold=0.5):
    """
    Weighted average ensemble predictions
    models : list of trained models
    weights: list of floats (same length as models)
    """

    weights = weights or [1/len(models)]*len(models)
    proba_sum = np.zeros(X_test.shape[0])

    for model, w in zip(models, weights):
        proba_sum += model.predict_proba(X_test)[:, 1] * w

    y_pred = (proba_sum >= threshold).astype(int)
    return y_pred, proba_sum