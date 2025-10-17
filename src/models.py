import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
from joblib import dump

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def cross_val(model, tscv, X_train, y_train, alpha=0.8, scoring='average precision'):
    """
    for now only 'average precision' is implemented.
    """
    
    scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_t, y_t)
        y_pred = model.predict_proba(X_v)[:, 1]
        if scoring == 'average precision':
            ap = average_precision_score(y_v, y_pred)
        scores.append(ap)
    
    if alpha is None:
        return np.mean(scores)
    
    weights = np.array([alpha**(len(scores) - i) for i in range(len(scores))])
    weighted_scores = np.array(scores) * weights

    return np.sum(weighted_scores) / np.sum(weights), scores

def build_rf(trial=None, params=None):
    if params is not None:
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

def build_xgb(trial=None, params=None):
    if params is not None:
        return XGBClassifier(**params)
    if trial is not None:
        return XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 200, 1500),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            scale_pos_weight=trial.suggest_int("scale_pos_weight", 20, 100),
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError("Either trial or params must be provided")
    
    
def build_lgbm(trial=None, params=None):
    if params is not None:
        return LGBMClassifier(**params)
    elif trial is not None:
        return LGBMClassifier(
            n_estimators= trial.suggest_int('n_estimators', 200, 1500),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            num_leaves=trial.suggest_int('num_leaves', 31, 256),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            class_weight='balanced',
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

def objective(trial, model_type, X_train, y_train, alpha=0.8, scoring='average precision'):
    model = MODEL_BUILDERS[model_type](trial)
    tscv = TimeSeriesSplit(n_splits=5)
    score, folds = cross_val(model, tscv, X_train, y_train, alpha, scoring)
    trial.set_user_attr('folds', folds)
    return score


def train_top_models(X_train, y_train, study, model_type, top_n=5, save_dir="models/"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:top_n]
    
    trained_models = []
    for rank, trial in enumerate(top_trials, 1):
        save_path = save_dir / f"{model_type}_trial_{rank}.joblib"
        model = MODEL_BUILDERS[model_type](params=trial.params)

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