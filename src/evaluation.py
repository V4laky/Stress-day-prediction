import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score)

import json
from datetime import datetime
from pathlib import Path

from joblib import load

def log_experiment(model_name, params, metrics, cv_scheme, features, extra_notes="", 
                   project_root=Path().resolve().parent, artifacts="", file_name="top_models.jsonl"):
    log = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "params": params,
        "cv_scheme": cv_scheme,
        "metrics": metrics,
        "features": features,
        "notes": extra_notes,
        "artifacts": artifacts
    }

    with open(project_root / f"results/{file_name}", "a") as f:
        f.write(json.dumps(log) + "\n")

def log_trials_to_csv(df, file_path=None, append = False):
    if not append:
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)
    print(f'Saved trials dataframe at {file_path}')


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

def eval_top_N(study_df, X_train, y_train, X_test, y_test, project_root=Path().resolve().parent, N=5, name='RF'):
    """
    give_metrics, fit, and dump the top N trials from an Optuna study DataFrame.
    returns results, models, metrics_df
    """
    from sklearn.ensemble import RandomForestClassifier
    from src.utils import best_threshold
    from joblib import dump

    top_trials = study_df.sort_values('value', ascending=False).head(N)
    results = {}
    models = {}
    metrics_df = pd.DataFrame()

    params = top_trials.columns.str.startswith('params_')

    for i, (_, row) in enumerate(top_trials.iterrows()):
        folds = row['user_attrs_folds']
        value = row['value']
        best_params = row[params].to_dict()
    
        rf = RandomForestClassifier( 
            n_estimators= best_params['params_n_estimators'],
            max_depth= best_params['params_max_depth'],
            min_samples_split= best_params['params_min_samples_split'],
            min_samples_leaf= best_params['params_min_samples_leaf'],
            max_features= best_params['params_max_features'],
            class_weight= best_params['params_class_weight'],
            random_state=42)
        rf.fit(X_train, y_train)
        
        pred_proba = rf.predict_proba(X_test)[:, 1]
        preds = pred_proba > best_threshold(y_test, pred_proba)

        metrics_df = give_metrics(y_test, preds, pred_proba, metrics_df, model_name=f'{name}_{i+1}')
        dump(rf, project_root / f'results/models/{name}_{i+1}.joblib')

        results[f'{name}_{i+1}'] = {'folds': folds, 'value': value, 'params': best_params}
        models[f'{name}_{i+1}'] = rf

    return results, models, metrics_df


def load_top_models(jsonl_path, models_dir):
    """
    Load top N models and their parameters/metrics from a JSONL log and joblib files.

    Parameters
    ----------
    jsonl_path : Path or str
        Path to the JSONL file logging top models (top_models.jsonl)
    models_dir : Path or str
        Directory where the saved model joblibs are stored

    Returns
    -------
    models_dict : dict
        {model_name: sklearn model object}
    metrics_dict : dict
        {model_name: {'params': ..., 'folds': ..., 'value': ...}}
    """
    jsonl_path = Path(jsonl_path)
    models_dir = Path(models_dir)
    
    models_dict = {}
    metrics_dict = {}
    
    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            name = entry.get("model")
            metrics_dict[name] = {
                "params": entry.get("params"),
                "folds": entry.get("metrics").get("folds"),
                "mean_weighted_ap": entry.get("metrics").get("mean_weighted_ap"),
                "model_scores" : entry.get("metrics").get("model_scores")
            }
            model_file = models_dir / f"{name}.joblib"
            if model_file.exists():
                models_dict[name] = load(model_file)
            else:
                print(f"[WARN] Model file {model_file} not found. Skipping.")
    
    return models_dict, metrics_dict