import numpy as np


# Model Initialization Functions

def get_shallow_rf(random_state=42):
    """Return a shallow Random Forest"""
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=1000,
        max_depth=3,
        class_weight='balanced',
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state
    )

def get_deep_rf(random_state=42):
    """Return a deeper, regularized Random Forest"""
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=1000,
        max_depth=5,
        class_weight='balanced',
        min_samples_leaf=10,
        min_samples_split=10,
        max_features='log2',
        random_state=random_state
    )

def get_xgb(scale_pos_weight=40, random_state=42):
    """Return an XGBoost classifier"""
    from xgboost import XGBClassifier
    
    return XGBClassifier(
        n_estimators=1000,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=40,
        eval_metric='aucpr',
        learning_rate=0.2,
        subsample = .8,
        max_depth=5,
        random_state=random_state
    )


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