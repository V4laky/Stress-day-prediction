from sklearn.metrics import f1_score
import numpy as np

def best_threshold(y_true, y_proba):
    """ Find the best threshold for binary classification based on F1 score."""
    preds = y_proba > .5
    best_tresh = .5
    best_score = f1_score(y_true, preds)
    for i in np.arange(0.05, 1, 0.005):
        preds = y_proba > i
        score = f1_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_tresh = i
    print(best_tresh)
    return best_tresh

def train_test_by_time(df, split_date='2018-12-31'):
    """ Split data into training and testing sets based on a date.
    returns X_train, X_test, y_train, y_test
    """
    X_train, X_test = df.loc[:split_date].drop('Stress', axis=1), df.loc[split_date:].drop('Stress', axis=1)
    y_train, y_test = df.loc[:split_date, 'Stress'], df.loc[split_date:, 'Stress']

    return X_train, X_test, y_train, y_test