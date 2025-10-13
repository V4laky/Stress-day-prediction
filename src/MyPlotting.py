import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc

def plot_permutation_importance(models, X_test, y_test, model_names=None, n_repeats=30, scoring='roc_auc'):
    """
    Compute and plot permutation feature importance as a boxplot for multiple models.

    Parameters:
    -----------
    models : list
        List of trained models (e.g., [rf_shallow, rf_deep, xgb])
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    model_names : list, optional
        List of model names for labeling
    n_repeats : int
        Number of permutation repeats
    scoring : str
        Scoring metric for importance ('roc_auc', 'f1', etc.)
    """
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(len(models))]

    # Store results
    perm_importances = {}

    for model, name in zip(models, model_names):
        result = permutation_importance(model, X_test, y_test,
                                        n_repeats=n_repeats,
                                        random_state=42,
                                        scoring=scoring)
        perm_importances[name] = pd.DataFrame(result.importances.T, columns=X_test.columns)

    # Prepare long-format DataFrame for boxplot
    df_plot = pd.concat([df.melt(var_name='Feature', value_name='Importance').assign(Model=name)
                         for name, df in perm_importances.items()], ignore_index=True)

    # Plot
    plt.figure(figsize=(12,6))
    sns.boxplot(x='Importance', y='Feature', hue='Model', data=df_plot, orient='h')
    plt.title(f'Permutation Feature Importance ({scoring})')
    plt.xlabel('Importance (drop in score)')
    plt.ylabel('Feature')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def plot_pr_curve(y_true, X_test, models, model_names=["Model"], curve_type='PR'):
    """
    Plots Precision–Recall curve and shows Average Precision (AP).

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    X_test : DataFrame
        Test features to predict probabilities.
    models : list of str
        List of models.
    model_names : list of str
        Label for the plot legend.
    curve_type: str
        PR or ROC curve.
    """
    row_number = ((len(models) -1) // 3) + 1

    fig, ax = plt.subplots(row_number, 3, figsize=(14, 3*row_number))

    for i, (model, model_name) in enumerate(zip(models, model_names)):
        r = i // 3  # row
        c = i % 3   # column

        if curve_type == 'PR':
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            Y, X, _ = precision_recall_curve(y_true, y_pred_proba)
            auc_score = average_precision_score(y_true, y_pred_proba)

        if curve_type == 'ROC':
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            X, Y, _ = roc_curve(y_true, y_pred_proba)
            auc_score = auc(X, Y)

        if row_number == 1:
            ax[c].plot(X, Y)
            ax[c].set_title(f'{curve_type} Curve: {model_name} ({curve_type} = {auc_score:.3f})')
        else:
            ax[r, c].plot(X, Y)
            ax[r, c].set_title(f'{curve_type} Curve: {model_name} ({curve_type} = {auc_score:.3f})')
            ax[r, c].legend(loc='best')
            ax[r, c].grid(alpha=0.3)

def auto_plot_features(Feature_df):

    sns.set(style="whitegrid")
    feature_names = Feature_df.columns
    row_number = ((len(feature_names) -1) // 3) + 1

    fig, ax = plt.subplots(row_number, 3, figsize=(14, 3*row_number))
    fig.suptitle('Feature Diagnostics', fontsize=16, fontweight='bold')


    # Loop through subplots
    for i, col in enumerate(feature_names):
        r = i // 3  # row
        c = i % 3   # column
        ax[r, c].plot(Feature_df[col], color=sns.color_palette("deep")[i%10])
        ax[r, c].set_title(col, fontsize=11, fontweight='semibold')
        ax[r, c].tick_params(axis='x', rotation=20)
        ax[r, c].spines['top'].set_visible(False)
        ax[r, c].spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # adjust for suptitle
    plt.show()