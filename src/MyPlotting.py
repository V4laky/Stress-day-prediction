import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc

def plot_permutation_importance(models, X_test, y_test, model_names=None, n_repeats=30, scoring='roc_auc', 
                                for_pdf=False):
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
                                        scoring=scoring,
                                        n_jobs=-1)
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
    
    if for_pdf:
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, dpi=300, bbox_inches='tight')
        plt.close()
        return buf
    else:
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



def plot_curves_in_one(y_test, y_pred_proba_dict, identifier='', save_png=False, file_path=None, show=True):
    """
    Plots Precision-Recall and ROC curves for multiple models in one figure.

    Parameters:
    -----------
    y_test : array-like
        True binary labels (0 or 1).
    y_pred_proba_dict : dict
        Dictionary with model names as keys and predicted probabilities as values.
    save_png : bool
        Whether to save the figure as a PNG file.
    identifier : str
        Identifier to include in the plot titles and file name.
    file_path : str, optional
        Path to save the PNG file. If None, saves in the current directory."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, proba in y_pred_proba_dict.items():
        # PR curve
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = auc(recall, precision)
        axes[0].plot(recall, precision, label=f'{name} (AP={ap:.3f})')

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')

    axes[0].set_title(f'Precision–Recall Curve {identifier}')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].legend()

    axes[1].set_title(f'ROC Curve {identifier}')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend()

    plt.tight_layout()
    if save_png:
        if file_path is None:
            plt.savefig(f'curves_{identifier}.png', dpi=300)
        else:
            plt.savefig(file_path / f'curves_{identifier}', dpi=300)
            print(f"Saved {file_path}/curves_{identifier}")
    if show:
        plt.show()

def plot_permutation_importance_v2(eval_dict, models_dict, n_repeats=30, scoring='roc_auc', 
                                   save_png = False, file_path=None, identifier='', show=True):
    """
    Compute and plot permutation feature importance as a boxplot for multiple models.

    Parameters:
    -----------
    eval_dict : dict
        index_name: (X_test, y_test)
    models_dict : dict
        model_name: model
    n_repeats : int
        number of repeats for permutation importance
    scoring : str
        Scoring metric for importance ('roc_auc', 'f1', etc.)
    save_png : bool
        Whether to save the figure as a PNG file.
    file_path : str, optional
        Path to save the PNG file. If None, saves in the current directory.
    identifier : str
        Identifier to include in the plot titles and file name.
    """

    # Store results
    perm_importances = {}

    for model_name, model in models_dict.items():
        for index_name, (X_test, y_test) in eval_dict.items():

            result = permutation_importance(model, X_test, y_test,
                                        n_repeats=n_repeats,
                                        random_state=42,
                                        scoring=scoring,
                                        n_jobs=-1)
            perm_importances[(model_name, index_name)] = pd.DataFrame(result.importances.T, columns=X_test.columns)

    # Prepare long-format DataFrame for boxplot
    df_plot = pd.concat([
            df.melt(var_name='Feature', value_name='Importance').assign(Model=model_name, Index=index_name)
            for (model_name, index_name), df in perm_importances.items()], 
            ignore_index=True)

    # Combine Model and Index into a single hue label for plotting
    df_plot['Label'] = df_plot['Index'] + " - " + df_plot['Model']

    # Plot
    plt.figure(figsize=(14,8))
    sns.boxplot(x='Importance', y='Feature', hue='Label', data=df_plot, orient='h')
    plt.title(f'Permutation Feature Importance ({scoring})')
    plt.xlabel('Importance (drop in score)')
    plt.ylabel('Feature')
    plt.legend(loc='lower right', fontsize='small', title='Index - Model')
    plt.tight_layout()

    if save_png:
        if file_path is None:
            plt.savefig(f'perm_imp_{identifier}.png', dpi=300)
        else:
            plt.savefig(file_path / f'perm_imp_{identifier}', dpi=300)
            print(f"Saved {file_path}/perm_imp_{identifier}")
    
    if show:
        plt.show()


def plot_abs_importances(imp_df, save_png=False, file_path=None, identifier="", show=True):
    
    cols = imp_df.columns

    imp_df = imp_df.abs()
    imp_df = imp_df.sort_values(cols[0], axis=0, ascending=False)

    ax = imp_df.plot(kind="barh", figsize=(8, 8), width=0.8, colormap='viridis')
    ax.set_xlabel("Abs Importance")
    ax.set_ylabel("Parameter")
    plt.title(f"Parameter Importance {identifier}")
    plt.tight_layout()
    
    if save_png:
        if file_path is None:
            plt.savefig(f'abs_imp_{identifier}.png', dpi=300)
        else:
            plt.savefig(file_path / f'abs_imp_{identifier}', dpi=300)    
            print(f"Saved {file_path}/abs_imp_{identifier}")
    
    if show:
        plt.show()




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