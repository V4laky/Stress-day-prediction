import argparse
from pathlib import Path
import sys

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from src.evaluation import load_and_eval_models
from src.data_processing import load_data
from src.MyPlotting import plot_abs_importances, plot_curves_in_one, plot_permutation_importance_v2
from src.reporting import make_pdf_report
import yaml

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ticker = config["data"]["ticker"]
    start = config["data"]["start"]
    end = config["data"]["end"]
    split_date = config["data"]["split_date"]
    features = config["features"]

    model_type = config["training"]["model"]
    n_optuna_trials = config["training"]["n_optuna_trials"]
    top_n_models = config["training"]["top_n_models"]
    # this might become a settable parameter
    metric_to_optimize = "average_precision"

    model_names = config["evaluation"]["model_names"]
    eval_tickers = config['evaluation']["eval_tickers"]
    pdf_report_name = config['evaluation']['pdf_report_name']


    project_root = Path(__file__).resolve().parent.parent # assumes file is in a subdirectory of the project root
    print(f'Project root is: {project_root}')

    train_sets = {}
    for ticker in eval_tickers:
        data = load_data(ticker, start, end, project_root / 'data')

        X_train, X_test = data.loc[start:split_date].drop('Stress', axis=1), data.loc[split_date:end].drop('Stress', axis=1)
        y_train, y_test = data.loc[start:split_date, 'Stress'], data.loc[split_date:end, 'Stress']

        missing = set(features) - set(data.columns)
        if missing:
            raise ValueError(f"Features not found in data: {missing}")
        X_train, X_test = X_train[features], X_test[features]

        train_sets[ticker] = (X_test, y_test)
    

    eval_dict = load_and_eval_models(model_names, train_sets, project_root)
    proba_series = eval_dict['proba_series']
    models = eval_dict['models']

    fold_scores_dict = eval_dict["fold_scores_dict"]
    objective_metric_dict = {}
    for name in models:
        objective_metric_dict[name] = fold_scores_dict[name]['model'][f'val_{metric_to_optimize}_fold_scores']
    
    import pandas as pd
    objective_metric_df = pd.DataFrame(objective_metric_dict)
    

    plot_abs_importances(eval_dict['feat_imp_df'], True, project_root / "results/figures", show=False)

    for market, (X_test, y_test) in train_sets.items():
        plot_curves_in_one(y_test, proba_series[market].to_dict(), identifier=market, 
                           save_png=True, file_path=project_root / "results/figures", show=False)
    
    has_perm_imp = False # debug
    if not has_perm_imp:
        for model in models:
            plot_permutation_importance_v2(train_sets, {model: models[model]}, scoring='average_precision',
                                   save_png=True, file_path=project_root / "results/figures", identifier=model, show=False)

    make_pdf_report(eval_dict["metrics_df"], eval_dict['params_df'], objective_metric_df, 
                    train_sets.keys(), project_root / "results", filename=pdf_report_name)



if __name__ == "__main__":
    main()
