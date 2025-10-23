import argparse
from pathlib import Path
import sys

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from src.data_processing import load_data
from src.models import objective, train_top_models
from src.evaluation import log_trials_to_json
import optuna
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
    early_stopping = config["training"].get("early_stopping", None)

    pdf_report_name = config['evaluation']['pdf_report_name'] # use it for json name
    json_name = pdf_report_name[:-4] + '.json'
    

    project_root = Path(__file__).resolve().parent.parent # assumes file is in a subdirectory of the project root
    print(f'Project root is: {project_root}')

    data = load_data(ticker, start, end, project_root / 'data')

    X_train, X_test = data.loc[start:split_date].drop('Stress', axis=1), data.loc[split_date:end].drop('Stress', axis=1)
    y_train, y_test = data.loc[start:split_date, 'Stress'], data.loc[split_date:end, 'Stress']

    missing = set(features) - set(data.columns)
    if missing:
        raise ValueError(f"Features not found in data: {missing}")
    X_train, X_test = X_train[features], X_test[features]

    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos if pos > 0 else 1

    def objective_wrapper(trial):
        return objective(trial, model_type=model_type, X_train=X_train, early_stopping=early_stopping,
                         y_train=y_train, scale_pos_weight=scale_pos_weight)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=42))
    study.optimize(objective_wrapper, n_trials=n_optuna_trials, n_jobs=-1)

    study_df = study.trials_dataframe().sort_values('value', ascending=False)
    
    results_dir = project_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    log_trials_to_json(study_df, results_dir / json_name)

    train_top_models(X_train, y_train, X_test, y_test, study, model_type, top_n_models, 
                     scale_pos_weight=scale_pos_weight, early_stopping=early_stopping, 
                     save_dir=results_dir/'models')

if __name__ == "__main__":
    main()
