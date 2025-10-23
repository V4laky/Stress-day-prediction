from pathlib import Path
import argparse
import yaml
import json
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).resolve().parent.parent # assumes file is in a subdirectory of the project root
sys.path.append(str(project_root))

from src.data_processing import load_data
from src.MyPlotting import line_plot_metric
from src.utils import process_optuna_df
from src.reporting import make_features_pdf

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
        config_path = project_root / config_path

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    ticker = config["data"]["ticker"]
    start = config["data"]["start"]
    end = config["data"]["end"]
    split_date = config["data"]["split_date"]
    features = config["features"]

    pdf_report_name = config['evaluation']['pdf_report_name']
    # this relying on _report might be an issue later!!!
    if "_report" in pdf_report_name:
        feature_report_name = pdf_report_name.replace("_report", '_features')
    else:
        feature_report_name = pdf_report_name[:-4] + '_features.pdf'
    
    # to trim _val
    json_name = pdf_report_name[:-4] + '.json'

    print(f'Project root is: {project_root}')

    data = load_data(ticker, start, end, project_root / 'data')

    X_train = data.loc[start:split_date].drop('Stress', axis=1)

    missing = set(features) - set(data.columns)
    if missing:
        raise ValueError(f"Features not found in data: {missing}")
    X_train = X_train[features]

    with open(project_root / f'results/{json_name}') as f:
        df = json.load(f)

    optuna_df = pd.DataFrame(df)

    by_metrics, fold_metrics_agg, fold_metrics_df = process_optuna_df(optuna_df)
    
    # to strip val_
    fold_metrics_agg.index = fold_metrics_agg.index.str[4:]
    fold_metrics_df.index = fold_metrics_df.index.str[4:]

    figures_path = project_root / 'results/figures'
    figures_path.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(3, 2, figsize=(12, 20))

    for i, (metric, df) in enumerate(by_metrics.items()):
        line_plot_metric(df, metric, ax[(i//2, i%2)])
    
    plt.tight_layout()
    plt.savefig(figures_path / f'feat_lines.png', dpi=300)  
    print(f"Saved {figures_path} / feat_lines.png")

    corr = X_train.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0) # center 0 helps with not making it seem better than it is
    plt.tight_layout()
    plt.title("Feature Correlation Matrix")

    plt.savefig(figures_path / 'heatmap.png', dpi=300)    
    print(f"Saved {figures_path} / heatmap.png")

    make_features_pdf(fold_metrics_agg, fold_metrics_df, figures_path.parent, feature_report_name)

if __name__ == "__main__":
    main()