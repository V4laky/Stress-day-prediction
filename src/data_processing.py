import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf

from src.features import *

def has_timeframe(df: pd.DataFrame, start: str, end: str, grace_days: int = 0):
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    df_start = df.index.min()
    df_end = df.index.max()

    print(f"Existing: start: {df_start}, end: {df_end}\n Need:  start: {start_ts}, end: {end_ts}")

    return (df_start <= start_ts + pd.Timedelta(days=grace_days) and
            df_end >= end_ts - pd.Timedelta(days=grace_days))



def get_or_download_index(ticker: str, start="2010-01-01", end="2023-01-01", data_dir="data"):
    """
    Checks if CSV for index exists, otherwise downloads it using yfinance.

    Parameters
    ----------
    ticker : str
        Ticker of the index.
    data_dir : str
        Directory where data should be stored.

    Returns
    -------
    pd.DataFrame
    
    """

    path = Path(data_dir / 'raw')
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"{ticker}.csv"

    if file_path.exists():
        print(f"Found existing data for {ticker}, loading from {file_path}")
        df = pd.read_csv(file_path, parse_dates=True, index_col=0)

        # 3 grace days for non trading days
        if has_timeframe(df, start, end, 3):
            print('Existing data contains timeframe.')
            return df
    
    print(f"No local data found for {ticker}, downloading...")

    df = yf.download(ticker, start=start, end=end)
    df.columns = df.columns.get_level_values(0)
    df.to_csv(file_path)
    print(f"Saved data to {file_path}")

    return df


def process_data(price_series, stress_threshold=np.log(0.98), vix_series=None):
    """
    Process a price series into ML-ready features for stress prediction.

    Parameters:

    price_series : pd.Series
        Series of close prices (should already be cleaned).
    stress_threshold : float, optional
        Log-return threshold to define a stress event. Default is log(0.98) ≈ -0.0202

    Returns: df : pd.Dataframe
        df : Feature dataframe with stress prediction features.
    """

    # IMPORTANT: Dont use stats that look ahead (e.g. mean, std of whole timeseries)


    price_series = pd.Series(np.squeeze(price_series.values), index=price_series.index, dtype=float)

    # core features
    log_returns = np.log(price_series) - np.log(price_series.shift(1))

    df = pd.DataFrame({'log_returns': log_returns,
                      'close': price_series})

    # Merge VIX (if provided) - Important to be done first since it would delete nans from rolling features
    if vix_series is not None:
        vix_series = pd.Series(np.squeeze(vix_series.values), index=vix_series.index)
        df = df.merge(vix_series.rename('VIX'), left_index=True, right_index=True, how='left')
        df['VIX'] = df['VIX'].ffill() # if dates dont match use previous VIX
        df.dropna(inplace=True) # Drop dates that were only in VIX

    # Target variable
    df['Stress'] = df['log_returns'].shift(-1) < stress_threshold

    l = [5, 10, 20, 40]
    possible_pairs = []
    for i in l:
        for j in l:
            if i<j  and (i,j) not in possible_pairs:
                possible_pairs.append((i,j))
    possible_pairs_VIX = possible_pairs + [(i, 'VIX') for i in l]

    df = add_rolling_volatility(df, [5, 10, 20, 40])

    df = add_SMA(df, [5, 10, 20, 40])

    df = add_momentum(df, possible_pairs)

    df = add_rolling_return(df, [5, 10, 20, 40])

    df = add_scaled_lag_return(df, [(i, 20) for i in range(1,21)])

    df = add_scaled_lag_weighted_avg(df, start=1, end=20, scaling_vol=20, alpha=.8)

    df = add_skew(df, [5, 10, 20, 40])
    df = add_kurt(df, [5, 10, 20, 40])

    df = add_volatility_ratios(df, possible_pairs_VIX)

    df = add_sharpe_like(df, possible_pairs_VIX)

    df = add_z_scores(df, ['log_returns', "VIX"], look_backs=[5, 10, 20, 40])

    df = add_rolling_vol_of_vol(df, [5, 10, 20, 40]) # only 20 because this loses double the number of days
    
    # VIX stats

    df = add_rolling_VIX(df, [5,10,20,40])
    df = add_rolling_vol_of_VIX(df, [5,10,20,40])
    df = add_VIX_momentum(df, possible_pairs)
    df = add_lag_VIX(df, [i for i in range(1,21)])
    df = add_VIX_skew(df, [5,10,20,40])
    df = add_kurt(df, [5,10,20,40])
    df = add_VIX_vol_ratios(df, possible_pairs)

    # VIX/vol

    df = add_rolling_VIX_per_vol(df, [5,10,20,40], [5,10,20,40])
    df = add_rolling_vol_of_VIX_per_vol(df, [5,10,20,40], [5,10,20,40])
    df = add_VIX_per_vol_momentum(df, possible_pairs, [5,10,20,40])
    df = add_lag_VIX_per_vol(df, [i for i in range(1,21)], [5,10,20,40])
    df = add_VIX_per_vol_skew(df, [5,10,20,40], [5,10,20,40])
    df = add_VIX_per_vol_kurt(df, [5,10,20,40], [5,10,20,40])
    df = add_VIX_per_vol_vol_ratios(df, possible_pairs, [5,10,20,40])


    df.replace([np.inf, -np.inf], np.nan, inplace=True) # for division issues
    df.dropna(inplace=True)

    df.drop('close', axis=1, inplace=True)
    
    new_df = df.copy()

    return new_df

def load_data(ticker: str, start="2010-01-01", end="2023-01-01", data_dir="data/", grace_days=3):
    
    data_dir = Path(data_dir)
    file_path = Path(data_dir / f'processed/{ticker}_processed.csv')

    if file_path.exists():
        df = pd.read_csv(file_path, parse_dates=True, index_col=0)

        # band-aid solution to rolling features making df start later
        late_start = pd.to_datetime(start) + pd.Timedelta(days=130)
        # 3 grace days for non trading days
        if has_timeframe(df, late_start, end, grace_days):
            print(f"Found existing processed data for {ticker}, loading from {file_path}")
            return df

        print("Existing processed data doesnt contain timeframe")
    
    data = get_or_download_index(ticker, start, end, data_dir)
    vix = get_or_download_index("^VIX", start, end, data_dir)

    processed = process_data(data['Close'], vix_series=vix['Close'])

    output_path = data_dir / "processed"
    output_path.mkdir(parents=True, exist_ok=True)
    processed.to_csv(data_dir / f"processed/{ticker}_processed.csv")
    print(f"Saved processed data to {output_path}/{ticker}_processed.csv")

    return processed