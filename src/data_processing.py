import numpy as np
import pandas as pd
from pathlib import Path

def get_or_download_index(symbol: str, download_fn, data_dir="data/raw"):
    """
    Checks if CSV for index exists, otherwise downloads it.

    Parameters
    ----------
    symbol : str
        Ticker or name of the index (e.g. "SP500").
    download_fn : callable
        Function that downloads data and returns a DataFrame.
    data_dir : str
        Directory where data should be stored.

    Returns
    -------
    pd.DataFrame
    
    """

    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"{symbol}.csv"

    if file_path.exists():
        print(f"Found existing data for {symbol}, loading from {file_path}")
        df = pd.read_csv(file_path, parse_dates=True, index_col=0)

    else:
        print(f"No local data found for {symbol}, downloading...")
        df = download_fn(symbol)
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

    # TODO: check and talk more about stationarity
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

    # Rolling volatility
    df['20_day_rolling_volatility'] = df['log_returns'].rolling(20).std()
    df['5_day_rolling_volatility'] = df['log_returns'].rolling(5).std()

    # SMA and momentum
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['SMA_10'] = df['close'].rolling(10).mean()
    df['SMA_5'] = df['close'].rolling(5).mean()
    df['SMA_3'] = df['close'].rolling(3).mean()

    df['Momentum5_20'] = (df['SMA_5'] / df['SMA_20']) -1 # for mean 0
    df['Momentum10_20'] = (df['SMA_10'] / df['SMA_20']) -1
    df['Momentum5_10'] = (df['SMA_5'] / df['SMA_10']) -1
    df['Momentum3_20'] = (df['SMA_3'] / df['SMA_20']) -1
    df['Momentum3_10'] = (df['SMA_3'] / df['SMA_10']) -1
    df['Momentum3_5'] = (df['SMA_3'] / df['SMA_5']) -1


    # Rolling return
    df['10_day_rolling_return'] = df['log_returns'].rolling(10).mean()
    df['5_day_rolling_return'] = df['log_returns'].rolling(5).mean()

    df['Scaled_Lag_1d'] = df['log_returns'].shift(1) / df['20_day_rolling_volatility']
    df['Scaled_Lag_3d'] = df['log_returns'].shift(3) / df['20_day_rolling_volatility']
    df['Scaled_Lag_5d'] = df['log_returns'].shift(5) / df['20_day_rolling_volatility']

    df['Scaled_weighted_avg'] = (df['Scaled_Lag_1d']*.6 + df['Scaled_Lag_3d']*.3 + df['Scaled_Lag_5d']*.1)
    # Skew and kurtosis

    df['Skew_20'] = df['log_returns'].rolling(20).skew()
    df['Kurt_20'] = df['log_returns'].rolling(20).kurt()

    # volatility ratios
    df['Vol_momentum'] = (df['5_day_rolling_volatility'] / df['20_day_rolling_volatility']) -1
    df['Vol20/VIX'] = df['20_day_rolling_volatility'] / df['VIX']
    df['Vol5/VIX'] = df['5_day_rolling_volatility'] / df['VIX']

    # Sharpe-like ratios
    df['Sharpe-like_10'] = df['10_day_rolling_return'] / df['20_day_rolling_volatility']
    df['Sharpe-like_5'] = df['5_day_rolling_return'] / df['20_day_rolling_volatility']

    # return and VIX z-score
    df['log_returns_zscore'] = (df['log_returns'] - df['10_day_rolling_return']) / df['20_day_rolling_volatility']
    df['VIX_zscore'] = (df['VIX'] - df['VIX'].rolling(10).mean()) / df['VIX'].rolling(20).std()

    df.replace([np.inf, -np.inf], np.nan, inplace=True) # for division issues
    df.dropna(inplace=True)

    df.drop(['SMA_20', 'SMA_5'], axis=1, inplace=True)
    df.drop('close', axis=1, inplace=True)

    return df