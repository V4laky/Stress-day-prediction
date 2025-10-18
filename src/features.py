import pandas as pd

def add_rolling_volatility(df : pd.DataFrame, n_of_days : list):

    for n in n_of_days:
        df[f'{n}_day_rolling_volatility'] = df['log_returns'].rolling(n).std()

def add_SMA(df: pd.DataFrame, n_of_days : list):

    for n in n_of_days:
        df[f'SMA_{n}'] = df['close'].rolling(n).mean()

def add_momentum(df: pd.DataFrame, n_of_days : list[tuple]):

    for (n, m) in n_of_days:
        df[f'Momentumn{n}_{m}'] = (df[f'SMA_{n}'] / df[f'SMA_{m}']) -1

def add_rolling_return(df: pd.DataFrame, n_of_days : list):

    for n in n_of_days:
        df[f'{n}_day_rolling_return'] = df['log_returns'].rolling(n).mean()

def add_lag_return(df: pd.DataFrame, n_of_days : list):

    for n in n_of_days:
        df[f'lag_return_{n}'] = df['log_returns'].shift(n)

def add_scaled_lag_return(df: pd.DataFrame, n_of_days : list[tuple]):
    """
    Doesnt need lag_returns to be calculated!
    """

    for (n, m) in n_of_days:
        df[f'lag_return_{n}/vol{m}'] = df['log_returns'].shift(n) / df[f'{m}_day_rolling_volatility']

def add_scaled_lag_weighted_avg(df: pd.DataFrame, start:int, end:int, scaling_vol:int, alpha:float=.8):
    """
    exponentially scaled weighted average of lagged returns scaled by volatility
    (uses scaled lag returns)
    """

    avg_return, sum_of_weights = 0, 0
    for i, lag in enumerate(range(start, end+1), 1):
        avg_return += df[f'lag_return_{lag}/vol{scaling_vol}']*(alpha**i)
        sum_of_weights += (alpha**i)
    
    df[f'EWLR_{alpha}_{start}-{end}/{scaling_vol}'] = avg_return / sum_of_weights

def add_skew(df: pd.DataFrame, n_of_days:list[int]):

    for n in n_of_days:
        df[f'Skew_{n}'] = df['log_returns'].rolling(n).skew()

def add_kurt(df: pd.DataFrame, n_of_days:list[int]):

    for n in n_of_days:
        df[f'Kurt_{n}'] = df['log_returns'].rolling(n).kurt()
    

def add_volatility_ratios(df: pd.DataFrame, ratios:list[tuple]):
    """
    ratios can be (n, m) as volatility momentum (Vol_momentum_n_m)
    or (n, 'VIX') as Voln/VIX
    """
    for (n,m) in ratios:
        if m == 'VIX':
            df[f'VOL{n}/VIX'] = df[f'{n}_day_rolling_volatility'] / df['VIX']
        else:
            df[f'Vol_momentum_{n}_{m}'] = (df['5_day_rolling_volatility'] / df['20_day_rolling_volatility']) -1            

def add_sharpe_like(df: pd.DataFrame, ratios:list[tuple]):

    for (n,m) in ratios:
        df[f'Sharpe-like_{n}/{m}'] = df[f'{n}_day_rolling_return'] / df[f'{m}_day_rolling_volatility']


def add_z_scores(df: pd.DataFrame, feature_names:list[str], look_back:int):
    
    for feature in feature_names:
        df[f'{feature}_z_{look_back}'] = (df[feature] - df[feature].rolling(look_back).mean()) / df[feature].rolling(look_back).std()
