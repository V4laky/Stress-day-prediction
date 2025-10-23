import pandas as pd

# using .copy() at the end of functions to defragment

def add_rolling_volatility(df : pd.DataFrame, n_of_days : list):

    for n in n_of_days:
        df[f'{n}_day_rolling_volatility'] = df['log_returns'].rolling(n).std()

    return df.copy()

def add_SMA(df: pd.DataFrame, n_of_days : list):

    for n in n_of_days:
        df[f'SMA_{n}'] = df['close'].rolling(n).mean()

    return df.copy()

def add_momentum(df: pd.DataFrame, n_of_days : list[tuple]):

    for (n, m) in n_of_days:
        df[f'Momentumn{n}_{m}'] = (df[f'SMA_{n}'] / df[f'SMA_{m}']) -1

    return df.copy()

def add_rolling_return(df: pd.DataFrame, n_of_days : list):

    for n in n_of_days:
        df[f'{n}_day_rolling_return'] = df['log_returns'].rolling(n).mean()

    return df.copy()

def add_lag_return(df: pd.DataFrame, n_of_days : list):

    for n in n_of_days:
        df[f'lag_return_{n}'] = df['log_returns'].shift(n)

    return df.copy()

def add_scaled_lag_return(df: pd.DataFrame, n_of_days : list[tuple]):
    """
    Doesnt need lag_returns to be calculated!
    """

    for (n, m) in n_of_days:
        df[f'lag_return_{n}/vol{m}'] = df['log_returns'].shift(n) / df[f'{m}_day_rolling_volatility']

    return df.copy()

def add_scaled_lag_weighted_avg(df: pd.DataFrame, start:int, end:int, scaling_vol:int, alpha:float=.8):
    """
    exponentially scaled weighted average of lagged returns scaled by volatility
    (uses scaled lag returns)
    """

    avg_return, sum_of_weights = 0, 0
    for i, lag in enumerate(range(start, end+1), 1):
        avg_return += df[f'lag_return_{lag}/vol{scaling_vol}']*(alpha**i)
        sum_of_weights += (alpha**i)

    return df.copy()
    
    df[f'EWLR_{alpha}_{start}-{end}/{scaling_vol}'] = avg_return / sum_of_weights

def add_skew(df: pd.DataFrame, n_of_days:list[int]):

    for n in n_of_days:
        df[f'Skew_{n}'] = df['log_returns'].rolling(n).skew()

    return df.copy()

def add_kurt(df: pd.DataFrame, n_of_days:list[int]):

    for n in n_of_days:
        df[f'Kurt_{n}'] = df['log_returns'].rolling(n).kurt()

    return df.copy()
    

def add_volatility_ratios(df: pd.DataFrame, ratios:list[tuple]):
    """
    ratios can be (n, m) as volatility momentum (Vol_momentum_n_m)
    or (n, 'VIX') as VIX/vol !!! these are important for VIX/vol stats
    """
    for (n,m) in ratios:
        if m == 'VIX':
            df[f'VIX/vol{n}'] = df['VIX'] / df[f'{n}_day_rolling_volatility']
        else:
            df[f'Vol_momentum_{n}_{m}'] = (df[f'{n}_day_rolling_volatility'] / df[f'{m}_day_rolling_volatility']) -1            

    return df.copy()

def add_sharpe_like(df: pd.DataFrame, ratios:list[tuple]):
    """
    second one in ratio can be 'VIX'
    """
    for (n,m) in ratios:
        if m == "VIX":
            df[f'Sharpe-like_{n}/{m}'] = df[f'{n}_day_rolling_return'] / df['VIX']
        else:    
            df[f'Sharpe-like_{n}/{m}'] = df[f'{n}_day_rolling_return'] / df[f'{m}_day_rolling_volatility']

    return df.copy()


def add_z_scores(df: pd.DataFrame, feature_names:list[str], look_backs:list[int]):
    
    for feature in feature_names:
        for look_back in look_backs:
            df[f'{feature}_z_{look_back}'] = (df[feature] - df[feature].rolling(look_back).mean()) / df[feature].rolling(look_back).std()


    return df.copy()


def add_rolling_vol_of_vol(df : pd.DataFrame, n_of_days : list[int]):
    """
    this is going to do it with all possible pairs (not caring about n < m)
    """
    for n in n_of_days:
        for m in n_of_days:
            df[f'{n}_day_rolling_vol_of_vol_{m}'] = df[f'{m}_day_rolling_volatility'].rolling(n).std()

    return df.copy()


# VIX stats

def add_rolling_VIX(df : pd.DataFrame, n_of_days : list):

    for n in n_of_days:
        df[f'{n}_day_rolling_VIX'] = df['VIX'].rolling(n).mean()

    return df.copy()

def add_rolling_vol_of_VIX(df: pd.DataFrame, n_of_days : list):

    for n in n_of_days:
        df[f'{n}_day_rolling_vol_of_VIX'] = df['VIX'].rolling(n).std()

    return df.copy()

def add_VIX_momentum(df: pd.DataFrame, n_of_days : list[tuple]):

    for (n, m) in n_of_days:
        df[f'VIX_Momentumn{n}_{m}'] = (df[f'{n}_day_rolling_VIX'] / df[f'{m}_day_rolling_VIX']) -1

    return df.copy()

def add_lag_VIX(df: pd.DataFrame, n_of_days : list):

    for n in n_of_days:
        df[f'lag_VIX_{n}'] = df['VIX'].shift(n)

    return df.copy()

def add_VIX_skew(df: pd.DataFrame, n_of_days:list[int]):

    for n in n_of_days:
        df[f'VIX_Skew_{n}'] = df['VIX'].rolling(n).skew()

    return df.copy()

def add_VIX_kurt(df: pd.DataFrame, n_of_days:list[int]):

    for n in n_of_days:
        df[f'VIX_Kurt_{n}'] = df['VIX'].rolling(n).kurt()
    

    return df.copy()

def add_VIX_vol_ratios(df: pd.DataFrame, ratios:list[tuple]):

    for (n,m) in ratios:
        if m == 'VIX':
            df[f'VIX_Vol_momentum_{n}_{m}'] = (df[f'{n}_day_rolling_vol_of_VIX'] / df[f'{m}_day_rolling_vol_of_VIX']) -1        

    return df.copy()

# VIX/vol stats

def add_rolling_VIX_per_vol(df : pd.DataFrame, n_of_days : list, vol_days : list):

    for vol_day in vol_days:
        for n in n_of_days:
            df[f'{n}_day_rolling_VIX/vol{vol_day}'] = df[f'VIX/vol{vol_day}'].rolling(n).mean()

    return df.copy()

def add_rolling_vol_of_VIX_per_vol(df: pd.DataFrame, n_of_days : list, vol_days:list):

    for vol_day in vol_days:
        for n in n_of_days:
            df[f'{n}_day_rolling_vol_of_VIX/vol{vol_day}'] = df[f'VIX/vol{vol_day}'].rolling(n).std()

    return df.copy()

def add_VIX_per_vol_momentum(df: pd.DataFrame, n_of_days : list[tuple], vol_days:list):

    for vol_day in vol_days:
        for (n, m) in n_of_days:
            df[f'VIX/vol{vol_day}_Momentumn{n}_{m}'] = (df[f'{n}_day_rolling_VIX/vol{vol_day}'] / df[f'{m}_day_rolling_VIX/vol{vol_day}']) -1

    return df.copy()

def add_lag_VIX_per_vol(df: pd.DataFrame, n_of_days : list, vol_days:list):
    
    for vol_day in vol_days:
        for n in n_of_days:
            df[f'lag_VIX/vol{vol_day}_{n}'] = df[f'VIX/vol{vol_day}'].shift(n)

    return df.copy()

def add_VIX_per_vol_skew(df: pd.DataFrame, n_of_days:list[int], vol_days:list):
    for vol_day in vol_days:
        for n in n_of_days:
            df[f'VIX/vol{vol_day}_Skew_{n}'] = df[f'VIX/vol{vol_day}'].rolling(n).skew()

    return df.copy()

def add_VIX_per_vol_kurt(df: pd.DataFrame, n_of_days:list[int], vol_days:list):
    for vol_day in vol_days:
        for n in n_of_days:
            df[f'VIX/vol{vol_day}_Kurt_{n}'] = df[f'VIX/vol{vol_day}'].rolling(n).kurt()
    

    return df.copy()

def add_VIX_per_vol_vol_ratios(df: pd.DataFrame, ratios:list[tuple], vol_days:list):

    for vol_day in vol_days:
        for (n,m) in ratios:
            if m == 'VIX':
                df[f'VIX/vol{vol_day}_Vol_momentum_{n}_{m}'] = (
                    df[f'{n}_day_rolling_vol_of_VIX/vol{vol_day}'] / df[f'{m}_day_rolling_vol_of_VIX/vol{vol_day}']) -1     

    return df.copy()
                
