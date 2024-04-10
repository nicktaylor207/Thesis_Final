import pandas as pd
from functools import reduce



# ----------------------------------------------Data Cleaning----------------------------------------------
# Option Cleaning
# Oil Pricing
# Historical Volitility

#Compile Dataset 

# ----------------------------------------------OPTION DATA----------------------------------------------

def process_option_data(file_path, ticker_symbol):

    """
    Cleans and formats option data by selecting approperiate options contracts and summarizing daily values .
    
    Parameters:
    - *file_paths: File path of the raw options data of interested company.
    - *ticker_symbol: The ticker symbol of the interested company.
    
    Returns:
    - A single DataFrame with interested data.
    """

    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Extracting Important Columns
    df_clean = df[['date', 'exdate', 'cp_flag', 'strike_price', 'best_bid', 'best_offer', 'volume', 'open_interest', 'impl_volatility', 'delta', 'vega']]

    

    
    # Generating bid-ask spread and proportional bid-ask spread
    df_clean['bid-ask_spread'] = df_clean['best_offer'] - df_clean['best_bid']
    midpoint = (df_clean['best_bid'] + df_clean['best_offer']) / 2
    df_clean['proportional_bid-ask_spread'] = df_clean['bid-ask_spread'] / midpoint

    
    # Calculating Days until expiration ('exdate' - 'date')
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['exdate'] = pd.to_datetime(df_clean['exdate'])
    t_to_exp = (df_clean['exdate'] - df_clean['date']).dt.days
    df_clean.insert(2, 't-to-exp', t_to_exp)
    
    # Cleaning Option DATA
    df_clean.set_index('date', inplace=True)
    df_ttoexp_filtered = df_clean[df_clean['t-to-exp'] < 366]
    df_filtered = df_ttoexp_filtered[(df_ttoexp_filtered['delta'].abs() >= 0.10) & (df_ttoexp_filtered['delta'].abs() <= 0.90)]
    
    # Extracting Important Data
    df_important = df_filtered.groupby('date').agg(
        **{
        f"{ticker_symbol}_ave_proportional_bid_ask_spread": ('proportional_bid-ask_spread', 'mean'),
        f"{ticker_symbol}_Implied_VOL": ('impl_volatility', 'mean'),
        f"{ticker_symbol}_volume_traded_dayof": ('volume', 'sum'),
        f"{ticker_symbol}_open_interest_dayof": ('open_interest', 'sum')
    }
    ).reset_index()

    
    df_important.set_index('date', inplace=True)


    # Forward fills nan values
    df_important = df_important.fillna(method='ffill')
    
    return df_important


# # COP options data --> 01/01/07 - 12/31/22
# COP_file_path = '/Users/nicktaylor/Desktop/ThesisCSVs/BAS_Data/COP/COP_Jan01-Dec22.csv'
# df_COP = process_option_data(COP_file_path, 'WMB')
# print(df_COP)

# rows = df_COP[df_COP.isna().any(axis=1)]
# print(rows)

# df_cleaned = df_COP.dropna()
# print(df_cleaned)

# rows_with_nan = df_COP[df_COP.isna().any(axis=1)]
# print(rows_with_nan)




# ----------------------------------------------OIL DATA----------------------------------------------
def OIL_data(file_path, ticker):

    """
    Formats historical price data of certain oil index.
    
    Parameters:
    - *file_paths: File path of the raw HV data of interested Oil Index.
    
    Returns:
    - A single DataFrame with interested data.
    """


    # Read the CSV file
    df = pd.read_csv(file_path)

    # Create a copy of the DataFrame to avoid modifying the original one
    df_important = df

    df_important.drop('NUMTRD', axis=1, inplace=True)
    df_important.drop('PERMNO', axis=1, inplace=True)
    df_important.drop('TICKER', axis=1, inplace=True)

    df_important['maxPriceDiff'] = df_important['ASKHI'] - df_important['BIDLO']

    df_important.rename(columns={
    'date': 'Date',
    'PRC': f'{ticker}_Price',
    'VOL': f'{ticker}_Volume',
    'VOL': f'{ticker}_Volume',
    'RET': f'{ticker}_Return',
    'maxPriceDiff': f'{ticker}_maxPriceDiff'

    }, inplace=True)

    df_important.drop('BIDLO', axis=1, inplace=True)
    df_important.drop('ASKHI', axis=1, inplace=True)

    # Setting 'Date as index
    df_important.set_index('Date', inplace=True)

    # Forward fills nan values
    df_important = df_important.fillna(method='ffill')

    return df_important


# ----------------------------------------------HISTORICAL VOL DATA----------------------------------------------

def format_historical_volatility(file_path, ticker_symbol):

    """
    Formats the historical value values dependent on: 10, 30, 91, 365, 730 days out.
    
    Parameters:
    - *file_paths: File path of the raw HV data of interested company.
    - *ticker_symbol: The ticker symbol of metric of interest.
    
    Returns:
    - A single DataFrame with interested data.
    """

    # Load the CSV data
    df_hv = pd.read_csv(file_path)
    
    columns_to_drop = ['secid', 'cusip', 'ticker', 'sic', 'index_flag', 'exchange_d', 'class', 'issue_type', 'industry_group']
    df_hv.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    days_to_keep = [10, 30, 91, 365, 730]
        
    # Filter the DataFrame to keep only the specified days
    df_filtered = df_hv[df_hv['days'].isin(days_to_keep)]

    df_pivoted = df_filtered.pivot(index='date', columns='days', values='volatility')

    df_pivoted.columns = [f'{ticker_symbol}_HistVolatility_{day}' for day in df_pivoted.columns]

    # Forward fills nan values
    df_pivoted = df_pivoted.fillna(method='ffill')

    return df_pivoted




# COP_file_path = '/Users/nicktaylor/Desktop/ThesisCSVs/BAS_Data/COP/COP_HV_Jan01-Dec22.csv'
# df_COP_HV = format_historical_volatility(COP_file_path, 'COP')
# print(df_COP_HV)


# rows = df_COP_HV[df_COP_HV.isna().any(axis=1)]
# print(rows)




# ----------------------------------------------Combined DataFrames----------------------------------------------

def merge_dataframes_on_index(*dfs):
    """
    Merges an arbitrary number of DataFrames on their index.
    
    Parameters:
    - *dfs: An arbitrary number of DataFrames to be merged.
    
    Returns:
    - A single DataFrame resulting from an outer merge of all input DataFrames on their indices.
    """
    # Convert the index of each DataFrame to datetime if not already done
    dfs_datetime_index = [df.set_index(pd.to_datetime(df.index)) if not pd.api.types.is_datetime64_any_dtype(df.index) else df for df in dfs]
    
    # Use reduce to iteratively merge all DataFrames on their index
    df_merged = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs_datetime_index)
    
    return df_merged


# big_df = merge_dataframes_on_index(df_WMB, df_WMB_HV)
# print(big_df.index)