import pandas as pd
import numpy as np

def process_spx_option_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Extracting Important Columns
    df_clean = df[['date', 'exdate', 'cp_flag', 'strike_price', 'best_bid', 'best_offer', 'volume', 'open_interest', 'impl_volatility', 'delta', 'vega']]
    
    # Generating bid-ask spread and proportional bid-ask spread
    df_clean['bid-ask_spread'] = df_clean['best_offer'] - df_clean['best_bid']
    midpoint = (df_clean['best_bid'] + df_clean['best_offer']) / 2
    df_clean['proportional_bid-ask_spread'] = df_clean['bid-ask_spread'] / midpoint
    
    # Calculating Days until expiration
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['exdate'] = pd.to_datetime(df_clean['exdate'])
    t_to_exp = (df_clean['exdate'] - df_clean['date']).dt.days
    df_clean.insert(2, 't-to-exp', t_to_exp)
    
    # Cleaning Option DATA
    df_clean.set_index('date', inplace=True)
    df_filtered = df_clean[df_clean['t-to-exp'] < 366]
    df_filtered = df_filtered[(df_filtered['delta'].abs() >= 0.20) & (df_filtered['delta'].abs() <= 0.80)]
    
    # Extracting Important Data
    df_important = df_filtered.groupby('date').agg(
        ave_bid_ask_spread=('bid-ask_spread', 'mean'),
        ave_proportional_bid_ask_spread=('proportional_bid-ask_spread', 'mean'),
        implied_VOL=('impl_volatility', 'mean'),
        volume_traded_dayof=('volume', 'sum'),
        open_interest_dayof=('open_interest', 'sum'),
        ave_delta=('delta', 'mean'),
        ave_Vega=('vega', 'mean')
    ).reset_index()
    
    df_important.set_index('date', inplace=True)
    
    return df_important

# Usage example for SPX data
# SPX_file_path = '/Users/nicktaylor/Desktop/ThesisCSVs/BAS_Data/SPX_OptionData_2020Jan1-2022Dec31.csv'
# df_SPX_IMPORTANT = process_spx_option_data(SPX_file_path)
# print(df_SPX_IMPORTANT)