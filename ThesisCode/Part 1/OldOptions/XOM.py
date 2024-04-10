import pandas as pd
import numpy as np


# ----------------------------------------------XOM DATA----------------------------------------------
# XOM options data --> 01/01/20 - 12/31/22
XOM_file_path = '/Users/nicktaylor/Desktop/ThesisCSVs/BAS_Data/XOM_OptionData_2020Jan1-2022Dec31.csv'
df_XOM = pd.read_csv(XOM_file_path)


# ----------------------------------------------Formatting DATA----------------------------------------------
# Calculating Bid-Ask Spread,  Proportional Bid-Ask Spread,  Days until Expiration
# Adding into columns in XOM Dataframe

#Extracting Important Columns
df_XOM_Clean = df_XOM[['date', 'exdate', 'cp_flag', 'strike_price', 'best_bid', 'best_offer', 'volume', 'open_interest', 'impl_volatility', 'delta', 'vega']]

#Generating bid-ask spread and proportional bid-ask spread
df_XOM_Clean['bid-ask_spread'] = df_XOM_Clean['best_offer'] - df_XOM_Clean['best_bid']

midpoint = (df_XOM_Clean['best_bid'] + df_XOM_Clean['best_offer']) / 2
df_XOM_Clean['proportional_bid-ask_spread'] = df_XOM_Clean['bid-ask_spread'] / midpoint

# Calculating Days until experations ('exdate' - 'date')
df_XOM_Clean['date'] = pd.to_datetime(df_XOM_Clean['date'])   # 'date' correct formatting
df_XOM_Clean['exdate'] = pd.to_datetime(df_XOM_Clean['exdate'])   # 'exdate' correct formatting

# Calculate the number of days from 'date' to 'exdate'
t_to_exp = (df_XOM_Clean['exdate'] - df_XOM_Clean['date']).dt.days
df_XOM_Clean.insert(2, 't-to-exp', t_to_exp) #t-to-exp' as the third column


# ----------------------------------------------Cleaning Option DATA----------------------------------------------
#              - Dropping Option Contracts with:
#                   - Expiring in more than 1 Year
#                   - Option Very In/Out of the Money
#                              - Calls: 0.20 <= Delta <= 0.8
#                              - Puts: -0.20 <= Delta <= -0.8

#Index the date
df_XOM_Clean.set_index('date', inplace=True)

#Filtering out spreads that are greater than a year until experation
df_XOM_ttoexp_filtered = df_XOM_Clean[df_XOM_Clean['t-to-exp'] < 366]

# #Filtering out spreads that are greater than a year until experation and delta (0.20 <= |Delta| <= 0.80)
df_XOM_t_to_exp_delta_filtered = df_XOM_ttoexp_filtered[(df_XOM_ttoexp_filtered['delta'].abs() >= 0.20) & (df_XOM_ttoexp_filtered['delta'].abs() <= 0.80)]



# ----------------------------------------------Extracting Important Data----------------------------------------------
# Important Data Per Day:
#       - Average Bid-Ask Spread
#       - Average Proportional Bid-Ask Spread
#       - Average Implied Volitility
#       - Volume Traded
#       - Open Interest

df_XOM_IMPORTANT = df_XOM_t_to_exp_delta_filtered.groupby('date').agg(
    XOM_ave_bid_ask_spread=('bid-ask_spread', 'mean'),
    XOM_ave_proportional_bid_ask_spread=('proportional_bid-ask_spread', 'mean'),
    XOM_Implied_VOL=('impl_volatility', 'mean'),
    XOM_volume_traded_dayof=('volume', 'sum'),
    XOM_open_interest_dayof=('open_interest', 'sum')
).reset_index()

df_XOM_IMPORTANT.set_index('date', inplace=True)


print(df_XOM_IMPORTANT)