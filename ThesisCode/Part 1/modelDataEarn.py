from data_detSurp import identify_EarningsSurprise, identify_EarningsDates
from data_fitting import forward_selection_OLS_spreads_lagged

from data_fitting import forward_selection_OLS

from data_import import process_and_merge_all_data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def get_df_W_pred(df, func_name, company_ticker, target_variable, lag):

    df_cleaned = df.dropna()
    if 'date' in df_cleaned.columns:
        df_cleaned.set_index('date', inplace=True)

    model, selected_vars, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred = forward_selection_OLS_spreads_lagged(df_cleaned, company_ticker, target_variable, lag)

    X_selected_original = df_cleaned[selected_vars].copy()

    # Dropping missing values from selected variables
    X_selected = X_selected_original.dropna()

    y_new_pred = model.predict(X_selected)

    if df_cleaned.index.name != 'date':
        df_cleaned.set_index('date', inplace=True)

    df_new = df_cleaned[selected_vars].copy()
    df_new[f'{company_ticker}_ave_proportional_bid_ask_spread'] = df_cleaned[f'{company_ticker}_ave_proportional_bid_ask_spread']

    # Dealing with droped data from selected variables
    dropped_dates = X_selected_original.index.difference(X_selected.index)
    df_new = df_new.drop(dropped_dates)
    df_new['spreads_predicted'] = y_new_pred   


    return df_new


# # -----------------------TEST-----------------------------------------------------------

# file_path = f'/Users/nicktaylor/Desktop/df_comb.csv'
# df = pd.read_csv(file_path)
# func_name = 'forward_selection_OLS'
# company_ticker = 'XOM'
# target_variable = f'{company_ticker}_ave_proportional_bid_ask_spread'
# lag = 1

# df = get_df_W_pred(df, func_name, company_ticker, target_variable, lag)
# print(df)

# # --------------------------------------------------------------------------------------



# Plotting Raw and Rredicted and Earnings Realeases
def plot_predSpreads_actualSpreads_Earnings(df, func_name, company_ticker, target_variable, lag, file_path):

    df_new = get_df_W_pred(df, func_name, company_ticker, target_variable, lag)


    announce_dates = identify_EarningsDates(file_path)


    plt.figure(figsize=(200, 60))  # Set the figure size
    #Plotting raw data
    plt.plot(df_new.index, df_new[f'{company_ticker}_ave_proportional_bid_ask_spread'], label='Average Proportional Bid-Ask Spread')

    #Plotting data within a year of experation
    plt.plot(df_new.index, df_new['spreads_predicted'], label='model predicted Spreads')


    for date in announce_dates:
        plt.axvline(x=date, color='r', linestyle='--', label='Important Date')
        plt.text(date, plt.ylim()[1]*0.95, date.strftime('%Y-%m-%d'), rotation=45, ha='right', va='top')



    # Labeling
    plt.title('Predicted Proportional Bid-Ask Spread and Proportional Bid-Ask Spread Over Time (Options Expiring Within a Year)')
    plt.xlabel('Date')
    plt.ylabel('Spread')
    plt.legend()

    plt.savefig(f'/Users/nicktaylor/Desktop/XOM_EarnModel_Plots/predicted_{func_name}_vs_actual_spreads_W_EarningsSurpriseDates{company_ticker}.png')

    # Show the plot
    # plt.show()

    return





# Plotting Raw and Rredicted and Earnings Realeases
def plot_predSpreads_actualSpreads_EarningsSurprising(df, func_name, company_ticker, target_variable, lag, file_path):

    df_new = get_df_W_pred(df, func_name, company_ticker, target_variable, lag)


    df, earningsSurprise_dates = identify_EarningsSurprise(file_path)


    plt.figure(figsize=(200, 60))  # Set the figure size
    #Plotting raw data
    plt.plot(df_new.index, df_new[f'{company_ticker}_ave_proportional_bid_ask_spread'], label='Average Proportional Bid-Ask Spread')

    #Plotting data within a year of experation
    plt.plot(df_new.index, df_new['spreads_predicted'], label='model predicted Spreads')


    for date in earningsSurprise_dates:

        date_num = mdates.date2num(date)

        plt.axvline(x=date_num, color='r', linestyle='--', label='Important Date')
        
        plt.text(date_num, plt.ylim()[1]*0.95, date_num, rotation=45, ha='right', va='top')



    # Labeling
    plt.title('Predicted Proportional Bid-Ask Spread and Proportional Bid-Ask Spread Over Time')
    plt.xlabel('Date')
    plt.ylabel('Spread')
    plt.legend()

    plt.savefig(f'/Users/nicktaylor/Desktop/XOM_EarnModel_Plots/predicted_{func_name}_vs_actual_spreads_W_EarningsSurpriseDates{company_ticker}.png')

    # Show the plot
    # plt.show()

    return




from pandas import to_datetime, DateOffset

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_predSpreads_actualSpreads_EarningsSurprising_interestTimes(df, func_name, company_ticker, target_variable, lag, file_path, focus_date, days_range=5):
    df_new = get_df_W_pred(df, func_name, company_ticker, target_variable, lag)
    df, earningsSurprise_dates = identify_EarningsSurprise(file_path)

    # Ensure the focus_date is in datetime format
    focus_date_dt = to_datetime(focus_date)
    
    # Calculate the range around the focus date
    start_date = focus_date_dt - DateOffset(days=days_range)
    end_date = focus_date_dt + DateOffset(days=days_range)

    plt.figure(figsize=(10, 6))  # Adjusted for more standard viewing

    if not isinstance(df_new.index, pd.DatetimeIndex):
        df_new.index = pd.to_datetime(df_new.index)


    # Plotting average bid-ask spread
    plt.plot(df_new.index, df_new[f'{company_ticker}_ave_proportional_bid_ask_spread'], label='Average Proportional Bid-Ask Spread')
    # Plotting predicted spreads
    plt.plot(df_new.index, df_new['spreads_predicted'], label='model predicted Spreads')

    # Highlight earnings surprise dates
    for date in earningsSurprise_dates:
        if start_date <= date <= end_date:
            date_num = mdates.date2num(date)  # Convert Timestamp to Matplotlib's date format
            plt.axvline(x=date_num, color='r', linestyle='--', label='Earnings Surprise Date' if 'Earnings Surprise Date' not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.text(date_num, plt.ylim()[1]*0.95, date.strftime('%Y-%m-%d'), rotation=45, ha='right', va='top')


    # Adjusting x-axis to focus on the specific date range
    plt.xlim(mdates.date2num(start_date), mdates.date2num(end_date))


    # Labeling
    plt.title('Close up')
    plt.xlabel('Date')
    plt.ylabel('Spread')
    plt.legend()

    plt.xticks(rotation=90)

    # Save the plot
    plt.savefig(f'/Users/nicktaylor/Desktop/XOM_EarnModel_Plots/{company_ticker}_focused_plot_{func_name}_{focus_date}.png')
    
    # Optional: Show the plot
    # plt.show()






# # -------------- Earnings Test Focus Date ------------- 


file_path = '/Users/nicktaylor/Desktop/PXD.csv'
df = pd.read_csv(file_path)



print(df.columns)
func_name = 'forward_selection_OLS_spreads_lagged'
company_ticker = 'PXD'
target_variable = 'PXD_ave_proportional_bid_ask_spread'
lag = 1

file_path = '/Users/nicktaylor/Desktop/ThesisCSVs/Earn_data/PXD_Earn.csv'

focus_date = '2020-02-19'

# 2008-02-01, 2012-01-31, 2017-01-31, 2021-02-02


# 12, 17, 21, 


plot_predSpreads_actualSpreads_EarningsSurprising_interestTimes(df, func_name, company_ticker, target_variable, lag, file_path, focus_date, days_range=5)










# # -------------- Earnings Test ------------- 
# company_names = ['XOM', 'PXD']

# for comp in  company_names :
#     file_path = f'/Users/nicktaylor/Desktop/{comp}.csv'
#     df = pd.read_csv(file_path)


#     print(df)
#     func_name = 'forward_selection_OLS_spreads_lagged'
#     company_ticker = comp
#     target_variable = f'{comp}_ave_proportional_bid_ask_spread'
#     lag = 1

#     file_path = f'/Users/nicktaylor/Desktop/ThesisCSVs/Earn_data/{comp}_Earn.csv'


#     plot_predSpreads_actualSpreads_EarningsSurprising(df, func_name, company_ticker, target_variable, lag, file_path)


#     print('------------------------------------------------------------------------------')
#     print('---------------------------XOM Just printed-----------------------')
#     print('------------------------------------------------------------------------------')




# -------------- Earnings Surprise Test ------------- 


# company_names = ['XOM']

# for company_name in company_names:

#     df = process_and_merge_all_data(company_name) # Ticker issue 
#     func_name = 'forward_selection_OLS_spreads_lagged'
#     company_ticker = company_name
#     target_variable = f'{company_name}_ave_proportional_bid_ask_spread'
#     lag = 1

#     file_path = f'/Users/nicktaylor/Desktop/ThesisCSVs/Earn_data/{company_name}_Earn.csv'


#     plot_predSpreads_actualSpreads_EarningsSurprising(df, func_name, company_ticker, target_variable, lag, file_path)


#     print(f'------------------------------------------------------------------------------')
#     print(f'--------------------------- {company_name} Just printed-----------------------')
#     print(f'------------------------------------------------------------------------------')


