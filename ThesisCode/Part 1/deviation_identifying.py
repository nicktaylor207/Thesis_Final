from data_detSurp import identify_EarningsSurprise, identify_EarningsDates

from data_fitting import forward_selection_OLS

from modelDataEarn import get_df_W_pred

from data_import import process_and_merge_all_data


import pandas as pd




# ------------------------------ Earnings Date Data ---------------------------------------


def earningsDate_deviation_valueDiff(df, func_name, company_ticker, target_variable, lag, file_path, window_size):

    # Getting dataframe with actual spread and predictive spread
    df_new = get_df_W_pred(df, func_name, company_ticker, target_variable, lag)

    # Getting dates of earnings release
    df_Earnings_Dates = identify_EarningsDates(file_path)

    points_of_interest = pd.to_datetime(df_Earnings_Dates)

    df_new.index = pd.to_datetime(df_new.index)


    df_full = pd.DataFrame()


    for point in points_of_interest:
        start = point - pd.Timedelta(days=window_size)
        end = point + pd.Timedelta(days=window_size)


        df_window = df_new.loc[start:end]


        df_window['Deviation'] = (df_window[target_variable] - df_window['spreads_predicted']).abs()


        df_full = pd.concat([df_full, df_window])


    return df_full




def earningsDate_deviation_directionDiff(df, func_name, company_ticker, target_variable, lag, file_path, window_size):


    # Getting dataframe with actual spread and predictive spread
    df_new = get_df_W_pred(df, func_name, company_ticker, target_variable, lag)

    # Getting dates of earnings release
    df_Earnings_Dates = identify_EarningsDates(file_path)


    points_of_interest = pd.to_datetime(df_Earnings_Dates)

    df_new.index = pd.to_datetime(df_new.index)


    df_full = pd.DataFrame()

    for point in points_of_interest:
        # Define window range
        start = point - pd.Timedelta(days=window_size)
        end = point + pd.Timedelta(days=window_size)
        
        # Extract window
        df_window = df_new.loc[start:end].copy()

        df_window['Actual_Change'] = df_window[target_variable].diff()
        df_window['Predicted_Change'] = df_window['spreads_predicted'].diff()

        
        opposite_directions = (df_window['Actual_Change'] * df_window['Predicted_Change'] < 0)


        df_opposite = df_window[opposite_directions]


        df_full = pd.concat([df_full, df_opposite])

    return df_full







# ------------------------------ Earnings Surprise Date Data ---------------------------------------



def earnings_Surprise_Date_deviation_valueDiff(df, func_name, company_ticker, target_variable, lag, file_path, window_size):

    # Getting dataframe with actual spread and predictive spread
    df_new = get_df_W_pred(df, func_name, company_ticker, target_variable, lag)

    # Getting dates of earnings release     df_filtered, df_dates 
    df_all_data, df_EarningsSurprise_Dates = identify_EarningsSurprise(file_path)

    points_of_interest = pd.to_datetime(df_EarningsSurprise_Dates)

    df_new.index = pd.to_datetime(df_new.index)


    df_full = pd.DataFrame()

    for point in points_of_interest:
        start = point - pd.Timedelta(days=window_size)
        end = point + pd.Timedelta(days=window_size)


        df_window = df_new.loc[start:end]


        df_window['Deviation'] = (df_window[target_variable] - df_window['spreads_predicted']).abs()


        df_full = pd.concat([df_full, df_window])

    return df_full





def earnings_Surprise_Date_deviation_directionDiff(df, func_name, company_ticker, target_variable, lag, file_path, window_size):


    # Getting dataframe with actual spread and predictive spread
    df_new = get_df_W_pred(df, func_name, company_ticker, target_variable, lag)

    # Getting dates of earnings release
    df_all_data, df_EarningsSurprise_Dates = identify_EarningsSurprise(file_path)


    points_of_interest = pd.to_datetime(df_EarningsSurprise_Dates)

    df_new.index = pd.to_datetime(df_new.index)

    df_full = pd.DataFrame()


    for point in points_of_interest:
        # Define window range
        start = point - pd.Timedelta(days=window_size)
        end = point + pd.Timedelta(days=window_size)
        
        # Extract window
        df_window = df_new.loc[start:end].copy()

        df_window['Actual_Change'] = df_window[target_variable].diff()
        df_window['Predicted_Change'] = df_window['spreads_predicted'].diff()

        
        opposite_directions = (df_window['Actual_Change'] * df_window['Predicted_Change'] < 0)


        df_opposite = df_window[opposite_directions]

        df_full = pd.concat([df_full, df_opposite])

    return df_full







# -------------------------- TEST -----------------------------------

# df = process_and_merge_all_data('XOM') 
# func_name = 'forward_selection_OLS'
# company_ticker = 'XOM'
# target_variable = f'{company_ticker}_ave_proportional_bid_ask_spread'
# lag = 1

# file_path = f'/Users/nicktaylor/Desktop/ThesisCSVs/Earn_Data/{company_ticker}_Earn.csv'

# window_size = 5

# df_results = earnings_Surprise_Date_deviation_directionDiff(df, func_name, company_ticker, target_variable, lag, file_path, window_size)
# print(df_results)

