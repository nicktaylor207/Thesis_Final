
# Earnings Dates usage
from deviation_identifying import earningsDate_deviation_valueDiff, earningsDate_deviation_directionDiff

# Earnings Surprise Dates usage
from deviation_identifying import earnings_Surprise_Date_deviation_valueDiff, earnings_Surprise_Date_deviation_directionDiff

# Compiling data
from data_import import process_and_merge_all_data

import pandas as pd




company_tickers = ['COP', 'CVX', 'EOG', 'MPC', 'PSX', 'PXD', 'SLB', 'VLO', 'WMB', 'XOM']





df_Interesting_Dates = pd.DataFrame()



for ticker in company_tickers:

    df = process_and_merge_all_data(ticker) 
    func_name = 'forward_selection_OLS'

    target_variable = f'{ticker}_ave_proportional_bid_ask_spread'

    lag = 1

    file_path = f'/Users/nicktaylor/Desktop/ThesisCSVs/Earn_Data/{ticker}_Earn.csv'

    window_size = 5


    # Earnings Dates
    df_earn_valueDiff = earningsDate_deviation_valueDiff(df, func_name, ticker, target_variable, lag, file_path, window_size) # [f'{ticker}_ave_proportional_bid_ask_spread', 'spreads_predicted', 'Deviation']
    df_earn_valueDiff = df_earn_valueDiff[[f'{ticker}_ave_proportional_bid_ask_spread', 'spreads_predicted', 'Deviation']]
    df_earn_valueDiff.rename(columns={'spreads_predicted': 'spreads_predicted_earn', 'Deviation': 'Deviation_earn'}, inplace=True)


    df_earn_directionDiff = earningsDate_deviation_directionDiff(df, func_name, ticker, target_variable, lag, file_path, window_size) # [f'{ticker}_ave_proportional_bid_ask_spread', 'spreads_predicted', 'Actual_Change', 'Predicted_Change']
    df_earn_directionDiff = df_earn_directionDiff[[f'{ticker}_ave_proportional_bid_ask_spread', 'spreads_predicted', 'Actual_Change', 'Predicted_Change']]
    df_earn_directionDiff.rename(columns={'Actual_Change': 'Actual_Change_earn', 'Predicted_Change': 'Predicted_Change_earn'}, inplace=True)

    # Earnings Surprises Dates
    df_earnSurp_valueDiff = earnings_Surprise_Date_deviation_valueDiff(df, func_name, ticker, target_variable, lag, file_path, window_size)
    df_earnSurp_valueDiff = df_earnSurp_valueDiff[[f'{ticker}_ave_proportional_bid_ask_spread', 'spreads_predicted', 'Deviation']]
    df_earnSurp_valueDiff.rename(columns={'spreads_predicted': 'spreads_predicted_earnSurp', 'Deviation': 'Deviation_earnSurp'}, inplace=True)


    df_earnSurp_directionDiff = earnings_Surprise_Date_deviation_directionDiff(df, func_name, ticker, target_variable, lag, file_path, window_size)
    df_earnSurp_directionDiff = df_earnSurp_directionDiff[[f'{ticker}_ave_proportional_bid_ask_spread', 'spreads_predicted', 'Actual_Change', 'Predicted_Change']]
    df_earnSurp_directionDiff.rename(columns={'Actual_Change': 'Actual_Change_earnSurp', 'Predicted_Change': 'Predicted_Change_earnSurp'}, inplace=True)


    # Find the index of the row with the largest 'Deviation_earn' value
    max_deviation_earn_idx = df_earn_valueDiff['Deviation_earn'].idxmax()

    # Find the index of the row with the largest 'Deviation_earnSurp' value
    max_deviation_earnSurp_idx = df_earnSurp_valueDiff['Deviation_earnSurp'].idxmax()



    df_combined = pd.concat([df_earn_valueDiff, df_earn_directionDiff, df_earnSurp_valueDiff, df_earnSurp_directionDiff], axis=1)


    # print(df_combined.loc[max_deviation_earn_idx])
    # print(df_combined.loc[max_deviation_earnSurp_idx])


    df_Interesting_Dates[f'{ticker}_dates'] = max_deviation_earn_idx, max_deviation_earnSurp_idx

    print(f'---------------------------{ticker} dates printed---------------------------')

print(df_Interesting_Dates)

file_path = '/Users/nicktaylor/Desktop/DeviationDates.csv'  # Specify your file path
df_Interesting_Dates.to_csv(file_path, index=True)







