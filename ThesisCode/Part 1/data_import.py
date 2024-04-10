from data_Clean import process_option_data,  OIL_data, format_historical_volatility, merge_dataframes_on_index


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def process_and_merge_all_data(ticker):

    # Independent Option Data

    c_file_path = f'/Users/nicktaylor/Desktop/ThesisCSVs/BAS_Data/{ticker}/{ticker}_Jan01-Dec22.csv'
    df_c = process_option_data(c_file_path, ticker)

    c_file_path_HV = f'/Users/nicktaylor/Desktop/ThesisCSVs/BAS_Data/{ticker}/{ticker}_HV_Jan01-Dec22.csv'
    df_c_HV = format_historical_volatility(c_file_path_HV, ticker)

    # Dependent Option Data

    SPX_file_path = f'/Users/nicktaylor/Desktop/ThesisCSVs/BAS_Data/Other_Market_Data/SPX/SPX_Jan01-Dec22.csv'
    df_SPX = process_option_data(SPX_file_path, 'SPX')

    VIX_file_path = f'/Users/nicktaylor/Desktop/ThesisCSVs/BAS_Data/Other_Market_Data/VIX/VIX_Jan01-Dec22.csv'
    df_VIX = process_option_data(VIX_file_path, 'VIX')

    # Dependent Market Data
    file_path_USO = '/Users/nicktaylor/Desktop/ThesisCSVs/BAS_Data/Other_Market_Data/USO_Jan01-Dec22.csv'
    df_USO = OIL_data(file_path_USO, 'USO')

    file_path_SPX_HV = '/Users/nicktaylor/Desktop/ThesisCSVs/BAS_Data/Other_Market_Data/SPX/SPX_HV_Jan01-Dec22.csv'
    df_SPX_HV = format_historical_volatility(file_path_SPX_HV, 'SPX')

    file_path_XLE_HV = '/Users/nicktaylor/Desktop/ThesisCSVs/BAS_Data/Other_Market_Data/XLE_HV_Jan01-Dec22.csv'
    df_XLE_HV = format_historical_volatility(file_path_XLE_HV, 'XLE')

    # Merge the DataFrames
    df_combined = merge_dataframes_on_index(df_c, df_c_HV, df_SPX, df_VIX, df_USO, df_SPX_HV, df_XLE_HV)

    return df_combined


# df = process_and_merge_all_data('PXD')

# file_path = '/Users/nicktaylor/Desktop/PXD.csv'  # Specify your file path
# df.to_csv(file_path, index=True)



# rows_with_nan = df[df.isna().any(axis=1)]
# print(rows_with_nan)

# #Plotting correlation heat map
# corr = df.corr()
# # Set up the matplotlib figure
# plt.figure(figsize=(24, 16))
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
# # Show the plot
# plt.show()





# file_path = '/Users/nicktaylor/Desktop/XOM.csv'  # Specify your file path
# df.to_csv(file_path, index=True)