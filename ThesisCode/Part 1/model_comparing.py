# Importing OLS fitting with no predictive variable regluarization
from data_fitting import OLS_linear_regression , OLS_linear_regression_lagged , OLS_linear_regression_all_data_lagged, OLS_linear_regression_company_data_lagged

# Importing OLS fitting with forward stepwise selection (Best Subset)
from data_fitting import forward_selection_OLS, forward_selection_OLS_spreads_lagged, forward_selection_OLS_all_data_lagged, forward_selection_OLS_company_data_lagged

from data_import import process_and_merge_all_data



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





# Model comparisons for exisiting models (Returns DataFrame of model units)
def model_compare(df, company_ticker, target_variable, lag): 

    # Dropping NA values and, if present, the 'date' column
    df_cleaned = df.dropna()
    if 'date' in df_cleaned.columns:
        df_cleaned.set_index('date', inplace=True)


    # List of your function names as strings
    function_names = [
        'OLS_linear_regression', 'OLS_linear_regression_lagged', 
        'OLS_linear_regression_all_data_lagged', 'OLS_linear_regression_company_data_lagged',
        'forward_selection_OLS', 'forward_selection_OLS_spreads_lagged', 
        'forward_selection_OLS_all_data_lagged', 'forward_selection_OLS_company_data_lagged'
    ]

    # Initialize a list to store results
    results = []

    # Loop through each function name and execute it
    for func_name in function_names:
        # Dynamically call the function with the required parameters
        model, selected_vars, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred = globals()[func_name](df, company_ticker, target_variable, lag)
        
        # Append the results in a structured form
        results.append({
            'Function Name': func_name,
            'Number of Variables': len(selected_vars),
            'r2 Test': r2_test,
            'r2 Train': r2_train,
            'MSE_Test': mse_test,
            'MSE_Train': mse_train
        })

    # Convert the list of results into a DataFrame
    results_df = pd.DataFrame(results)

    # # access the selected variables from the first function:
    # selected_vars_first_func = results_df.loc[0, 'selected_vars']

    # print(results_df)


    # Export DataFrame to LaTeX
    latex_code = results_df.to_latex(index=False)

    # You might want to save this LaTeX code to a .tex file
    file_path = f'/Users/nicktaylor/Desktop/ThesisCSVs/ModelCompare/{company_ticker}/{company_ticker}_model_specs'  # Specify your file path
    with open(file_path, 'w') as latex_file:
        latex_file.write(latex_code)

    # print(results_df.to_latex())

    return results_df



# ------------------------------------ Plotting model comparisons ------------------------------------

# Training data and testing data - Predicticted data from specified model vs Actual figure 

def plot_actual_vs_testing_TrainTest(df, func_name, company_ticker, target_variable, lag):

    # Dropping NA values and, if present, the 'date' column
    df_cleaned = df.dropna()
    if 'date' in df_cleaned.columns:
        df_cleaned.set_index('date', inplace=True)

    model, selected_vars, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred = globals()[func_name](df, company_ticker, target_variable, lag)

    df_train_pred = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred}, index=y_train.index)
    df_test_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred}, index=y_test.index)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Training data plot
    df_train_pred.plot(kind='scatter', x='Actual', y='Predicted', alpha=0.5, color='blue', ax=ax[0], title='Training Data')

    # Draw a line representing the perfect predictions
    ax[0].plot(df_train_pred['Actual'], df_train_pred['Actual'], color='red', linewidth=2)

    # Testing data plot
    df_test_pred.plot(kind='scatter', x='Actual', y='Predicted', alpha=0.5, color='green', ax=ax[1], title='Testing Data')

    # Draw a line representing the perfect predictions
    ax[1].plot(df_test_pred['Actual'], df_test_pred['Actual'], color='red', linewidth=2)


    plt.savefig(f'/Users/nicktaylor/Desktop/ThesisCSVs/ModelCompare/{company_ticker}/Testing_vs_Training_spreads_{company_ticker}.png')

    # Showing Plot
    # plt.show()

    return 



# Plotting raw and predicted data
def plot_predSpreads_actualSpreads(df, func_name, company_ticker, target_variable, lag):

    # Dropping NA values and, if present, the 'date' column
    df_cleaned = df.dropna()
    if 'date' in df_cleaned.columns:
        df_cleaned.set_index('date', inplace=True)

    model, selected_vars, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred = globals()[func_name](df_cleaned, company_ticker, target_variable, lag)

    X_selected_original = df_cleaned[selected_vars].copy()

    X_selected = X_selected_original.dropna()

    y_new_pred = model.predict(X_selected)

    if df_cleaned.index.name != 'date':
        df_cleaned.set_index('date', inplace=True)


    df_new = df_cleaned[selected_vars].copy()
    df_new[f'{company_ticker}_ave_proportional_bid_ask_spread'] = df_cleaned[f'{company_ticker}_ave_proportional_bid_ask_spread']

    dropped_dates = X_selected_original.index.difference(X_selected.index)

    df_new = df_new.drop(dropped_dates)

    df_new['spreads_predicted'] = y_new_pred


    plt.figure(figsize=(200, 60))  # Set the figure size
    #Plotting raw data
    plt.plot(df_new.index, df_new[f'{company_ticker}_ave_proportional_bid_ask_spread'], label='Volume Traded')

    # #Plotting data within a year of experation
    plt.plot(df_new.index, df_new['spreads_predicted'], label='model predicted Spreads')

    # Labeling
    plt.title('Predicted Proportional Bid-Ask Spread and Proportional Bid-Ask Spread Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Proportional Spread')
    plt.legend()

    
    plt.savefig(f'/Users/nicktaylor/Desktop/XOM_MODELS_PLOTS/PXD_predicted_vs_actual_spreads_{func_name}.png')


    # Show the plot
    # plt.show()

    return df_new






df = process_and_merge_all_data('PXD') # Ticker issue
func_name = 'OLS_linear_regression_lagged'
company_ticker = 'PXD'
target_variable = 'PXD_ave_proportional_bid_ask_spread'
lag = 1


df_new = plot_predSpreads_actualSpreads(df, func_name, company_ticker, target_variable, lag)
print(df.columns)
print('----------------------------------------------------------------------------------------------------')
print(f'--------------------------------------------{func_name}--------------------------------------------')
print('----------------------------------------------------------------------------------------------------')



# # -------------- Test -------------
# company_names = ['COP', 'CVX', 'EOG', 'MPC', 'PSX', 'PXD', 'SLB', 'VLO', 'WMB', 'XOM']


# for company_name in company_names:

#     df = process_and_merge_all_data(company_name) # Ticker issue 
#     func_name = 'forward_selection_OLS'
#     company_ticker = company_name
#     target_variable = f'{company_name}_ave_proportional_bid_ask_spread'
#     lag = 1



#     # # Dataframe of model stats 
#     model_compare(df, company_ticker, target_variable, lag)

#     plot_actual_vs_testing_TrainTest(df, func_name, company_ticker, target_variable, lag)


#     plot_predSpreads_actualSpreads(df, func_name, company_ticker, target_variable, lag)


#     print(f'------------------------------------------------------------------------------')
#     print(f'--------------------------- {company_name} Just printed-----------------------')
#     print(f'------------------------------------------------------------------------------')


# ----------------------------------------------------------------------------------------------

# company_name = 'XOM'

# df = process_and_merge_all_data(company_name) # Ticker issue 
# func_name = 'forward_selection_OLS'
# company_ticker = company_name
# target_variable = f'{company_name}_ave_proportional_bid_ask_spread'
# lag = 1

# plot_predSpreads_actualSpreads(df, func_name, company_ticker, target_variable, lag)