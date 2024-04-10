from data_import import process_and_merge_all_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score





# ------------------------- OLS Regression (All Variables) --------------------------------
def OLS_linear_regression(df, company_ticker, target_variable, lag):
    


    df_cleaned = df.dropna()
    if 'date' in df_cleaned.columns:
        df_cleaned.set_index('date', inplace=True)

    print(df_cleaned.columns)

    # Separate the features and the target variables
    X = df_cleaned.drop([target_variable], axis=1)
    y = df_cleaned[[target_variable]]

    # Test size and Random state for train/test split
    test_size = 0.2
    random_state = 42

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict on both the training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    r2_test = r2_score(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)

    variable_names = X.columns


    # Calculate Performance Metrics for Training Data
    print("Training Data:")
    print("R^2 Score:", r2_train)
    print("Mean Squared Error (MSE):", mse_train)
    print()

    # Calculate and Display Performance Metrics for Testing Data
    print("\nTesting Data:")
    print("R^2 Score:", r2_test)
    print("Mean Squared Error (MSE):", mse_test)
    print()

    # # Variables used
    # print(variable_names)



    # Returns R^2 score (Test, Train), Mean Squared error (Test, Train)
    return model, variable_names, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred


# -----------------------------------------Implementing lagged Terms --------------------------------------------------------------------------

def OLS_linear_regression_lagged(df, company_ticker, target_variable, lag_terms):

    for lag in range(1, 5):
        df[f'{target_variable}_{lag}'] = df[target_variable].shift(lag)

    model, variable_names, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred = OLS_linear_regression(df, company_ticker, target_variable, lag_terms)

    return model, variable_names, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred



# OLS - All Variables w/ all data from day before - past market sentament and comapany data
def OLS_linear_regression_all_data_lagged(df, company_ticker, target_variable, lag):

  df[f'{target_variable}_forward_1'] = df[target_variable].shift(1)

  df = df.drop(columns = target_variable)

  model, variable_names, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred = OLS_linear_regression(df, company_ticker, f'{target_variable}_forward_1', lag)


  return model, variable_names, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred





# OLS - All Variables w/ lagged company data - Current Maket Sentament + past company data
def OLS_linear_regression_company_data_lagged(df, company_ticker, target_variable, lag):

  
  for col in df.columns:
    if col.startswith(company_ticker) and col != target_variable:

      df[f'{col}_lag_{1}'] = df[col].shift(1)
      df = df.drop(columns = col)

  model, variable_names, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred = OLS_linear_regression(df, company_ticker, target_variable, lag)


  return model, variable_names, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred






# ------------------------- OLS Regression (Forward Selection - Best Subset) --------------------------------

from ISLP.models import ModelSpec as MS
from statsmodels.api import OLS

from functools import partial

from ISLP.models import \
    (Stepwise ,
     sklearn_selected)

from sklearn.metrics import mean_squared_error, r2_score


# Cp measure function
def nCp(sigma2, estimator, X, Y):
  "Negative Cp statistic"
  n, p = X.shape
  Yhat = estimator.predict(X)
  RSS = np.sum((Y - Yhat)**2)
  return -(RSS + 2 * p * sigma2) / n



# Finding best subset of predictive variables using Forward Stepwise Selection
def forward_selection_OLS(df, company_ticker, target_variable, lag):
    # Dropping NA values and, if present, the 'date' column
    df_cleaned = df.dropna()
    if 'date' in df_cleaned.columns:
        df_cleaned.set_index('date', inplace=True)

    print(df_cleaned.columns)

    # Splitting data into features and target based on 'target_variable'
    X = df_cleaned.drop(columns=[target_variable])
    Y = df_cleaned[target_variable]

    # Test size and Random state for train/test split
    test_size = 0.2
    random_state = 42

    # Splitting dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    # Fit design on training data
    design = MS(X_train.columns).fit(X_train)
    X_train_transformed = design.transform(X_train)

    # Calculating sigma2 using OLS on transformed training data
    sigma2 = OLS(y_train, X_train_transformed).fit().scale

    neg_Cp = partial(nCp, sigma2)

    # Applying forward selection strategy
    strategy = Stepwise.first_peak(design, direction='forward', max_terms=len(design.terms))
    spreads_Cp = sklearn_selected(OLS, strategy, scoring=neg_Cp)
    spreads_Cp.fit(X_train_transformed, y_train)

    # Extracting selected features from the strategy
    selected_features_names = spreads_Cp.selected_state_

    # Creating training and testing datasets with selected features only
    X_train_selected = X_train[list(selected_features_names)]
    X_test_selected = X_test[list(selected_features_names)]

    # Fitting Linear Regression model on selected features of the training set
    model = LinearRegression().fit(X_train_selected, y_train)

    # Predicting on training and testing sets
    y_train_pred = model.predict(X_train_selected)
    y_test_pred = model.predict(X_test_selected)


    r2_test = model.score(X_test_selected, y_test)
    r2_train = model.score(X_train_selected, y_train)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)

    variable_names = np.array(selected_features_names)


    # Evaluating the model
    print("Training Data:")
    print("R^2 score:", r2_train)
    print("Mean Squared Error:", mse_train)
    print()

    print("\nTesting Data:")
    print("R^2 score:", r2_test)
    print("Mean Squared Error:", mse_test)
    print()

    print(selected_features_names)

    # Returns R^2 score (Test, Train), Mean Squared error (Test, Train)
    return model, variable_names, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred




# -----------------------------------------Implementing lagged Terms --------------------------------------------------------------------------


# Ordinary Least Square - Best Subset w/ Historical spreads of past #(lag_terms) days included
def forward_selection_OLS_spreads_lagged(df, company_ticker, target_variable, lag_terms):

    for lag in range(1, 5):
        df[f'{target_variable}_{lag}'] = df[target_variable].shift(lag)

    model, variable_names, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred = forward_selection_OLS(df, company_ticker, target_variable, lag_terms)


    return model, variable_names, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred




# OLS - Best subest w/ all data from day before - past market sentament and comapany data
def forward_selection_OLS_all_data_lagged(df, company_ticker, target_variable, lag):
  
  df[f'{target_variable}_forward_1'] = df[target_variable].shift(1)

  df = df.drop(columns = target_variable)

  model, variable_names, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred = forward_selection_OLS(df, company_ticker, f'{target_variable}_forward_1', lag)


  return model, variable_names, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred





# OLS - Best Subset w/ lagged company data - Current Maket Sentament + past company data
def forward_selection_OLS_company_data_lagged(df, company_ticker, target_variable, lag):
  
  for col in df.columns:
    if col.startswith(company_ticker) and col != target_variable:

      df[f'{col}_lag_{1}'] = df[col].shift(1)
      df = df.drop(columns = col)

  model, variable_names, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred = forward_selection_OLS(df, company_ticker, target_variable, lag)


  return model, variable_names, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred












# df = process_and_merge_all_data('PXD')

# func_name = 'OLS_linear_regression_lagged'
# company_ticker = 'PXD'
# target_variable = 'PXD_ave_proportional_bid_ask_spread'
# lag = 1



# model, selected_vars, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred = globals()[func_name](df, company_ticker, target_variable, lag)

# print(r2_train, mse_train, r2_test, mse_test)
# print(selected_vars)







# -------------------------------------------------- TEST ---------------------------------------------------------

# # dataframe for usage ------ Looking for nan values
# df_comb = process_and_merge_all_data('COP')
# print(df_comb)

# rows_with_nan = df_comb[df_comb.isna().any(axis=1)]
# print(rows_with_nan)


# print('----------------OLS With No Regulation----------------')
# print('No Lagged terms')
# # OLS model all together
# model, variables_names, r2_test, r2_train, mse_test, mse_train = OLS_linear_regression(df_comb, 'XOM_ave_proportional_bid_ask_spread')

# print('Lagged terms')
# # OLS model all together with lagged 
# model, variables_names, r2_test, r2_train, mse_test, mse_train = OLS_linear_regression_lagged(df_comb, 'XOM_ave_proportional_bid_ask_spread', 5)

# print()
# print()

# print('----------------OLS With Regulation----------------')
# print('No Lagged terms')


# # Best_subset_OLS 
# model, variables_names, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred = forward_selection_OLS(df_comb, 'COP', 'COP_ave_proportional_bid_ask_spread', 1)
# print(variables_names, r2_test, r2_train, mse_test, mse_train)


# print('Lagged terms')
# # Best_subset_OLS with lagged
# model, variables_names, r2_test, r2_train, mse_test, mse_train = forward_selection_OLS_spreads_lagged(df_comb, 'XOM_ave_proportional_bid_ask_spread', 5)






# model, variable_names, r2_test, r2_train, mse_test, mse_train, y_train, y_train_pred, y_test, y_test_pred = OLS_linear_regression_company_data_lagged(df_comb, 'XOM', 'XOM_ave_proportional_bid_ask_spread', 1)