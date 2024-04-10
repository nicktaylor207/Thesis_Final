import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




# The Forecast Period End Date is the ending month and year of the fiscal period to which the estimate applies. 
# The Activation Date is the date that the forecast/actual was recorded by Thomson Reuters. 
# The Announce Date is the date that the forecast/actual was reported.




# Identiftying when earnings reports are above a certain number of SDs above 
def identify_EarningsSurprise(file_path):
    # Load the DataFrame from the given file path
    df = pd.read_csv(file_path)

    # Drop rows with any missing values
    df = df.dropna()

    # Convert 'ACTDATS' and 'ACTDATS_ACT' columns to datetime
    df['ACTDATS'] = pd.to_datetime(df['ACTDATS'])
    df['ACTDATS_ACT'] = pd.to_datetime(df['ACTDATS_ACT'])

    # Calculate the difference in days between 'ACTDATS_ACT' and 'ACTDATS'
    df['date_diff'] = (df['ACTDATS_ACT'] - df['ACTDATS']).dt.days

    # Filter rows where 'date_diff' is 8 days or less
    df = df[df['date_diff'] <= 8]

    # Calculate the difference between 'ACTUAL' and 'VALUE'
    df['price_diff'] = df['ACTUAL'] - df['VALUE']

    # Calculate mean and standard deviation of 'price_diff'
    mean_price_diff = df['price_diff'].mean()
    std_price_diff = df['price_diff'].std()

    # Defining Earnings Surprise: 1 SD above 
    plus_one_sd = mean_price_diff + 2 * std_price_diff
    minus_one_sd = mean_price_diff - 2 * std_price_diff

    # Filter the DataFrame based on 'price_diff' being greater than +1 SD or less than -1 SD
    df_filtered = df[(df['price_diff'] > plus_one_sd) | (df['price_diff'] < minus_one_sd)]

    # Return the filtered DataFrame

    df_ES_Dates = df_filtered['ACTDATS_ACT'].unique()
    df_dates = pd.to_datetime(df_ES_Dates, format='%m/%d/%y')

    return df_filtered, df_dates 



# Identifying all dates when earnings were released 
def identify_EarningsDates(file_path):
    # Load the DataFrame from the given file path
    df = pd.read_csv(file_path)

    df = df.dropna()
    announce_dates = df['ACTDATS_ACT'].unique()

    df_announce_date = pd.to_datetime(announce_dates)

    return df_announce_date



# Outputting distribution and plotting distribution 

def earn_Surprise_Distribution(file_path, company_ticker):

    df = pd.read_csv(file_path)

    df = df.dropna()

    df['ACTDATS'] = pd.to_datetime(df['ACTDATS'])
    df['ACTDATS_ACT'] = pd.to_datetime(df['ACTDATS_ACT'])

    df['date_diff'] = (df['ACTDATS_ACT'] - df['ACTDATS']).dt.days

    df = df[(df['date_diff'] <= 8)]

    df['price_diff'] = df['ACTUAL'] - df['VALUE']


    # Details of distribution
    summary = df['price_diff'].describe()


    # Plotting Distribution
    mean_price_diff = df['price_diff'].mean()
    std_price_diff = df['price_diff'].std()

    plt.figure(figsize=(10, 6))  # Adjust figure size as desired
    sns.kdeplot(df['price_diff'], fill=True)
    plt.title('Distribution of price_diff values')
    plt.xlabel('price_diff')
    plt.ylabel('Density')


    plt.axvline(mean_price_diff, color='r', linestyle='--', label='Mean')


    for i in range(1, 4):  # You can adjust the range if you want more or fewer standard deviations
        plt.axvline(mean_price_diff + i * std_price_diff, color='g', linestyle='--', label=f'+{i} SD' if i == 1 else '')
        plt.axvline(mean_price_diff - i * std_price_diff, color='g', linestyle='--', label=f'-{i} SD' if i == 1 else '')
        plt.text(mean_price_diff + i * std_price_diff, plt.ylim()[1]*0.95, f'+{i} SD', ha='center', va='top')
        plt.text(mean_price_diff - i * std_price_diff, plt.ylim()[1]*0.95, f'-{i} SD', ha='center', va='top')


    plt.savefig(f'/Users/nicktaylor/Desktop/predicted_vs_actual_spreads_{company_ticker}.png')

    # Show legend
    plt.legend()

    plt.show()

    return summary 









# -------------------------------------------------- TEST ---------------------------------------------------------

# XOM_E_file_path = '/Users/nicktaylor/Desktop/ThesisCSVs/Earn_Data/XOM_Earn.csv'




# XOM_ES, df_ES_Dates = identify_EarningsSurprise(XOM_E_file_path)
# print()
# print(XOM_ES)
# print()
# print(df_ES_Dates)
# print()

# company_name = 'XOM'
# file_path = f'/Users/nicktaylor/Desktop/ThesisCSVs/Earn_Data/{company_name}_Earn.csv'

# XOM_ED = identify_EarningsDates(file_path)

# print(XOM_ED)


# surprise_summary = earn_Surprise_Distribution(XOM_E_file_path, 'XOM')


# print(surprise_summary)
