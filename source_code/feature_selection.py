'''
A draft script that employs a backward elimination algorithm to identify 
significant features to be retained and insignificant features to be removed 
from the dataset. Afterwards, several temporal features of interests are 
incorporated into the dataset.

This script does the following:
    1. Temporarily load climate dataset for feature selection process
    2. Apply backward elimination algorithm
    3. Reload climate dataset and remove irrelevant features
    4. Add temporal features of interests
'''


import numpy as np
import pandas as pd
import statsmodels.formula.api as sm

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


def load_dataset(file_name):
    '''
    Function:
        Read data from .csv file into returned Pandas DataFrame of Series. 
    Arguments:
        file_name: Name of .csv file
    Returns:
        Pandas DataFrame of series.
    '''
    return pd.read_csv(file_name)


def write_dataset(data, file_name):
    '''
    Function:
        Write data from Pandas DataFrame into .csv file of specified file name.
    Arguments:
        data: Sequence of observations as a Pandas DataFrame of Series.
        file_name: String text of .csv file name.
    '''
    data.to_csv(file_name, index=False, sep=',')
    

def difference(data, interval=1):
    '''
    Function:
        Apply lag differencing to dataset by subtracting the current 
        observations' values by the observation's value recorded at
        the previous time interval specified by the user.

        i.e. observation(t) = observation(t) - observation(t-interval)
    Arguments:
        data: Sequence of observations as a Pandas DataFrame of Series.
        interval: Specified interval prior to current interval.
    '''
    for column in range(0, len(data.columns)):
        data.iloc[:,column] = data.iloc[:,column].diff(interval)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    '''
    Function:
        Frame a time series dataset as a supervised learning dataset.
    Arguments:
        data: Sequence of observation as a list or NumPy array
        n_in: Number of lag observations as input (X)
        n_out: Number of observations as output (y)
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning
    '''
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    # Input Sequence (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        
    # Forecast Sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            
    # Combine all columns into dataframe to be returned
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    
    return agg


# 1. Temporarily load climate dataset for feature selection process
temp_dataset = load_dataset("Climate Dataset.csv")
temp_dataset = temp_dataset.iloc[:, 2:]   # Temporarily remove 'Date' column

# (Optional) Transform time series dataset as stationary
difference(temp_dataset, 8760)
temp_dataset.dropna(inplace=True)
temp_dataset.reset_index(inplace=True)
temp_dataset = temp_dataset.iloc[:, 1:]

# Transform time series dataset to supervised learning dataset
temp_dataset = series_to_supervised(temp_dataset, 1, 1, True)
temp_dataset.drop(temp_dataset.columns[18:35], axis=1, inplace=True)

# 2. Apply backward elimination
# Insert intercept column
dataset_opt = np.append(arr = np.ones((len(temp_dataset), 1)).astype(int), values = temp_dataset, axis = 1)

# Apply feature scaling
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset_opt = scaler.fit_transform(dataset_opt)

# Split dataset into feature matrix (X) and output vector (y)
X_opt = dataset_opt[:, :-1]
y_opt = dataset_opt[:, -1]

# Fit multilinear regression model
regressor_OLS = sm.OLS(endog=y_opt, exog=X_opt).fit()
regressor_OLS.summary()

# Below is used to remove feature columns of high p-value from the dataset
#remove_column = 11      # Insert feature column index
#X_opt = np.delete(X_opt, remove_column, 1)

# 3. Reload climate dataset and remove irrelevant features
dataset = load_dataset("Climate Dataset.csv")
dataset = dataset.iloc[:, 1:]
del dataset['Wind speed in km/h']
del dataset['Wind direction in degrees true']
del dataset['Cloud height (of first group) in feet']
del dataset['Cloud height (of second group) in feet']
del dataset['Cloud height (of third group) in feet']
del dataset['Cloud height (of fourth group) in feet']
dataset.columns = ['Date', 'DBT', 'WBT', 'Dew Point', 
                   'Relative Humidity', 'Vapour Pressure', 
                   'Saturated Vapour Pressure', 'Cloud Amount (1st group)', 
                   'Cloud Amount (2nd group)', 'Cloud Amount (3rd group)', 
                   'Cloud Amount (4th group)', 'Station Level Pressure', 'Mean GHI']

# Add temporal features of interests
dataset['Date'] = pd.to_datetime(dataset['Date'])

# - Seasonal Features
seasonal_months = [
        (dataset['Date'].dt.month == 1) | (dataset['Date'].dt.month == 2) | (dataset['Date'].dt.month == 12),
        (dataset['Date'].dt.month == 3) | (dataset['Date'].dt.month == 4) | (dataset['Date'].dt.month == 5),
        (dataset['Date'].dt.month == 6) | (dataset['Date'].dt.month == 7) | (dataset['Date'].dt.month == 8),
        (dataset['Date'].dt.month == 9) | (dataset['Date'].dt.month == 10) | (dataset['Date'].dt.month == 11)]
seasons = ['Summer', 'Autumn', 'Winter', 'Spring']
dataset['Season'] = np.select(seasonal_months, seasons)

labelencoder_season = LabelEncoder()
dataset['Season'] = labelencoder_season.fit_transform(dataset['Season'])
ohe_season = OneHotEncoder(categorical_features=[13])
dataset = ohe_season.fit_transform(dataset).toarray()
dataset = dataset[:, 1:]    # Remove dummy variable

# - Diurnal Features
dataset['Diurnal'] = np.where(dataset['Mean GHI'] > 0, 1, 0)

# - 'Day_Of_Hour' Features
dataset['Hour'] = dataset['Date'].dt.hour

labelencoder_hourly = LabelEncoder()
dataset.iloc[:, -1] = labelencoder_hourly.fit_transform(dataset.iloc[:, -1])
ohe_hourly = OneHotEncoder(categorical_features=[17])
dataset = ohe_hourly.fit_transform(dataset).toarray()
dataset = dataset[:, 1:]    # Remove dummy variable

# - 'Month_Of_Year' Features
dataset['Month'] = dataset['Date'].dt.month

labelencoder_monthly = LabelEncoder()
dataset.iloc[:, -1] = labelencoder_monthly.fit_transform(dataset.iloc[:, -1])
ohe_monthly = OneHotEncoder(categorical_features=[40])
dataset = ohe_monthly.fit_transform(dataset).toarray()
dataset = dataset[:, 1:]    # Remove dummy variable

# 5. Write to dataset
#write_dataset(dataset, "Insert_Name.csv")
