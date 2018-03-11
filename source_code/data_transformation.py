'''
A draft script that applies data transformation to the dataset.

This script does the following:
    1. Load Dataset
    2. (Optional) Transform time series dataset as stationary
    3. Transform time series dataset to supervised learning dataset
'''


import pandas as pd


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


# 1. Load Dataset
#dataset = load_dataset("Insert_dataset.csv") 

# 2. (Optional) Transform time series dataset as stationary
difference(dataset, 8760)
dataset.dropna(inplace=True)    
dataset.reset_index(inplace=True)
dataset = dataset.iloc[:, 1:]

# 3. Transform time series dataset to supervised learning dataset
dataset = series_to_supervised(dataset, 1, 1, True)
dataset.drop(dataset.columns[int(len(dataset.columns)/2):len(dataset.columns)-1], axis=1, inplace=True)

# 4. Write to .csv file
#write_dataset(dataset, "Insert_name.csv")
