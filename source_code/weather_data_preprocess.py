'''
A data pre-processing script tailored towards the Australian Bureau of
Meteorology's Automatic Weather Station (AWS) datasets with observations
recorded at a half-hourly resolution. 

This script reads and compresses the weather dataset into hourly resolution 
before applying data filling methods to all missing observations. The
outcome is a continuous hourly weather dataset written as a .csv file.

Note: This script is hard-coded to extract the features of interest from the
      weather dataset at a timeframe from 1 January 2005 to 8 September 2015.

This script does the following:
    1. Load weather data
    2. Extract features of interest from dataset
    3. Compress dataset into hourly resolution and extract timeframe of interests
    4. Apply data-filling methods:
        - Linear Interpolation
        - Proxy Estimation
        - Climatological Estimation (only to non-cloud associated features)
        - Assign zero values to missing cloud-associated features 
            (assume for clear sky conditions)
    5. Write weather data to .csv file     
'''


import numpy as np
import pandas as pd
import datetime


def get_missing_info(data):
    ''' 
    Function:
        Count the number of missing value in each feature columns in the 
        dataset and return it as a list of the total number and percentage
        of missing values within its respective feature column.
    Arguments:
        data: Sequence of observations as a Pandas DataFrame of Series.
    Return:
        missing_data_count: List containing the total missing data in each feature column.
        missing_data_percent: List containing the percentage of missing data in each feature column.
    '''
    missing_data_count = data.isnull().sum().tolist()
    missing_data_percent = [100*n/len(data) for n in missing_data_count]
    return missing_data_count, missing_data_percent


def get_mean_columns(data, col_list, mean_list):
    '''
    Function:
        Compute the mean of each specified feature columns in the dataset 
        and store the value into the passed list.
    Arguments:
        data: Sequence of observations as a Pandas DataFrame of Series.
        col_list: List containing specified column indices.
        mean_list: List to store computed mean values of each specified feature columns.
    '''
    for col in col_list:
        mean_list.append(data.iloc[:,col].mean())


def linear_interpolation(index, column, data):
    '''
    Function:
        Perform linear interpolation to single missing observation surrounded
        by non-missing observations. Therefore, fill in the missing 
        observation at index 'i' by the computed average of observations
        recorded at index 'i+1' and 'i-1'. If this missing observation is
        located at the edge of the dataset and is neighboured by a non-
        missing observation, assign that value to it.          
    Arguments:
        index: Row index of dataset.
        column: Column index of dataset.
        data: Sequence of observations as a Pandas DataFrame of Series.
    '''
    if ((index == 0) and (not np.isnan(data.iloc[1,column]))):
        data.iloc[0,column] = data.iloc[1,column]
    elif ((index == len(data)) and (not np.isnan(data.iloc[-2,column]))):
        data.iloc[-1,column] = data.iloc[-2,column] 
    elif ((not np.isnan(data.iloc[index-1,column])) and (not np.isnan(data.iloc[index+1,column]))):
        data.iloc[index,column] = 0.5*float(data.iloc[index+1,column]+data.iloc[index-1,column])
    else:
        pass
    

def proxy_estimation(index, data):
    '''
    Function:
        Perform proxy estimation to missing observation if all of its
        relational observations at the corresponding time have recorded.
    Arguments:
        index: Row index of dataset.
        data: Sequence of observations as a Pandas DataFrame of Series.      
        
    Note: 
        In this script, proxy estimation uses an empirical equation provided
        by the Australian Bureau of Meteorology [1] to calculate the relative
        humidity (U) as a function of dry bulb temperature (T), wet bulb
        temperature (Tw), and station level pressure (P) or as a function of
        dry bulb temperature (T) and dew point temperature (Td).
        Therefore, U = f(T, Tw, P) or U = f(T, Td).
        
        NB: The calculation of wet-bulb temperature is omitted due to
        the complication of mathematical manipulation.
        
        [1] www.bom.gov.au/climate/averages/climatology/relhum/calc-rh.pdf
    '''
    T = data.iloc[index, 4]        # Dry Bulb Temperature (deg C)
    Tw = data.iloc[index, 5]       # Wet Bulb Temperature (deg C)
    Td = data.iloc[index, 6]       # Dew Point Temperature (deg C)
    U = data.iloc[index, 7]        # Relative Humidity (%)
    P = data.iloc[index, -1]       # Station Level Pressure (hPa)
    
    # Check if any of the common variables (i.e. U or T) have a value... 
    if ((not np.isnan(U)) or (not np.isnan(T))):
        # Compute the following if both common variables have values
        if ((not np.isnan(U)) and (not np.isnan(T))):
            # Compute Td using U = f(T, Td)
            if (np.isnan(Td)):
                a = np.log((U/100)*np.exp(1.8096+((17.2694*T)/(237.3+T))))-1.8096
                Td = (237.3 * a)/(17.2694 - a)
                data.iloc[index,6] = Td
            # Compute P using U = f(T, Tw, P)
            elif ((np.isnan(P)) and (not np.isnan(Tw))):
                a = np.exp(1.8096+((17.2694*Tw)/(237.3+Tw)))
                b = 7.866 * np.power(10.0,-4) * (T - Tw) * (1 + (Tw/610))
                c = np.exp(1.8096+((17.2694*T)/(237.3+T)))
                P = (a - ((U*c)/100))/b
                data.iloc[index, -1] = P
            # Compute Tw using U = f(T, Tw, P)
            elif ((np.isnan(Tw)) and (not np.isnan(P))):
                # Note: As mentioned above, this section is omitted
                pass        
            else:
                pass
        # Compute for T using U = f(T, Tw, P) or U = f(T, Td)
        elif ((np.isnan(T)) and (not np.isnan(U))):
            if (not np.isnan(Td)):
                a = np.log((100/U)*np.exp(1.8096+((17.2694*Td)/(237.3+Td))))-1.8096
                T = (237.3 * a)/(17.2694 - a)
                data.iloc[index,4] = T
            elif ((not np.isnan(P)) and (not np.isnan(Tw))):
                if (not np.isnan(Td)):
                    a = np.exp(1.8096+((17.2694*Tw)/(237.3+Tw)))
                    b = 7.866 * np.power(10.0,-4) * P * (1+(Tw/610))
                    c = np.log((100/U)*np.exp(1.8096+((17.2694*Td)/(237.3+Td))))
                    T = (a+b*Tw-(U/100)*np.exp(c))/b   
                    data.iloc[index, 4] = T
                else:
                    pass
            else:
                pass
        # Compute for U using U = f(T, Tw, P) or U = f(T, Td)
        elif ((np.isnan(U)) and (not np.isnan(T))):
            if (not np.isnan(Td)):
                U = 100 * (np.exp(1.8096+((17.2694*Td)/(237.3+Td))))/(np.exp(1.8096+((17.2694*T)/(237.3+T))))
                data.iloc[index, 7] = U
            elif ((not np.isnan(P)) and (not np.isnan(Tw))):
                a = np.exp(1.8096+((17.2694*Tw)/(237.3+Tw)))
                b = 7.866 * np.power(10.0,-4) * P * (T - Tw) * (1 + (Tw/610))
                c = np.exp(1.8096+((17.2694*T)/(237.3+T)))
                U = 100 * (a - b) / c
                data.iloc[index, 7] = U
            else:
                pass
        else:
            pass


def climatological_estimation(index, column, data):
    '''
    Function:
        Perform climatological estimation to missing observations by computing
        the averages of non-missing observations recorded at the same time
        at all other years across the dataset.
        
        To elaborate, the function retrieves all observations recorded at the
        same time (i.e. month, day, and hour) at all other years and filter
        out any missing observations. The missing observation is then filled
        in by the computed average of the retrieved non-missing observations.
    Arguments:
        index: Row index of dataset.
        column: Column index of dataset.
        data: The dataset in which the function is to perform on.
    '''   
    # Setup
    unique_index_list = []  # List to store index position of non-missing observations taken in the same time at all other years
    value_list = []         # List to store non-missing observations to compute average
    
    # Get time of missing observation (i.e. time = [month, day, hour])
    time = data.iloc[index, 1:4].values
    
    # Create dataframe containing observations of corresponding 'time' at all other years
    time_df = data[(data.MM == time[0]) & (data.DD == time[1]) & (data.HH24 == time[2])][['MM','DD','HH24']]
    
    # Get index position of observations of corresponding 'time' at all other years
    for n in range(0, len(time_df)):
        unique_index_list.append(time_df.index[n])
        
    # Cycle through the list of index position for the current feature column and retrieve non-missing observations
    for i in range(0, len(unique_index_list)):
        if (not np.isnan(data.iloc[unique_index_list[i], column])):
            value_list.append(data.iloc[unique_index_list[i], column])
    
    # Fill in corresponding missing observation with computed average of non-missing data
    if len(value_list) > 0:
        data.iloc[index, column] = sum(value_list)/float(len(value_list))


# 1. Load weather data
raw_data = pd.read_csv("HM01X_Data_086282_999999998765053.txt")

# 2. Extract features of interest from dataset
col_of_interest = [2,3,4,5,6,14,16,18,20,22,24,26,28,32,34,36,38,40,42,44,46,62]
year_of_interest = [year for year in range(2005, 2016)]

dataset = pd.DataFrame()
for year in year_of_interest:
    data = raw_data[raw_data['Year Month Day Hour Minutes in YYYY'] == year].iloc[:, col_of_interest]
    if dataset.empty:
        dataset = data
    else:
        dataset = pd.concat([dataset, data])
        
dataset = dataset.apply(pd.to_numeric, errors='coerce')
dataset.reset_index(drop=True, inplace=True)

# Get information about missing observations and mean feature values in dataset
missing_count, missing_percent = get_missing_info(dataset)
mean_list = []
get_mean_columns(data=dataset, col_list = [n for n in range(5,22)], mean_list=mean_list)

# 3. Compress dataset into hourly resolution and extract timeframe of interests
dataset = dataset.groupby(np.arange(len(dataset))//2).mean()
del dataset['MI format in Local time']
dataset = dataset.iloc[:93481,:]    # Dataset timeframe: 1 Jan 2005 - 31 Aug 2015

# Get information about missing observations and mean feature values in dataset
missing_count, missing_percent = get_missing_info(dataset)
mean_list = []
get_mean_columns(data=dataset, col_list = [n for n in range(4,21)], mean_list=mean_list)


# 4. Apply data-filling methods
# - Linear Interpolation
print("Linear Interpolation")
for n in range(4,len(dataset.columns)):
    
    # Get list of index position of missing observations in current column
    print("\nColumn #", n)
    idx_list = dataset[dataset.iloc[:,n].isnull()].iloc[:,n]
    
    # Apply linear interpolation onto current feature column in dataset
    print("Applying linear interpolation...")
    for i in range(0, len(idx_list)):
        idx = idx_list.index[i]
        linear_interpolation(idx, n, dataset)
    print("Linear interpolation completed!")
    
# # Get information about missing observations and mean feature values in dataset after applying linear interpolation
missing_count, missing_percent = get_missing_info(dataset)
mean_list = []
get_mean_columns(data=dataset, col_list = [n for n in range(4,21)], mean_list=mean_list)
    
# - Proxy Estimation
col_of_interest = [4,5,6,7,20]
for col in col_of_interest:
    
    # Get list of index position of missing observations in current column
    idx_list = dataset[dataset.iloc[:,col].isnull()].iloc[:,col]
    
    # Apply proxy estimation onto current feature column in dataset
    for i in range(0, len(idx_list)):
        idx = idx_list.index[i]
        proxy_estimation(idx, dataset)
        
# Get information about missing observations and mean feature values in dataset after applying proxy estimation
missing_count, missing_percent = get_missing_info(dataset)
mean_list = []
get_mean_columns(data=dataset, col_list = [4,5,6,7,20], mean_list=mean_list)

# - Climatological Estimation
# Get information about mean feature values of interest in dataset applying climatological estimation
mean_list = []
col_of_interest = [4,5,6,7,8,9,10,11,20]
get_mean_columns(data=dataset, col_list = col_of_interest, mean_list=mean_list)

for col in col_of_interest:
    
    # Get list of index position of missing observations in current column
    print("Column #", col)
    idx_list = dataset[dataset.iloc[:,col].isnull()].iloc[:,col]
    
    # Apply climatological estimation onto current feature column in dataset
    print("Applying climatological estimation...")
    for i in range(0, len(idx_list)):
        idx = idx_list.index[i]
        climatological_estimation(idx, col, dataset)       
    print("...climatological estimation ended.")
     
# Get information about missing observations and mean feature values in dataset after applying climatological estimation
missing_count, missing_percent = get_missing_info(dataset)
mean_list = []
get_mean_columns(data=dataset, col_list = [4,5,6,7,8,9,10,11,20], mean_list=mean_list)  

# - Assign zero values to missing cloud-associated features
dataset.fillna(0, inplace=True)

# Get information about missing observations in dataset
missing_count, missing_percent = get_missing_info(dataset)

# Insert DateTime column into the dataset
time = pd.DataFrame([datetime.datetime(*x) for x in dataset.iloc[:, :4].values.astype(int)], columns=['DateTime'])
dataset = pd.concat([time, dataset], axis=1)

# 5. Write dataset into .csv file
dataset.to_csv("Weather Data.csv", sep=',')
