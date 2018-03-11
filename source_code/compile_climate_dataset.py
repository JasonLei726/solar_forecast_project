'''
A data pre-processing and compilation script that compiles both hourly
weather and solar irradiance datasets into an hourly climate dataset.
If the two datasets have unequal number of observations, apply data
filling methods. The outcome is a continuous hourly climate dataset
written as a .csv file.

Note: This script is hard-coded to extract the features of interest for the
      climate dataset from 1 January 2005 to 31 August 2015.

This script does the following:
    1. Load and compile weather and solar irradiance dataset into climate dataset
    2. Apply data-filling methods:
        - Linear Interpolation
        - Daily Average Profiling
        - Climatological Estimation
    3. Write climate data to .csv file
'''


import pandas as pd
import numpy as np


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


def daily_average_profile(index, column, data):
    '''
    Function:
        Perform daily average profiling to single missing observation by
        computing the averages of the observations recorded at the same 
        hours of the previous and subsequent days.
    Arguments:
        index: Row index of dataset.
        column: Column index of dataset.
        data: Sequence of observations as a Pandas DataFrame of Series.
    '''
    if ((not np.isnan(data.iloc[index-24,column])) and (not np.isnan(data.iloc[index+24,column]))): 
        data.iloc[index,-1] = 0.5*float(data.iloc[index+24,-1]+data.iloc[index-24,-1])


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
    time = [data['DateTime'].dt.month[index], data['DateTime'].dt.day[index], data['DateTime'].dt.hour[index]]
    
    # Create dataframe containing observations of corresponding 'time' at all other years
    time_df = data[(data['DateTime'].dt.month == time[0]) & (data['DateTime'].dt.day == time[1]) & (data['DateTime'].dt.hour == time[2])][['DateTime']]
    
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


# 1. Load and compile weather and solar irradiance dataset into climate dataset
weather_data = pd.read_csv("Weather Data.csv")
solar_data = pd.read_csv("Solar Data.csv")
weather_data = weather_data.iloc[:, 1:]
solar_data = solar_data.iloc[:, 1:]

dataset = pd.concat([weather_data.iloc[:,0], weather_data.iloc[:, 5:]], axis=1)
dataset = pd.merge(dataset, solar_data.iloc[:, [0, 5]], on='DateTime', how='left')

# Get information about missing observations in dataset
missing_count, missing_percent = get_missing_info(dataset)

# 2. Apply data filling methods
# - Linear Interpolation
print("Linear Interpolation")
for n in range(len(dataset.columns)-1,len(dataset.columns)): 
    # Get list of index position of missing observations in current column
    print("\nColumn #", n)
    idx_list = dataset[dataset.iloc[:,n].isnull()].iloc[:,n]
    
    # Apply linear interpolation onto current feature column in dataset
    print("Applying linear interpolation...")
    for i in range(0, len(idx_list)):
        idx = idx_list.index[i]
        linear_interpolation(idx, n, dataset)
    print("Linear interpolation completed!")

# Get information about missing observations in dataset after applying linear interpolation
missing_count, missing_percent = get_missing_info(dataset)

# - Daily Average Profiling
print("Daily Average Profiling")
for n in range(len(dataset.columns)-1,len(dataset.columns)):
    # Get list of index position of missing observations in current column
    print("\nColumn #", n)
    idx_list = dataset[dataset.iloc[:,n].isnull()].iloc[:,n]
    
    # Apply daily average profiling onto current feature column in dataset
    print("Applying Daily Average Profiling...")
    for i in range(0, len(idx_list)):
        idx = idx_list.index[i]
        daily_average_profile(idx, n, dataset)
    print("Daily Average Profiling completed!")

# Get information about missing observations in dataset after applying daily average profiling
missing_count, missing_percent = get_missing_info(dataset)

# - Climatological Estimation
dataset['DateTime'] = pd.to_datetime(dataset['DateTime'])
print("Climatological Estimation")
for n in range(len(dataset.columns)-1,len(dataset.columns)):
    # Get list of index position of missing observations in current column
    print("\nColumn #", n)
    idx_list = dataset[dataset.iloc[:,n].isnull()].iloc[:,n]
    
    # Apply climatological estimation onto current feature column in dataset
    print("Applying climatological estimation...")
    for i in range(0, len(idx_list)):
        idx = idx_list.index[i]
        climatological_estimation(idx, n, dataset)       
    print("...climatological estimation ended.")
    
# Get information about missing observations in dataset after applying climatological estimation
missing_count, missing_percent = get_missing_info(dataset)

# 3. Write dataset into .csv file
dataset.to_csv("Climate Dataset.csv", sep=',')
