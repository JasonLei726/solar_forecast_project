'''
A data pre-processing script tailored towards the Australian Bureau of
Meteorology's One Minute Solar datasets with observations recorded at
a one-minute resolution.

This script reads and compresses the solar irradiance dataset into hourly 
resolution before applying data filling methods to all missing observations.
The outcome is a continuous hourly solar irradiance dataset written as a .csv file.

Note: This script is hard-coded to extract the features of interest from the
      solar irradiance dataset from 1 January 2005 to 31 August 2015.

This script does the following:
    1. Cycle through multiple solar irradiance datasets and compile an hourly
       solar irradiance dataset containing features and timeframe of interest. 
    2. Apply data-filling methods:
        - Linear Interpolation
        - Proxy Estimation
        - Climatological Estimation
    3. Write solar irradiance data to .csv file    
'''


import numpy as np
import pandas as pd
import datetime


def get_missing_info_1(data):
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


def get_missing_info_2(data, missing_count, missing_percent):
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
        
    Note:
        The year and month of the current dataset are inserted for referencing
    '''
    year = data.iloc[0,0]
    month = data.iloc[0,1]
    count = data.isnull().sum().tolist()       
    percent = [100*n/len(data) for n in count]
    
    count.insert(0, year)
    count.insert(1, month) 
    percent.insert(0, year)
    percent.insert(1, month)
    
    missing_count.append(count)
    missing_percent.append(percent)


def get_mean_columns_1(data, col_list, mean_list):
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

   
def get_mean_columns_2(data, col_list, mean_list):
    '''
    Function:
        Compute the mean of each specified feature columns in the dataset 
        and store the value into the passed list.
    Arguments:
        data: Sequence of observations as a Pandas DataFrame of Series.
        col_list: List containing specified column indices.
        mean_list: List to store computed mean values of each specified feature columns.
        
    Note:
        The year and month of the current dataset are inserted for referencing
    '''
    year = data.iloc[0,0]
    month = data.iloc[0,1]
    mean = data.iloc[:,col_list].mean().tolist()
    mean.insert(0, year)
    mean.insert(1, month)
    mean_list.append(mean)


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
        In this script, proxy estimation uses the equation of global
        horizontal irradiance as follows:
            
            Global Horizontal Irradiance
                = Direct Normal Irradiance * cos(Zenith Angle) + Diffuse Horizontal Irradiance
                = Direct Horizontal Irradiance + Diffuse Horizontal Irradiance
    '''
    GHI = data.iloc[index,5]        # Global Horizontal Irradiance
    DirNI = data.iloc[index,6]      # Direct Normal Irradiance
    DiffHI = data.iloc[index,7]     # Diffuse Horizontal Irradiance
    DirHI = data.iloc[index,9]      # Direct Horizontal Irradiance
    Zenith = data.iloc[index,10]    # Zenith Angle
    
    if (not np.isnan(DiffHI)):
        if (not np.isnan(Zenith) and (not np.isnan(DirNI))):
            GHI = DirNI * np.cos(Zenith) + DiffHI
            data.iloc[index,5] = GHI
        elif (not np.isnan(DirHI)):
            GHI = DirHI + DiffHI
            data.iloc[index,5] = GHI
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
    time = data.iloc[index, 2:6].values
    
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


# Setup to construct hourly solar dataset from 1st Jan 2005 to 31st Aug 2015
year_list = ['2005','2006','2007','2008','2009','2010','2011','2012','2013','2014', '2015']
month_list = ['01','02','03','04','05','06','07','08','09','10','11','12']
col_of_interest = [2,3,4,5,6,7,12,17,22,27,35]
dataset = pd.DataFrame()

init_miss_count = []    # List to store information about missing observations in raw dataset
init_miss_percent = []  # List to store information about missing observations in raw dataset
final_miss_count = []   # List to store information about missing observations in compiled dataset
final_miss_percent = [] # List to store information about missing observations in compiled dataset
init_mean_list = []     # List to store information about mean feature values in raw dataset
final_mean_list = []    # List to store information about mean feature values in compiled dataset

# 1. Compile hourly solar irradiance dataset containing features and timeframe of interest.
for year in year_list:
    for month in month_list:
        # Exclude observations beyond 31 August 2015
        if ((year == '2015') and (month == '09')):
            break
        else:
            # Import dataset and extract features of interest
            print("\nDataset (Year: {}, Month: {})".format(year, month))
            print("Importing Dataset...")
            file_name = "sl_086282_" + year + "_" + month + ".txt"
            data = pd.read_csv(file_name, delimiter=',')
            data = data.apply(pd.to_numeric, errors='coerce')
            data = data.iloc[:, col_of_interest]
            
            # Get information about missing observations and mean feature values in dataset
            get_missing_info_2(data, init_miss_count, init_miss_percent)
            get_mean_columns_2(data, [x for x in range(4, 11)], init_mean_list)
            
            # Convert dataset into hourly resolution
            print("Convert dataset into hourly resolution...")
            data = data.groupby(np.arange(len(data))//60).mean()
            
            # Get information about missing observations and mean feature values in dataset
            get_missing_info_2(data, final_miss_count, final_miss_percent)    
            get_mean_columns_2(data, [x for x in range(4,11)], final_mean_list)
            
            # Insert current hourly dataset into new dataframe
            print("Inserting dataset into final dataset...")
            if dataset.empty:
                dataset = data
            else:
                dataset = pd.concat([dataset, data])
dataset.reset_index(drop=True, inplace=True)
del dataset['MI format in Local standard time']               
                
# Get information about missing observations and mean feature values in dataset
init_miss_count = pd.DataFrame(init_miss_count)
init_miss_percent = pd.DataFrame(init_miss_percent)
final_miss_count = pd.DataFrame(final_miss_count)
final_miss_percent = pd.DataFrame(final_miss_percent)
init_mean_list = pd.DataFrame(init_mean_list)
final_mean_list = pd.DataFrame(final_mean_list)
init_mean_list.iloc[:,3:].mean()
final_mean_list.iloc[:, 3:].mean()

# Insert DateTime column into the dataset
time = pd.DataFrame([datetime.datetime(*x) for x in dataset.iloc[:, :4].values.astype(int)], columns=['DateTime'])
dataset = pd.concat([time, dataset], axis=1)

# Remove duplicated DateTime entries (Total observations = 92820)
dupl_count = time['DateTime'].value_counts()
dupl_time = dupl_count[dupl_count == 2].index.sort_values()
get_dupl_time = time.loc[time.iloc[:,0].isin(dupl_time)]
remove_idx = []
for i in range(0, len(get_dupl_time)):
    if i % 2 == 1:
        remove_idx.append(get_dupl_time.index[i])
dataset = dataset.drop(dataset.index[remove_idx])

# 2. Apply data filling methods
# Get information about missing observations and mean feature values in dataset
missing_count, missing_percent = get_missing_info_1(dataset)
mean_list = []
get_mean_columns_1(data=dataset, col_list = [5,6,7,8,9,10], mean_list=mean_list)

# - Linear Interpolation
print("Linear Interpolation")
for n in range(5,len(dataset.columns)-1):
    # Get list of index position of missing observations in current column
    print("\nColumn #", n)
    idx_list = dataset[dataset.iloc[:,n].isnull()].iloc[:,n]
    
    # Apply linear interpolation onto current feature column in dataset
    print("Applying linear interpolation...")
    for i in range(0, len(idx_list)):
        idx = idx_list.index[i]
        linear_interpolation(idx, n, dataset)
    print("Linear interpolation completed!")
    
# Get information about missing observations and mean feature values in dataset after applying linear interpolation
missing_count, missing_percent = get_missing_info_1(dataset)
mean_list = []
get_mean_columns_1(data=dataset, col_list = [5,6,7,8,9,10], mean_list=mean_list)

# - Proxy Estimation for mean GHI feature set
# Get list of index position of missing 'Mean GHI' observations
idx_list = dataset[dataset.iloc[:,5].isnull()].iloc[:,5]
    
# Apply proxy estimation on 'Mean GHI' column set
for i in range(0, len(idx_list)):
    idx = idx_list.index[i]
    proxy_estimation(idx, dataset)
    
# Get information about missing observations and mean feature values in dataset after applying proxy estimation
missing_count, missing_percent = get_missing_info_1(dataset)
mean_list = []
get_mean_columns_1(data=dataset, col_list = [5,6,7,8,9,10], mean_list=mean_list)
    
# - Climatological Estimation
col_of_interest = [5]
for col in col_of_interest:
    # Get list of index position of missing 'Mean GHI' observations
    print("Column #", col)
    idx_list = dataset[dataset.iloc[:,col].isnull()].iloc[:,col]
    
    # Apply climatological estimation onto current feature column in dataset
    print("Applying climatological estimation...")
    for i in range(0, len(idx_list)):
        idx = idx_list.index[i]
        climatological_estimation(idx, col, dataset)       
    print("...climatological estimation ended.")
        
# Get information about missing observations and mean feature values in dataset after applying climatological estimation
missing_count, missing_percent = get_missing_info_1(dataset)
mean_list = []
get_mean_columns_1(data=dataset, col_list = [5,6,7,8,9,10], mean_list=mean_list)

# 3. Write dataset into .csv file
dataset.to_csv("Solar Data.csv", sep=',')
