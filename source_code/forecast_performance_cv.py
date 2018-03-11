'''
This script performs a time-series cross-validation of multiple training-test
splits on a loaded supervised learning dataset with applied feature scaling.

Note: The 'perform_timeseries_CV' function is hard coded to only perform
      cross-validation on either: multilinear regression, support vector
      regression, multi-layer perceptron, or long short-term memory model.

This script does the following:
    1. Load supervised learning dataset (either stationary or non-stationary)
    2. Apply feature scaling 
    3. Perform cross-validation over 9 training-test splits
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM


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
    

def display_graph(data, features_of_interest):
    '''
    Function:
        Display multiple line plots of separate feature variables against time
    Arguments:
        data: Sequence of observations as a Pandas DataFrame of Series.
        features_of_interest: Specified features columns in dataset
    '''
    i = 1
    plt.figure()
    for feature in features_of_interest:
        plt.subplot(len(features_of_interest), 1, i)
        plt.plot(data.iloc[:, feature])
        plt.title(data.columns[feature], y=0.5, loc='right')
        i += 1
    plt.show()
    

def create_mlr_model(train_X, train_y):
    '''
    Function:
        Create multi-linear regression (MLR) model and fit to training set.
    Arguments:
        train_X: Training set's feature matrix as Pandas Dataframe.
        train_y: Training set's output vector as Pandas Dataframe.
    Returns:
        sklearn LinearRegression object fitted to the training set.
    '''
    # Design MLR model
    model = LinearRegression()
    
    # Fit MLR model
    model.fit(train_X, train_y)
    return model


def create_svr_model(train_X, train_y):
    '''
    Function:
        Creates support vector regression (SVR) model and fit to training set.
    Arguments:
        train_X: Training set's feature matrix as Pandas Dataframe.
        train_y: Training set's output vector as Pandas Dataframe.
    Returns:
        sklearn SVR object fitted to the training set.
    '''
    # Design SVR model
    model = SVR(kernel='rbf')
    
    # Fit SVR model
    model.fit(train_X, train_y)
    return model


def create_mlp_model(train_X, train_y, n_epoch, n_batch):
    '''
    Function:
        Creates multi-layer perceptron (MLP) model and fit to training set.
    Arguments:
        train_X: Training set's feature matrix as Pandas Dataframe.
        train_y: Training set's output vector as Pandas Dataframe.
        n_epoch: Number of training epochs.
        n_batch: Number of mini batch size.
    Returns:
        keras Sequential object fitted to the training set.
    '''
    # Design MLP network
    model = Sequential()
    model.add(Dense(units=50, input_dim=50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Fit MLP network
    history = model.fit(train_X, train_y, batch_size=n_batch, epochs=n_epoch, shuffle=False)
    return model, history

def create_lstm_model(train_X, valid_X, train_y, valid_y, n_neuron, n_epoch, n_batch):
    '''
    Function:
        Creates long short-term memory (LSTM) model and fit to training set.
    Arguments:
        train_X: Training set's feature matrix as Pandas Dataframe.
        train_y: Training set's output vector as Pandas Dataframe.
        valid_X: Validation set's feature matrix as Pandas Dataframe.
        valid_y: Validation set's output vector as Pandas Dataframe.
        n_neuron: Number of memory units.
        n_epoch: Number of training epochs.
        n_batch: Number of mini batch size.
    Returns:
        keras Sequential object fitted to the training set.
    '''
    # Reshape feature input into 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    valid_X = valid_X.reshape((valid_X.shape[0], 1, valid_X.shape[1]))

    # Design LSTM network
    model = Sequential()
    model.add(LSTM(n_neuron, input_shape=(train_X.shape[1], train_X.shape[2])))   
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit network
    history = model.fit(train_X, train_y, epochs=n_epoch, batch_size=n_batch, validation_data=(valid_X, valid_y), verbose=2, shuffle=False)
    return model,history


def split_dataset(dataset, n_year):
    '''
    Function:
        Split the dataset into its training and validation sets as well as
        its feature matrix (X) and output vector (y). The split is made at
        the defined number of years beyond the beginning of the dataset.

    Arguments:
        dataset: Sequence of observations as a Pandas DataFrame of Series.
        n_year: Number of years from the beginning of the dataset.
    Returns:
        Pandas DataFrames of series of the feature matrix (X) and output
        vector (y) from its training and validation set, respectively.
        
    Note: 
        This function is hard coded for multiple training-test splits for
        cross-validation.
    '''
    # Split dataset into training and test set
    split = 365 * 24 * n_year    
    train = dataset[:split, :]              
    valid = dataset[split:split+8760, :]  
                    
    # Split each set into feature matrix (X) and output vector (y)
    train_X, train_y = train[:, :-1], train[:, -1]
    valid_X, valid_y = valid[:, :-1], valid[:, -1]
    
    return train_X, train_y, valid_X, valid_y


def performance_test(dataset, model, scaler, input_dim, n_fold):
    '''
    Function:
        Compute the predicted output vector using the 'predict' method by
        the 'model' object. Inverse transformation is applied to revert
        both predicted and observed output vectors back to its original scale.
    Arguments:
        dataset: Sequence of observations as a Pandas DataFrame of Series.
        model: Object created via 'create_model' function.
        scaler: sklearn object associated with feature scaling.
        input_dim: Input dimensions that the model object is compatible with where
            'input_dim=2' represent 2D and 'input_dim=3' represent 3D. Any other
            value will cause the function to return None values.
    Returns:
        Pandas DataFrame of the predicted and observed output vectors respectively
    '''
    # Split dataset
    train_X, train_y, test_X, test_y = split_dataset(dataset, n_fold)
    
    # Compute predicted outputs
    if input_dim == 2:
        y_predict = model.predict(test_X)
        y_predict = y_predict.reshape((len(y_predict),1))    
    elif input_dim == 3:
        y_predict = model.predict(test_X.reshape((test_X.shape[0], 1, test_X.shape[1])))        
    else:
        print("Invalid Dimensions for model prediction")
        inv_y_predict, inv_y = None, None
        return inv_y_predict, inv_y
    
    # Invert scaling for predicted output
    inv_y_predict = np.concatenate((test_X, y_predict), axis=1)
    inv_y_predict = scaler.inverse_transform(inv_y_predict)
    inv_y_predict = inv_y_predict[:,-1]

    # Invert scaling for observed output
    inv_y = np.concatenate((test_X, test_y.reshape((len(test_y),1))), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,-1]
    
    return inv_y_predict, inv_y


def performance_evaluate(dataset, y_pred, y):
    '''
    Function:
        Compute performance metrics that includes:
            - mean square error
            - root mean square error
            - mean absolute error
            - mean bias error
            - R-squared
            - adjusted R-squared
    Arguments:
        dataset: Sequence of observations as a Pandas DataFrame of Series.
        y_pred: Predicted output vector as a Pandas DataFrame of Series.
        y: Observed output vector as a Pandas DataFrame of Series.
    Returns:
        List of performance metric values
    '''
    # Performance Metrics
    mbe = sum(y-y_pred)/len(y)
    mse = np.sqrt(mean_squared_error(y, y_pred))
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    adj_r2 = r2*((dataset.shape[0]-1)/(dataset.shape[0]-(dataset.shape[1]-1)-1))
    
    return [mse, rmse, mae, mbe, r2, adj_r2]


def perform_timeseries_CV(dataset, n_folds, algorithm, scaler, n_epoch=50, n_batch=72):
    '''
    Function:
        Perform time series cross validation on the specified learning algorithm and
        evaluate the performance metrics over multiple training-test splits.
    Arguments:
        dataset: Sequence of observations as a Pandas DataFrame of Series.
        n_folds: Number of iterations
        algorithm: String text of learning model's abbreviation.
        scaler: sklearn object associated with feature scaling.
        n_epoch: Number of training epochs
        n_batch: Number of mini batch size
    Returns:
        Pandas DataFrame of series containing performance metrics.
    '''
    result_array = []
    for n in range(1, n_folds):        
        # Perform training-test according to current n-fold
        print("K-Fold: ", n)
        train_X, train_y, valid_X, valid_y = split_dataset(dataset, n)

        # Train model based on passed 'algorithm' argument
        if algorithm == 'LSTM':
            print("Training LSTM model...")
            model, history = create_lstm_model(train_X, valid_X, train_y, valid_y, n_neuron=100, n_epoch=n_epoch, n_batch=n_batch)
            inv_yhat, inv_y = performance_test(dataset, model, scaler, 3, n)
            plt.plot(history.history['loss'], label='train', color='blue')
            plt.plot(history.history['val_loss'], label='test', color='orange')
        elif algorithm == 'MLP':
            print("Training MLP model...")
            model, history = create_mlp_model(train_X, train_y, n_epoch=n_epoch, n_batch=n_batch)
            inv_yhat, inv_y = performance_test(dataset, model, scaler, 2, n)
            plt.plot(history.history['loss'], label='train')
        elif algorithm == 'SVR':
            print("Training SVR model...")
            model = create_svr_model(train_X, train_y)
            inv_yhat, inv_y = performance_test(dataset, model, scaler, 2, n)
        elif algorithm == 'MLR':
            print("Training MLR model...")  
            model = create_mlr_model(train_X, train_y)             
            inv_yhat, inv_y = performance_test(dataset, model, scaler, 2, n)
        else:
            print("No Algorithms Found...")
            print("K-Fold Cross Validation ended")
            break       
        print("Training model complete!")
        
        # Compute performance metrics and store into array
        result = performance_evaluate(dataset, inv_yhat, inv_y)
        result.insert(0, n)
        result_array.append(result)
    
    # Format and print mean performance metric results
    result_array = pd.DataFrame(result_array)
    result_array.columns = ['k-Fold', 'MSE', 'RMSE', 'MAE', 'MBE', 'R2', 'adj R2']
    
    print("Mean MSE: %.3f" % result_array['MSE'].mean())
    print("Mean RMSE: %.3f" % result_array['RMSE'].mean())
    print("Mean MAE: %.3f" % result_array['MAE'].mean())
    print("Mean MBE: %.3f" % result_array['MBE'].mean())
    print("Mean r2: %.3f" % result_array['R2'].mean())
    print("Mean adj r2: %.3f" % result_array['adj R2'].mean())
    
    print("K-Fold Cross Validation has been completed!")
    plt.show()
    return result_array


def persistence_model(data):
    '''
    Function:
        Assuming the dataset is transformed for supervised learning, return
        the predicted and observed output vectors.
    Arguments:
        data: Sequence of observations as a Pandas DataFrame of Series.
    Returns:
        Pandas DataFrame of the predicted and observed output vectors respectively
    '''
    y_actual = data.iloc[:,-1]
    y_predict = data.iloc[:,-2]
    return y_actual, y_predict


def persistence_forecast(data):
    '''
    Function:
        Perform persistence forecast and return its performance metric values.
    Arguments:
        data: Sequence of observations as a Pandas DataFrame of Series.
    Returns:
        List of performance metric values
    '''
    y_actual, y_predict = persistence_model(data)
    return performance_evaluate(data, y_predict, y_actual)

    
# 1. Load supervised learning dataset (either stationary or non-stationary)
dataset = load_dataset("Case 5.1 - stationary.csv")   
dataset = load_dataset("Case 5.2 - non stationary.csv")

# (Optional) Dataset Visualization
col_of_interest = [n for n in range(0, len(dataset.columns))]
display_graph(dataset, col_of_interest)

# 2. Apply feature scaling 
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = scaler.fit_transform(dataset)

# 3. Perform cross-validation over 9 training-test splits
result_mlr = perform_timeseries_CV(dataset, 9, 'MLR', scaler)
result_svr = perform_timeseries_CV(dataset, 9, 'SVR', scaler)
result_mlp = perform_timeseries_CV(dataset, 9, 'MLP', scaler)
result_lstm = perform_timeseries_CV(dataset, 9, 'LSTM', scaler)
