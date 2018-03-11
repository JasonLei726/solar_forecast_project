'''
This script performs a time-series performance evaluation on unseen data
for models trained on a supervised learning dataset with applied feature
scaling.

Note: This script is hard coded for the intended project that takes the
      first 9 years of the dataset for model training and the rest for
      final performance evaluation.


The script does the following:
    1. Load and apply feature scaling to supervised learning dataset
    2. Create models
    3. Predict forecast
    4. Performance evaluation
    5. Visualise results
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
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


def split_dataset(dataset):
    '''
    Function:
        Split the dataset into its training and validation sets as well as
        its feature matrix (X) and output vector (y). 
    Arguments:
        dataset: Sequence of observations as a Pandas DataFrame of Series.
    Returns:
        Pandas DataFrames of series of the feature matrix (X) and output
        vector (y) from its training and validation set, respectively.
        
    Note: 
        This function is hard coded for a specific training-test split.
    '''
    # Split dataset into training and test set
    split = 365 * 24 * 9    
    train = dataset[:split, :]    # Training set - Observations up to 31 Dec 2014               
    test = dataset[split:, :]     # Test set - Observations from 1 Jan 2015 and onwards
                    
    # Split each set into feature matrix (X) and output vector (y)
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    
    return train_X, train_y, test_X, test_y


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


def create_lstm_model(train_X, test_X, train_y, test_y, n_neuron, n_epoch, n_batch):
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
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # Design LSTM network
    model = Sequential()
    model.add(LSTM(n_neuron, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit network
    model.fit(train_X, train_y, epochs=n_epoch, batch_size=n_batch, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    return model
    

def create_model(dataset, algorithm):
    '''
    Function:
        Creates either an 'MLR', 'SVR', or 'LSTM' model object based on the 
        specified 'algorithm'.        
    Arguments:
        dataset: Sequence of observations as a Pandas DataFrame of Series.
        algorithm: String text of learning model's abbreviation.
    Returns:
        Either an sklearn or keras object fitted to the training set.
    '''
    
    # Split dataset
    train_X, train_y, test_X, test_y = split_dataset(dataset)
    
    # Create model based on 'algorithm' argument
    if algorithm == 'MLR':
        print("Training MLR Model.")
        model = create_mlr_model(train_X, train_y)
    elif algorithm == 'SVR':
        print("Training SVR Model.")
        model = create_svr_model(train_X, train_y)
    elif algorithm == 'LSTM':
        print("Training LSTM Model.")
        model = create_lstm_model(train_X, test_X, train_y, test_y, n_neuron=50, n_epoch=50, n_batch=72)
    else:
        print("Algorithm does not exist in this function")
        
    print("Model Training complete.")
    return model
    

def performance_test(dataset, model, scaler, input_dim):
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
    train_X, train_y, test_X, test_y = split_dataset(dataset)
    
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

    # Invert scaling for actual output
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
    
    result = pd.DataFrame([mse, rmse, mae, mbe, r2, adj_r2]).T
    result.columns = ['MSE', 'RMSE', 'MAE', 'MBE', 'R2', 'adj R2']
    
    return result
    

# 1. Load and apply feature scaling to non-stationary supervised learning dataset
dataset = load_dataset("Case 5.2 - non stationary.csv")
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = scaler.fit_transform(dataset)

# 2. Create Models
mlr_model = create_model(dataset, 'MLR')
svr_model = create_model(dataset, 'SVR')
lstm_model = create_model(dataset, 'LSTM')

# 3. Predict forecast
y_pred_mlr, y_mlr = performance_test(dataset, mlr_model, scaler, 2)
y_pred_svr, y_svr = performance_test(dataset, svr_model, scaler, 2)
y_pred_lstm, y_lstm = performance_test(dataset, lstm_model, scaler, 3)

# 4. Evaluate performance
mlr_result = performance_evaluate(dataset, y_pred_mlr, y_mlr)
svr_result = performance_evaluate(dataset, y_pred_svr, y_svr)
lstm_result = performance_evaluate(dataset, y_pred_lstm, y_lstm)

# 5. Visualise Results
plt.plot(y_mlr, label='actual', color='blue')
plt.plot(y_pred_mlr, label='predicted (MLR)', color='red')
plt.plot(y_pred_svr, label='predicted (SVR)', color='orange')
plt.plot(y_pred_lstm, label='predicted (LSTM)', color='green')
plt.legend()
plt.show()
