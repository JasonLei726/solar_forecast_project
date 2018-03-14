# Solar Forecast Project
A personal project aimed to analyse solar irradiance forecast performance across various machine learning models.

NB: Refer to Solar Irradiance Forecast.pdf file for more details of this project.

# Abstract
Within this study, a comparative model analysis was conducted to evaluate the forecast performance of hourly solar irradiance at Melbourne Airport, Australia. Both weather and solar irradiance data for the location of interest were obtained from the Australian Bureau of Meteorology. Various models were considered such as: (i) Multi-Linear Regression (MLR), (ii) Support Vector Regression (SVR), (iii) Multi-Layer Perceptron (MLP), and (iv) Long Short-Term Memory (LSTM) network. During cross-validation, the SVR and LSTM models attained high predictive performances relative to the MLR model as a baseline when temporal structures (i.e. trends and seasonality) were retained in the dataset. Upon evaluating the model performance on unseen data, the LSTM model was the best performing model as it learnt to recognise temporal patterns throughout the dataset and incorporate it in its predictions. Despite the findings presented in this study, more emphasis should be placed on feature selection, model tuning and optimizing model design to achieve better forecast performance.

# Results
![Figure 1](https://github.com/JasonLei726/solar_forecast_project/blob/master/fig%20-%20forecast%201.png)
Figure 1: Predicted and observed hourly mean solar irradiance (January 2015 to August 2015)

![Figure 2](https://github.com/JasonLei726/solar_forecast_project/blob/master/fig%20-%20forecast%202.png)
Figure 2: Predicted and observed hourly mean solar irradiance (middle of July 2015)

# Scripts
- For data filling:
  - weather_data_preprocess.py : For Australian Bureau of Meteorology's AWS weather data.
  - solar_data_preprocess.py : For Australian Bureau of Meteorology's One Minute Solar data.
  - compile_climate_dataset.py : Combine weather and solar irradiance data into climate dataset.
- For data transformation:
  - feature_selection.py : Backward elimination and inclusion of temporal features into climate dataset.
  - data_transformation.py : Transform dataset for supervised learning problem and normalised.
- For cross-validation:
  - forecast_performance_cv.py : Multiple train-test split for cross validation.
- For final performance tests:
  - forecast_performance_test.py : Evaluate model performance on unseen data.

# Datasets
- Case 5.1 - stationary.csv : Stationary Climate Dataset with all temporal features (feature scaling not applied)
- Case 5.2 - non stationary.csv : Non-stationary Climate Dataset with all temporal features (feature scaling not applied)
