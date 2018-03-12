# solar_forecast_project
A personal project aimed to analyse solar irradiance forecast performance across various machine learning models.

Refer to 'Solar Irradiance Forecast' pdf file for more details of this project.

# Abstract
Within this study, a comparative model analysis was conducted to evaluate the forecast performance of hourly solar irradiance at Melbourne Airport, Australia. These data for the location of interest were obtained from the Australian Bureau of Meteorology. Various models were considered such as: (i) Multi-Linear Regression (MLR), (ii) Support Vector Regression (SVR), (iii) Multi-Layer Perceptron (MLP), and (iv) Long Short-Term Memory (LSTM) network. During cross-validation, the SVR and LSTM models attained high predictive performances relative to the MLR model as a baseline when temporal structures (i.e. trends and seasonality) were retained in the dataset. Upon evaluating the model performance on unseen data, the LSTM model was the best performing model as it learnt to recognise temporal patterns throughout the dataset and incorporate it in its predictions. Despite the findings presented in this study, more emphasis should be placed on feature selection, model tuning and optimizing model design to achieve better performance.

# Results
![test](/solar_forecast_project/fig%20-%20forecast%201.png)
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

