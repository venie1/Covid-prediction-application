# Covid-prediction-application
 data analysis and forecasting project for COVID-19 cases and deaths :
     Data Preparation:
        necessary libraries like Pandas, NumPy, Plotly, Scikit-learn, and Prophet.
        It loads and preprocesses COVID-19 data.

    Time Series Data Analysis:
        The code calculates cumulative cases and deaths over time and computes the "Days Since" the start of the data.
        It splits the data into training and validation sets.

    Time Series Forecasting:
        The code applies various time series forecasting models to predict future COVID-19 cases.
        Models used include Polynomial Regression, Support Vector Machine (SVM), Holt's Linear Model, Holt's Winter Model, AR Model, MA Model, ARIMA Model, SARIMA Model, and Prophet Model.
        For each model, it trains the model on the training data and evaluates its performance on the validation data using Root Mean Squared Error (RMSE).

    Visualization:
        The code generates interactive plots using Plotly to visualize the training, validation, and predicted data for each forecasting model.
        It also plots the components of the Prophet model (trend, weekly seasonality, etc.).

    Model Comparison:
        The code calculates and displays the RMSE for each forecasting model to compare their performance.

    Repetition for Multiple Countries:
        The code repeats the forecasting process for different countries by grouping the data for each country and following similar steps.

There is also a pickle model  , it  sets up a web application that allows users to upload COVID-19 data, select a prediction model, and get visualizations of the predictions for future days based on the chosen model. It serves as an interface for interacting with the trained time series forecasting models.
