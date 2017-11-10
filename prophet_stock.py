# Prophet example

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fbprophet import Prophet

# Reading the Stock market data
stock_market_data = pd.read_csv('Google_Stock_Price_Train.csv', parse_dates = True, usecols = [0, 1])
stock_market_data.plot()

# Changing the input type for prophet - (df, y)
stock_market_data = stock_market_data.rename(columns = {'Date':'ds', 'Open':'y'})
df = stock_market_data
df['y'] = np.log(df['y'])

# Fitting the data for Prophet
model = Prophet() 
model.fit(df)

# Making predictions using on future
future = model.make_future_dataframe(periods = 0)
forecast = model.predict(future)

# Ploting the figure
figure = model.plot(forecast)

df['y'] = np.exp(df['y'])
stock_market_data['ds'] = pd.to_datetime(stock_market_data['ds'])

# Merging test data and prediction on test
two_years = pd.merge(forecast, df, on = 'ds')

# 
two_years['yhat'] = np.exp(two_years['yhat'])
two_years['yhat_lower'] = np.exp(two_years['yhat_lower'])
two_years['yhat_upper'] = np.exp(two_years['yhat_upper'])
two_years[['y', 'yhat']].plot()

future1 = model.make_future_dataframe(periods = 365)
forecast1 = model.predict(future1)

figure = model.plot(forecast1)

# Fetching only the predicted values
forecast_test = forecast1[ forecast1['ds'] > '2016-12-31']
forecast_test['yhat'].plot()

# Reading the Test data
real_stock_data = pd.read_csv('Google_Stock_Price_Test.csv', parse_dates = True, usecols = [0, 1])
real_stock_data = real_stock_data.rename(columns = {'Date':'ds', 'Open':'y'})
real_stock_data['ds'] = pd.to_datetime(real_stock_data['ds'])

# Joining predicted and test data
predicted_test = pd.merge(forecast_test, real_stock_data, on = 'ds')
predicted_test['yhat'] = np.exp(predicted_test['yhat'])
predicted_test[['y', 'yhat']].plot()