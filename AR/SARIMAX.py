import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm

register_matplotlib_converters()
from time import time

# read data
catfish_sales = pd.read_csv('E:\четвертый курс\Дипломка\DiplomaSeries\AR\AAPL.csv', usecols=['Date', 'Close'])
print(catfish_sales.tail())
catfish_sales['Date'] = pd.to_datetime(catfish_sales['Date'])
catfish_sales.set_index('Date', inplace=True)

plt.figure(figsize=(10, 4))
plt.plot(catfish_sales)
plt.title('Stock prices', fontsize=20)
plt.ylabel('Close', fontsize=16)

# plt.show()

#
# def rmse(predictions, targets):
#     return np.sqrt(((predictions - targets) ** 2).mean())


train_data = catfish_sales.iloc[:-7, :]
test_data = catfish_sales.iloc[-7:, :]

print(train_data.tail())
print(test_data)

first_diff = catfish_sales.diff()[1:]
plt.figure(figsize=(10, 4))
plt.plot(first_diff)
plt.title('Stock prices', fontsize=20)
plt.ylabel('Close', fontsize=16)

# plt.show()

acf_vals = acf(first_diff)
num_lags = 20
plt.bar(range(num_lags), acf_vals[:num_lags])
plt.show()

pacf_vals = pacf(first_diff)
num_lags = 15
plt.bar(range(num_lags), pacf_vals[:num_lags])
plt.show()

train_end = datetime(2023, 4, 12)
test_end = datetime(2023, 4, 21)

# my_order = (0, 1, 0)
# my_seasonal_order = (1, 0, 1, 12)
# # define model
# model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)
# # fit the model
# start = time()
# model_fit = model.fit()
# end = time()
# print('Model Fitting Time:', end - start)
# print(model_fit.summary())
#
# # get the predictions and residuals
# predictions = model_fit.forecast(len(test_data))
# predictions = pd.Series(predictions, index=test_data.index)
# residuals = test_data - predictions
# print(predictions.dropna())
# plt.figure(figsize=(10, 4))
#
# plt.plot(catfish_sales)
# plt.plot(predictions)
#
# plt.legend(('Data', 'Predictions'), fontsize=16)
#
# plt.title('Catfish Sales in 1000s of Pounds', fontsize=20)
# plt.ylabel('Production', fontsize=16)
#
# plt.show()

model = sm.tsa.statespace.SARIMAX(catfish_sales['Close'], order=(11, 2, 11), seasonal_order=(1, 1, 1, 12))
results = model.fit()
print(results.summary())

catfish_sales['forecast'] = results.predict(start=748, end=755, dynamic=True)
catfish_sales[['Close', 'forecast']].plot(figsize=(12, 8))
plt.show()

# calculate RMSE
rmse = np.sqrt(mean_squared_error(catfish_sales['Close'].iloc[748:755], catfish_sales['forecast'].iloc[748:755]))

# print actual and predicted values
print('Actual\tPredicted')
for i in range(748, 755):
    print(f'{catfish_sales.iloc[i, 0]:.2f}\t{catfish_sales.iloc[i, 1]:.2f}')

# print efficiency score
print(f'Efficiency: {100 * (1 - rmse / np.mean(catfish_sales["Close"].iloc[748:755])):.2f}%')
print(rmse)
# plot actual vs predicted
catfish_sales[['Close', 'forecast']].plot(figsize=(12, 8))
plt.show()
# автоарима/сарима
# from pmdarima.arima import auto_arima
#
# # fit auto-ARIMA model
# model = auto_arima(catfish_sales['Close'], seasonal=True, m=12, trace=True)
#
# # print model summary
# print(model.summary())
#
# # make predictions
# forecast = model.predict(n_periods=7)
#
# # calculate RMSE
# rmse = np.sqrt(mean_squared_error(test_data['Close'], forecast))
#
# # print efficiency score
# print(f'Efficiency: {100 * (1 - rmse / np.mean(test_data["Close"])):.2f}%')
# print(rmse)
#
# # plot actual vs predicted
# plt.figure(figsize=(12, 8))
# plt.plot(train_data.index, catfish_sales['Close'], label='Actual Price')
# plt.plot(test_data.index, forecast, label='Forecast')
# plt.legend(loc='best')
# plt.show()
#
# for i in range(len(forecast)):
#     print(f"predicted={forecast[i]:.6f}, expected={catfish_sales['Close'][i]:.6f}")
# rmse_val = rmse(forecast, catfish_sales['Close'])
# efficiency = (1 - (rmse_val / test_data['Close'].mean())) * 100
# print("RMSE = {:.2f}, Efficiency = {:.2f}%".format(rmse_val, efficiency))