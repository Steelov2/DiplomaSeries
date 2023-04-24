import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')

# read data
data = pd.read_csv('E:\четвертый курс\Дипломка\DiplomaSeries\AR\AAPL.csv', usecols=['Date', 'Close'])
# infer the frequency of the data

# convert Date column to datetime object
data['Date'] = pd.to_datetime(data['Date'])

# set Date column as index
data.set_index('Date', inplace=True)
data.plot()
plt.title('Stock prices', fontsize=20)
plt.ylabel('Close price')
plt.show()


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


result = adfuller(data)
# plot seasonal decomposition
decomposition = sm.tsa.seasonal_decompose(data, model='additive', period=25)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

fig = decomposition.plot()
fig.set_size_inches(14, 7)
plt.show()

# print test statistics
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

train_data = data.iloc[:-7, :]
test_data = data.iloc[-7:, :]
print(train_data.tail())
print(test_data)
# plot ACF and PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(data.Close.diff().dropna(), ax=ax1)
plot_pacf(data.Close.diff().dropna(), ax=ax2)
plt.show()

# Original Series
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(data.Close)
ax1.set_title('Original Series')
ax1.axes.xaxis.set_visible(False)
# 1st Differencing
ax2.plot(data.Close.diff())
ax2.set_title('1st Order Differencing')
ax2.axes.xaxis.set_visible(False)
# 2nd Differencing
ax3.plot(data.Close.diff().diff())
ax3.set_title('2nd Order Differencing')
plt.show()

# initialize variables for storing best model and minimum RMSE
best_model = None
best_rmse = np.inf

# loop over different values of p and q to find the best model
for p in range(1, 25):
    for q in range(1, 25):
        for d in range(1, 3):
            # fit ARIMA model
            model = ARIMA(data['Close'], order=(p, d, q))
            model_fit = model.fit()

            # calculate RMSE for test data
            test_preds = model_fit.predict(start=test_data.index[0], end=test_data.index[-1])
            rmse_val = rmse(test_preds, test_data['Close'])

            # update best model and minimum RMSE
            if rmse_val < best_rmse:
                best_model = model_fit
                best_rmse = rmse_val

            print(f"ARIMA({p}, {d}, {q}) - RMSE: {rmse_val:.2f}")

# print summary of best model
print("\nBest Model:")
print(best_model.summary())

# plot predictions against actual values for test data
test_preds = best_model.predict(start=test_data.index[0], end=test_data.index[-1])
plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_data['Close'], label='Actual')
plt.plot(test_data.index, test_preds, label='Predicted')
plt.title('ARIMA Model - Test Data', fontsize=20)
plt.xlabel('Date')
plt.ylabel('Close')
plt.show()
# plot predictions against actual values for test data

plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data['Close'], label='Train Data')
plt.plot(test_preds.index, test_preds, label='Predicted')
plt.plot(test_data.index, test_data['Close'], label='Test Data')
plt.title('ARIMA Model - Test Data', fontsize=20)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

for i in range(len(test_preds)):
    print(f"predicted={test_preds[i]:.6f}, expected={test_data['Close'][i]:.6f}")
rmse_val = rmse(test_preds, test_data['Close'])
efficiency = (1 - (rmse_val / test_data['Close'].mean())) * 100
print("RMSE = {:.2f}, Efficiency = {:.2f}%".format(rmse_val, efficiency))
