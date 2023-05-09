from math import sqrt

import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot, pyplot as plt
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg

data = pd.read_csv('E:\четвертый курс\Дипломка\DiplomaSeries\AR\AAPL.csv', usecols=['Date', 'Close'])
# Null values check
if data.isnull().values.any():
    print("Null values are in the file")
    # Null values interpolation
    data = data.interpolate()
else:
    print("No null values are in the file")

# deleting duplicating values
data = data.drop_duplicates()
# convert Date column to datetime object
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

print(data.head())
print(data.tail())
# plot the raw data
data.plot()
pyplot.title("Apple Inc Stock price")
pyplot.show()

# seaonal decompose
decomposition = sm.tsa.seasonal_decompose(data, model='additive', period=25)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

fig = decomposition.plot()
fig.set_size_inches(14, 7)
for ax in fig.axes:
    ax.set_xticklabels([])
pyplot.show()

# plot lag plot
series = pd.Series(data["Close"])
lag_plot(series)
pyplot.show()

# plot autocorrelation
fig, (ax1, ax2) = pyplot.subplots(2, 1, figsize=(10, 8))
plot_acf(data.Close.diff().dropna(), ax=ax1)
plot_pacf(data.Close.diff().dropna(), ax=ax2)
pyplot.show()

pyplot.show()

# split dataset
X = series.values
train, test = X[1:len(X) - 7], X[len(X) - 7:]
print("train")
print(train)
print("test")
print(test)
# train autoregression
window = 29
model = AutoReg(train, lags=29)
model_fit = model.fit()
coef = model_fit.params

# walk forward over time steps in test
history = train[len(train) - window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length - window, length)]
    yhat = coef[0]
    for d in range(window):
        yhat += coef[d + 1] * lag[window - d - 1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# plot forecasts against actual outcomes
pyplot.plot(data.index[-7:], test, label='Actual')
pyplot.xlabel('Date')
pyplot.ylabel('Value')
pyplot.title('Actual Close')
pyplot.show()
pyplot.gcf().autofmt_xdate()

pyplot.plot(data.index[-7:], predictions, color='red', label='Predicted')
pyplot.xlabel('Date')
pyplot.ylabel('Value')
pyplot.title('Predicted Close')
pyplot.show()
pyplot.gcf().autofmt_xdate()

# plot forecasts against actual outcomes
pyplot.plot(data.index[-7:], test, label='Actual')
pyplot.plot(data.index[-7:], predictions, color='red', label='Predicted')
pyplot.legend()
pyplot.show()
pyplot.gcf().autofmt_xdate()

# plot forecasts against actual outcomes
pyplot.plot(data.index, data['Close'], label='Actual')
pyplot.plot(data.index[-7:], predictions, color='red', label='Predicted')
pyplot.legend()
pyplot.xlabel('Date')
pyplot.ylabel('Close')
pyplot.xticks(rotation=-45)

pyplot.show()
pyplot.gcf().autofmt_xdate()


# Convert test to DataFrame
test_df = pd.DataFrame(test, index=data.index[-len(test):], columns=['Close'])

# Compute RMSE and efficiency
rmse_val = sqrt(mean_squared_error(test_df['Close'], predictions))
efficiency = (1 - (rmse_val / test_df['Close'].mean())) * 100
print("RMSE = {:.2f}, Efficiency = {:.2f}%".format(rmse_val, efficiency))

print(model_fit.summary())