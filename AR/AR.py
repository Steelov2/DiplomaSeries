from math import sqrt

import pandas as pd
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm


data = pd.read_csv('AAPL.csv', usecols=['Date', 'Close'])
data.set_index('Date', inplace=True)
print(data.head())
print(data.tail())
# plot the raw data
data.plot()
pyplot.show()


# seaonal decompose
def seasonal_decompose(df):
    decomposition = sm.tsa.seasonal_decompose(df, model='additive', period=25)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    fig = decomposition.plot()
    fig.set_size_inches(14, 7)
    for ax in fig.axes:
        ax.set_xticklabels([])
    pyplot.show()
    return trend, seasonal, residual


seasonal_decompose(data.tail(75))
# plot lag plot
series = pd.Series(data["Close"])
lag_plot(series)
pyplot.show()

# plot autocorrelation
autocorrelation_plot(series)
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
pyplot.plot(data.index[-7:], test)
pyplot.xlabel('Date')
pyplot.ylabel('Value')
pyplot.show()
pyplot.gcf().autofmt_xdate()

pyplot.plot(data.index[-7:], predictions, color='red')
pyplot.xlabel('Date')
pyplot.ylabel('Value')
pyplot.show()
pyplot.gcf().autofmt_xdate()

# plot forecasts against actual outcomes
pyplot.plot(data.index[-7:], test, label='Actual')
pyplot.plot(data.index[-7:], predictions, color='red', label='Predicted')
pyplot.legend()
pyplot.show()
pyplot.gcf().autofmt_xdate()

# plot forecasts against actual outcomes
pyplot.plot(data.index[-21:-7], data['Close'][-21:-7], label='Actual')
pyplot.plot(data.index[-7:], predictions, color='red', label='Predicted')
pyplot.legend()
pyplot.xlabel('Date')
pyplot.ylabel('Close')
pyplot.xticks(rotation=-45)

pyplot.show()
pyplot.gcf().autofmt_xdate()
