from math import sqrt
from statistics import mean

import matplotlib.dates as mdates
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data = pd.read_csv('E:\четвертый курс\Дипломка\DiplomaSeries\AR\AAPL.csv', usecols=['Date', 'Close'])
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
print(data.head())
print(data.tail())
data.plot()
pyplot.show()



# plot seasonal decomposition
decomposition = sm.tsa.seasonal_decompose(data, model='additive', period=25)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

fig = decomposition.plot()
fig.set_size_inches(14, 7)
pyplot.show()

# Tail-rolling average transform
rolling = data.rolling(window=3)
rolling_mean = rolling.mean()
print(rolling_mean.head(10))
# Plot the original data and the rolling mean against each other
pyplot.plot(data.index, data['Close'], label='Original Data')
pyplot.plot(rolling_mean.index, rolling_mean['Close'], color='red', label='Rolling Mean')
pyplot.legend()  # Add a legend to the plot
pyplot.show()  # Show the plot
# plot autocorrelation
fig, (ax1, ax2) = pyplot.subplots(2, 1, figsize=(10, 8))
plot_acf(data.Close.diff().dropna(), ax=ax1)
plot_pacf(data.Close.diff().dropna(), ax=ax2)
pyplot.show()
# prepare situation
X = data.values
window = 3
history = [X[i] for i in range(window)]
test = [X[i] for i in range(window, len(X))]
predictions = list()
# walk forward over time steps in test
for t in range(len(test)):
    length = len(history)
    yhat = mean([history[i][0] for i in range(length - window, length)])
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red', label='Predicted')
pyplot.title('Stock prices', fontsize=20)
pyplot.ylabel('Close price')
pyplot.xlabel('Date')
pyplot.show()
# zoom plot
pyplot.plot(test[0:100])
pyplot.plot(predictions[0:100], color='red', label='Predicted')
pyplot.show()

# plot original data up to 7 days before the end of the dataset
pyplot.plot(data.index, data['Close'], label='Actual')

# plot predictions starting from 7 days before the end of the dataset
pyplot.plot(data.index[-7:], predictions[-7:], color='red', label='Predicted')

pyplot.legend()
pyplot.xlabel('Date')
pyplot.ylabel('Close')
pyplot.xticks(rotation=-45)

pyplot.show()
# Convert test to DataFrame
test_df = pd.DataFrame(test, index=data.index[-len(test):], columns=['Close'])
rmse_val = sqrt(mean_squared_error(test_df['Close'], predictions))
efficiency = (1 - (rmse_val / test_df['Close'].mean())) * 100
print("RMSE = {:.2f}, Efficiency = {:.2f}%".format(rmse_val, efficiency))
