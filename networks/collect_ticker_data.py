from yahoo_finance import Share
from pprint import pprint
import datetime
from dateutil import parser
import csv
import numpy as np

# the stocks you want to consider
tickers = ['YHOO', 'AAPL']

# Start and End dates based on previous parameters and choice of end_date
numdays = 100
end_date = parser.parse('2014-04-23')
start_date = end_date - datetime.timedelta(days=numdays)
start_date = str(start_date.date())
end_date = str(end_date.date())
print("Start: " + start_date, "End: " + end_date)


date_range = [start_date, end_date]
data_by_stock = {}
for ticker in tickers:
    curr_share = Share(ticker)
    historical_data = curr_share.get_historical(date_range[0], date_range[1])

    high_data = [float(day['High']) for day in historical_data]
    low_data = [float(day['Low']) for day in historical_data]
    close_data = [float(day['Close']) for day in historical_data]
    open_data = [float(day['Open']) for day in historical_data]
    date_data = [str(day['Date']) for day in historical_data]

    stock_daterange = zip(high_data, low_data, close_data, open_data, date_data)
    data_by_stock[ticker] = sorted(stock_daterange, key=lambda x: x[4])

window = 10 # the tenth day will be the prediction
moving_averages_by_stock = {}

for ticker in tickers:
	samples = []
	for i in range(len(data_by_stock[ticker])):
		if i + 10 >= len(data_by_stock[ticker]):
			break
		else:
			data_window = data_by_stock[ticker][i:i+window-1]
			unzipped = zip(*data_window)

			mean_high = np.mean(unzipped[0])
			mean_low = np.mean(unzipped[1])
			mean_close = np.mean(unzipped[2])
			mean_open = np.mean(unzipped[3])

			training = (mean_high, mean_low, mean_close, mean_open)
			prediction = data_by_stock[ticker][i+window]
			samples.append((training, prediction))
	moving_averages_by_stock[ticker] = samples
print(moving_averages_by_stock['YHOO'])

with open('moving_averages.csv', 'wb') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(moving_averages_by_stock.keys())
	writer.writerows(zip(*moving_averages_by_stock.values()))




			
