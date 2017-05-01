from yahoo_finance import Share
from pprint import pprint
import datetime
from dateutil import parser
import csv
import numpy as np

# the stocks you want to consider
ticker = 'YHOO'

# Start and End dates based on previous parameters and choice of end_date
numdays = 1000
end_date = parser.parse('2014-04-23')
start_date = end_date - datetime.timedelta(days=numdays)
start_date = str(start_date.date())
end_date = str(end_date.date())
# print("Start: " + start_date, "End: " + end_date)


date_range = [start_date, end_date]
curr_share = Share(ticker)
historical_data = curr_share.get_historical(date_range[0], date_range[1])

high_data = [float(day['High']) for day in historical_data]
low_data = [float(day['Low']) for day in historical_data]
close_data = [float(day['Close']) for day in historical_data]
open_data = [float(day['Open']) for day in historical_data]
date_data = [str(day['Date']) for day in historical_data]

stock_daterange = zip(high_data, low_data, close_data, open_data, date_data)
stock_daterange = sorted(stock_daterange, key=lambda x: x[4])

window = 10 # the tenth day will be the prediction

samples = []
for i in range(len(stock_daterange)):
	if i + 10 >= len(stock_daterange):
		break
	else:
		data_window = stock_daterange[i:i+window-1]
		unzipped = zip(*data_window)
		high = unzipped[0]
		low = unzipped[1]
		close = unzipped[2]
		openp = unzipped[3]
		training = (high, low, close, openp)
		prediction = stock_daterange[i+window]
		# print("Training")
		# print(training)
		# print("Testing")
		# print(prediction)
		# first 36 are for input (sections of 9); next 4 output values
		samples.append((training, prediction))

matrix = []
for row in samples:
	entry = []
	training_in = row[0]
	training_out = row[1]
	for tup in training_in:
		for i in range(len(tup)):
			entry.append(tup[i])
		entry.append(-1)
	for val in training_out:
		entry.append(val)
	entry.append(-1)
	matrix.append(entry)
		

with open('samples.csv', 'wb') as csvfile:
	writer = csv.writer(csvfile)
	for row in matrix:
		writer.writerow(row)

			
