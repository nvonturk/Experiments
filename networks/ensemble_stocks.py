from yahoo_finance import Share
from pprint import pprint
import datetime
from dateutil import parser

memory_parameter = 10 # this includes the training history
numpoints_parameter = 10 # how many training examples should we have per net

# Total number of training data points for one single net =
#   total_points_for_training_single_net = (memory_parameter + 1)*(numpoints_parameter)
#   -> total number days needed for single net

# How many nets do you want? -> numnets_parameter
numnets_parameter = 10 # how many nets will be used to forecast next day value

# total number of days = (memory_parameter + 1)*(numpoints_parameter)*(numnets_parameter)
#   this assumes no overlap in training

# the stocks you want to consider
tickers = ['YHOO']

# Start and End dates based on previous parameters and choice of end_date
numdays = (memory_parameter + 1)*(numnets_parameter)*(numpoints_parameter)
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
    data_by_stock[ticker] = stock_daterange

pprint(data_by_stock['YHOO'])

# Have individual stock data sets over ranges of data_by_stock
