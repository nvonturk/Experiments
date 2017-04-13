import csv

def calculate_revenue(valuation, signal, alloc_probabilities, payments, distribution):

	# Dont change the conditional distribution -> the agent still believes one thing which means belief of the signal is f(s|v = v')
	EV_1 = sum([prob*(alloc_probabilities[0][index]*valuation - payments[0][index]) for index, prob in enumerate(distribution[valuation - 1])])
	EV_2 = sum([prob*(alloc_probabilities[1][index]*valuation - payments[1][index]) for index, prob in enumerate(distribution[valuation - 1])])
	EV_3 = sum([prob*(alloc_probabilities[2][index]*valuation - payments[2][index]) for index, prob in enumerate(distribution[valuation - 1])])
	EV_4 = sum([prob*(alloc_probabilities[3][index]*valuation - payments[3][index]) for index, prob in enumerate(distribution[valuation - 1])])
	EV_5 = sum([prob*(alloc_probabilities[4][index]*valuation - payments[4][index]) for index, prob in enumerate(distribution[valuation - 1])])

	options = [EV_1, EV_2, EV_3, EV_4, EV_5, 0]
	action = options.index(max(options))

	revenue = 0

	if action == 5:
		# Chose to not participate
		print("HEREE")
		revenue = 0
	else:
		revenue = payments[action-1][signal-1]

	return revenue

alloc_probs = [
[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1]
]

payments = [
[-0.014330535,	-0.4298075,	3.40092,	6.7264475,	5.76514],
[-1.7330075,	4.33721675,	-1.772841,	9.9781725,	2.6704925],
[0,	2.570195,	-0.7272775,	9.33615,	3.46576],
[0,	1.454965,	2.7343825,	3.4294375,	10.7592125],
[0,	1.6128725,	1.0250075,	7.035725,	6.397585]
]

valuations = []
signals = []

with open('test.csv', 'rb') as test_file:
	reader = csv.reader(test_file)
	counter = 0
	for row in reader:
		if counter == 0:
			valuations = row
		else:
			signals	= row
		counter = counter + 1
revenue = 0

full_dist = []

with open('full_conditional_dist.csv', 'rb') as full_dist_file:
	reader = csv.reader(full_dist_file)
	for row in reader:
		full_dist.append(row)

# Clean up data to contain ints or floats
valuations = [int(valuation) for valuation in valuations]
signals = [int(signal) for signal in signals]
full_dist = [[float(value) for value in row] for row in full_dist]

revenue = 0

for draw in range(len(valuations)):
	valuation = valuations[draw]
	signal = signals[draw]
	revenue += calculate_revenue(valuation, signal, alloc_probs, payments, full_dist)

print("Average REVENUE: ", float(revenue)/float(len(valuations)))










