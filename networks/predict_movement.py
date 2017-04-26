import csv
import numpy as np
from sklearn.neural_network import MLPRegressor
import random

reader = csv.reader(open('moving_averages.csv'))

counter = 0
keys = []
columns = {}
for row in reader:
	if counter == 0:
		keys = row
		for key in keys:
			columns[key] = []
		counter += 1
	else:
		for i in range(len(keys)):
			columns[keys[i]].append(row[i])

# Set up nets
num_nets = 20
for key in columns.keys():

	nets = [MLPRegressor() for i in range(num_nets)]

	data = columns[key]
	random.shuffle(data)

	training_input = [[eval(row)[0]] for row in data]
	training_output = [[eval(row)[1][0:4]] for row in data]
	print(training_output)
	