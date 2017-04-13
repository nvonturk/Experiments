import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import csv

# Declare data array
valuations = []
signals = []

# Open file and parse
f = open('HW3_data.txt', 'r')
for line in f:
    val_arry = line.split(',')
    first_val = int(val_arry[0])
    second_val = int(val_arry[1][0])
    valuations.append(first_val)
    signals.append(second_val)

# Shuffle data points
zipped_data = zip(valuations, signals)
random.shuffle(zipped_data)
[valuations, signals] = [list(t) for t in zip(*zipped_data)]

# Split data into two sets: training (70%) and testing (%30)
num_training = int(round(0.7*len(valuations)))

training_vals = valuations[:num_training]
training_sigs = signals[:num_training]

# Write these values to a file for later
testing_vals = valuations[num_training:]
testing_sigs = signals[num_training:]

with open('test.csv', 'wb') as test_file:
    wr = csv.writer(test_file)
    wr.writerow(testing_vals)
    wr.writerow(testing_sigs)

# Fill frequency matrix
frequency_matrix = [[0 for i in range(5)] for j in range(5)]
for i in range(len(training_vals)):
    frequency_matrix[training_vals[i]-1][training_sigs[i]-1] += 1

# Normalize by total frequency
total = 0
for i in range(5):
    for j in range(5):
        total += frequency_matrix[i][j]

for i in range(5):
    for j in range(5):
        frequency_matrix[i][j] = float(frequency_matrix[i][j])/float(total)

# Check for appropriate pdf
# print(sum(sum(l) for l in frequency_matrix))

# Get distribution values for .mod file
print("/////////////////////////////////////////")
print("/////////////////////////////////////////")
print("/////////////////////////////////////////")
print("prior probs")
counter = 1
for row in frequency_matrix:
    out_string = "v" + str(counter)
    for col in row:
        out_string = out_string + "   " + str(col)
    counter = counter + 1
    print(out_string)



# Get conditionals for .mod file
conditionals = []
for i in range(len(frequency_matrix)):
    row = frequency_matrix[i]
    conditional = [float(col)/float(sum(row)) for col in row]
    conditionals.append(conditional)

# print(np.matrix(frequency_matrix))
print("conditional probs")
print("-----------------------------------------")
print("-----------------------------------------")

counter = 1
for row in conditionals:
    out_string = "v" + str(counter)
    for col in row:
        out_string = out_string + "   " + str(col)
    counter = counter + 1
    print(out_string)
# print(np.matrix(conditionals))





# Save the full marginal distribution
full_frequency_matrix = [[0 for i in range(5)] for j in range(5)]
for i in range(len(valuations)):
    full_frequency_matrix[valuations[i]-1][signals[i]-1] += 1

# Normalize by total frequency
full_total = 0
for i in range(5):
    for j in range(5):
        full_total += full_frequency_matrix[i][j]

for i in range(5):
    for j in range(5):
        full_frequency_matrix[i][j] = float(full_frequency_matrix[i][j])/float(full_total)

full_conditionals = []
for i in range(len(full_frequency_matrix)):
    row = full_frequency_matrix[i]
    conditional = [float(col)/float(sum(row)) for col in row]
    full_conditionals.append(conditional)

with open('full_conditional_dist.csv', 'wb') as full_cond_dist_file:
    wr = csv.writer(full_cond_dist_file)
    for row in full_conditionals:
        wr.writerow(row)







# Calculate the covariance matrix
# cov = np.cov(np.vstack([valuations, signals]))

# # Plot data
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# # Iterate over valuations
# for c, z in zip(['r', 'g', 'b', 'y', 'c'], [1, 2, 3, 4,5]):
#     xs = [1,2,3,4,5]
#     ys = frequency_matrix[z-1]

#     cs = [c] * len(xs)
#     ax.bar(xs, ys, zs=z, zdir='y', color=cs)

# plt.xlabel('Valuation')
# plt.ylabel('Signal')
# plt.show()
