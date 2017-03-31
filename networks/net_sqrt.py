from sklearn.neural_network import MLPRegressor
from math import *
import numpy as np


# Method for splitting list of data into smaller lists with roughly even numbers
def split_data(data, num_nets):
    for i in range(0, len(data), num_nets):
        yield data[i:i + num_nets]

def detectOverlap(training_input_sets):
    # Try to find average overlap using Jaccard index

    # Setup similarity matrix
    dist_values = [[0 for i in range(len(training_input_sets))] for j in range(len(training_input_sets))]

    # Iterate over pairs, not including i == j, and calculate Jaccard index
    for i in range(len(training_input_sets)):
        for j in range(len(training_input_sets)):
            if i <= j:
                continue
            else:
                # Cast to sets and find Jaccard values
                datai = [x[0] for x in training_input_sets[i]]
                dataj = [x[0] for x in training_input_sets[j]]
                if len(datai) < len(dataj):
                    datai = datai + ([0]*(len(dataj) - len(datai)))
                elif len(datai) > len(dataj):
                    dataj = dataj + ([0]*(len(datai) - len(dataj)))

                squares = [(datai[x] - dataj[x])**2 for x in range(len(datai))]
                dist = sqrt(np.sum(squares))
                dist_values[i][j] = dist

    # Calculate statistics
    flattened_list = [entry for sublist in dist_values for entry in sublist]
    overlap_mean = np.average(flattened_list)
    overlap_variance = np.var(flattened_list)

    return (overlap_mean, overlap_variance)

def runSingle(training_input, training_output, testing_input, testing_output):

    # Instantiate single net
    single_net = MLPRegressor()

    # Train single net
    single_net.fit(training_input, training_output)

    # Test single net
    single_net_prediction = single_net.predict(testing_input)

    # Check accuracy of single net through MSE calculation
    single_net_MSE = float((np.sum([(single_net_prediction[i] - testing_output[i])**2 for i in range(len(testing_input))]))/len(testing_input))

    return single_net_MSE

def runMultiple(training_input_sets, training_output_sets, testing_input, testing_output):

    # Instantiate multiple nets
    multi_nets = [MLPRegressor() for i in range(len(training_input_sets))]

    # Train nets
    multi_nets = [multi_nets[i].fit(training_input_sets[i], training_output_sets[i]) for i in range(len(training_input_sets))]

    # Test nets
    multi_nets_predictions = [net.predict(testing_input) for net in multi_nets]

    # Check accuracy of single net through MSE calculation
    multi_nets_MSE = float(np.sum([(np.average([multi_nets_predictions[j][i] for j in range(len(multi_nets))]) - testing_output[i])**2 for i in range(len(testing_output))])/len(testing_input))

    return multi_nets_MSE

def run(size, breadth, num_nets):
    # size indicates the number of data points to use in the experiment
    # breadth indicates the range of values, from 0 to breadth
    # num_nets indicates how many nets to include in the multiple net comparison

    # Define the data sets
    rand = np.random.uniform
    training_input = [[rand(0.0, breadth)] for i in range(size)]
    testing_input = [[rand(0.0, breadth)] for i in range(size)]

    # Compute output, may switch between functions
    # training_output = [sqrt(datum[0]) for datum in training_input]
    # testing_output = [sqrt(datum[0]) for datum in testing_input]

    training_output = [-sqrt(datum[0])+datum[0]**1.5 for datum in training_input]
    testing_output = [-sqrt(datum[0])+datum[0]**1.5 for datum in testing_input]

    # Using 2d polynomial with second variable
    # mu, sigma = breadth/2, breadth/5
    # sqrt_output_testing = [sqrt(datum[0]) for datum in testing_input]
    # extra_set_testing = np.random.normal(mu, sigma, size)
    # testing_output = [sqrt_output_testing[i]*extra_set_testing[i] for i in range(size)]
    # sqrt_output_training = [sqrt(datum[0]) for datum in training_input]
    # extra_set_training = np.random.normal(mu, sigma, size)
    # training_output = [sqrt_output_training[i]*extra_set_training[i] for i in range(size)]



    # Define global data statistics
    training_input_mean, testing_input_mean = np.average(training_input), np.average(testing_input)
    training_input_variance, testing_input_variance = np.var(training_input), np.var(testing_input)

    # Run a single net with all of the data
    single_net_MSE = runSingle(training_input, training_output, testing_input, testing_output)

    # Run multiple nets with divided data sets

    # First divide data evenly
    training_input_sets = list(split_data(training_input, size/num_nets))
    training_output_sets = list(split_data(training_output, size/num_nets))

    # Define statistics of split data
    training_input_sets_overlap_data = detectOverlap(training_input_sets)

    # Run multiple nets with split data
    multi_net_MSE = runMultiple(training_input_sets, training_output_sets, testing_input, testing_output)

    data_statistics = (training_input_mean, training_input_variance, testing_input_mean, testing_input_variance)
    statistic_set = (data_statistics, training_input_sets_overlap_data)

    return (single_net_MSE, multi_net_MSE, statistic_set)
