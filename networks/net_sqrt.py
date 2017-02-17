from sklearn.neural_network import MLPRegressor
from math import *
import numpy as np


# Method for splitting list of data into smaller lists with roughly even numbers
def split_data(data, num_nets):
    for i in range(0, len(data), num_nets):
        yield data[i:i + num_nets]

def singleVsMultiple(size, breadth, num_nets, split_factor):
    # Generate Data
    rand = np.random.uniform
    training_input = [[rand(0.0, breadth)] for i in range(size)]
    testing_input = [[rand(0.0, breadth)] for i in range(size)]
    training_output = [sqrt(datum[0]) for datum in training_input]
    testing_output = [sqrt(datum[0]) for datum in testing_input]

    # Clean Up Data for Training
    training_input_sets = list(split_data(training_input, split_factor))
    training_output_sets = list(split_data(training_output, split_factor))

    # Instantiate Nets
    single_net = MLPRegressor()
    multiple_nets = [MLPRegressor() for i in range(num_nets)]
    multiple_nets_fullinfo = [MLPRegressor() for i in range(num_nets)]

    # Train Nets
    single_net.fit(training_input, training_output)
    multiple_nets = [multiple_nets[i].fit(training_input_sets[i], training_output_sets[i]) for i in range(num_nets)]
    multiple_nets_fullinfo = [multiple_nets_fullinfo[i].fit(training_input, training_output) for i in range(num_nets)]

    # Test Nets
    single_prediction = single_net.predict(testing_input)
    multiple_predictions = [multiple_nets[i].predict(testing_input) for i in range(num_nets)]
    multiple_predictions_fullinfo = [multiple_nets_fullinfo[i].predict(testing_input) for i in range(num_nets)]

    # Checking Accuracy (Mean Square Error - Median Composite Estimate)
    single_net_error = (np.sum([(single_prediction[i] - testing_output[i])**2 for i in range(size)]))/size
    multiple_nets_error = (np.sum([ (np.average([ multiple_predictions[j][i] for j in range(num_nets) ]) - testing_output[i])**2 for i in range(size)]))/size
    multiple_nets_error_fullinfo = (np.sum([ (np.average([ multiple_predictions_fullinfo[j][i] for j in range(num_nets) ]) - testing_output[i])**2 for i in range(size)]))/size

    print("1")
    return (single_net_error, multiple_nets_error, multiple_nets_error_fullinfo)


