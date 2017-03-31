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

    # Instantiate Nets
    single_net = MLPRegressor()
    multiple_nets = [MLPRegressor() for i in range(num_nets)]

    # Train Nets
    single_net.fit(training_input, training_output)
    multiple_nets = [multiple_nets[i].fit(training_input, training_output) for i in range(num_nets)]

    # Test Nets
    single_prediction = single_net.predict(testing_input)
    multiple_predictions = [multiple_nets[i].predict(testing_input) for i in range(num_nets)]

    # Checking Accuracy (Print Out)
    # for i in range(size):
    #     print(testing_output[i], single_prediction[i])
    #     print("////////////////////////////////////")
    #     for j in range(num_nets):
    #         print(testing_output[i], multiple_predictions[j][i])
    #     print("####################################")

    # Checking Accuracy (Mean Squared Error)
    # single_net_error = (np.sum([(single_prediction[i] - testing_output[i])**2 for i in range(size)]))/size
    # print("Single Net MSE: ",single_net_error)
    # multiple_nets_errors = []
    # for i in range(num_nets):
    #     error = (np.sum([(multiple_predictions[i][j] - testing_output[j])**2 for j in range(size)]))/size
    #     multiple_nets_errors.append(error)
    #     print("Other Net MSE: ", error)

    # Checking Accuracy (Mean Square Error - Median Composite Estimate)
    single_net_error = (np.sum([(single_prediction[i] - testing_output[i])**2 for i in range(size)]))/size
    print("Single Net MSE: ",single_net_error)
    multiple_nets_error = (np.sum([ (np.average([ multiple_predictions[j][i] for j in range(num_nets) ]) - testing_output[i])**2 for i in range(size)]))/size
    print("Multiple Net MSE: ", multiple_nets_error)

    return (single_net_error, multiple_nets_error)
