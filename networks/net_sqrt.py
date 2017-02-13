from sklearn.neural_network import MLPClassifier as MLPC
from math import *
import numpy as np

def experimentSingleNet(training_sizes, iterations, breadth):
    for size in training_sizes:
            rand = np.random.uniform
            training_data = rand(0.0, breadth, size)
            testing_data = rand(0.0, breadth, size)
            performance = []
            for i in range(iterations):
                # TODO: check this trainAndTestSingleNet call
                performance.append(trainAndTestSingleNet(training_data, testing_data))
            print(np.average(performance))

def experimentMultipleNets(num_nets, training_sizes, iterations, breadth):
    for size in training_sizes:
        rand = np.random.uniform
        training_data = rand(0.0, breadth, size)
        testing_data = rand(0.0, breadth, size)
        testing_data = [[datum] for datum in testing_data]

        chunk_size = int(size/num_nets)

        split_training_data = [training_data[i:i+chunk_size] for i in range(0, len(training_data), chunk_size)]

        nets = []
        for i in range(num_nets):
            nets.append(trainSingleNet(split_training_data[i]))

        testing_output = [round(sqrt(testing_data[i][0])) for i in range(len(testing_data))]

        results = []
        for i in range(iterations):
            # Loop through iterations to average out single runs
            predictions = []
            iteration_results = []
            for j in range(num_nets):
                prediction = nets[j].predict(testing_data)
                predictions.append(prediction)
            # validation
            # for loop through the indicies of one prediction vector
            for j in range(len(predictions[0])):
                # for each net
                prediction_set = []
                for k in range(len(predictions)):
                    prediction_set.append(predictions[k][j])
                if np.median(prediction_set) == testing_output[j]:
                    iteration_results.append(1)
                else:
                    iteration_results.append(0)
            results.append(np.average(iteration_results))
        print(np.average(results))



def trainSingleNet(training_data):
    # training_data is the raw list of randomly generated

    training_input = [[round(datum)] for datum in training_data]
    training_output = [round(sqrt(datum[0])) for datum in training_input]

    mlp = MLPC(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    mlp.fit(training_input, training_output);

    return mlp

def testSingleNet(testing_data, mlp):

    test_input = [[round(datum)] for datum in testing_data]
    test_output = [round(sqrt(datum[0])) for datum in test_input]
    results = []
    predictions = mlp.predict(test_input)
    for i in range(len(predictions)):
        if predictions[i] == test_output[i]:
            results.append(1)
        else:
            results.append(0)
    return np.average(results)

def trainAndTestSingleNet(testing_data, training_data):
    return testSingleNet(testing_data, trainSingleNet(training_data))


training_sizes = [10, 100, 1000]
iterations = 10
breadth = 100
num_nets = 5

experimentSingleNet(training_sizes, iterations, breadth)
experimentMultipleNets(num_nets, training_sizes, iterations, breadth)
