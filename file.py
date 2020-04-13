from copy import deepcopy

import pandas as pd
import time
import matplotlib.pyplot as plot
import random
import numpy


def train_data(features, y, weight, training_rate, max_iteration, cost):
    while cost != 0 and max_iteration <= 1000:
        c = float(0),
        error = 0
        error_sum = 0
        for index, sample in enumerate(features):
            yHat = numpy.matmul(weight, sample)
            yHat = yHat.tolist()
            diff = y[index] - yHat
            error = diff ** 2
            error_sum = error_sum +error
            c = c + 0.5 * error
            sample = numpy.reshape(sample, (1, 5))
            for k in range(len(weight)):
                weight[k] = weight[k] - training_rate * (-1) * diff * sample[0][k]
        cost = c
        if max_iteration % 100 == 0:
            plot.scatter(max_iteration,cost)
            plot.axis([0,1000, 0, 100])
            plot.xlabel('Iterations')
            plot.ylabel('Cost')
            plot.title('Cost and Iteration Evaluation!')
            plot.show()
            print("Iteration:",max_iteration,"       Cost:", cost, "       Error: ",error)
        max_iteration = max_iteration + 1
    print("Data Set is Trained....")
    return weight


def test_data(features, actual, weight):
    mismatch = 0
    mismatch2 = 0
    for index, sample in enumerate(features):
        yHat = numpy.matmul(weight, sample)
        yHat = yHat.tolist()
        if yHat != actual[index]:
            mismatch = mismatch + 1
        if yHat <= 1.5:
            yHat = 1
        if yHat > 1.5 and yHat <= 2.5:
            yHat = 2
        if yHat > 2.5:
            yHat = 3
        if yHat != actual[index]:
            mismatch2 = mismatch2 + 1
    print("Mismatch Without Step Function = ", mismatch)
    print("Mismatch With Step Function = ", mismatch2)


def main():
    data_set = pd.read_excel(r"iris.xlsx")
    data_set = data_set.replace(to_replace=["I. setosa","I. versicolor","I. virginica"],value=[1,2,3])
    data_set.insert(0, "Bias", [1 for i in range(len(data_set))], True)
    data_set = data_set.sample(frac=1)  # shuffling
    data_set = data_set.to_numpy()  # converting DataFrame to Numpy array

    data_train = data_set[:int(len(data_set) * 0.66)]  # fisrt 66% data
    data_test = data_set[int(len(data_set) * 0.66):]  # last 34% data
    train_x = [data[:5] for data in data_train]  # traing features
    train_y = [data[5] for data in data_train]  # training/target class
    test_x = [data[:5] for data in data_test]
    test_y = [data[5] for data in data_test]

    weight = [random.random() for j in range(5)]
    training_rate = 0.005
    max_iteration = 1
    cost = 1
    print("Weight Vector Before Training: ", weight)
    weight=train_data(train_x, train_y, weight, training_rate, max_iteration, cost)
    print("Weight Vector After Training: ",weight)
    test_data(test_x, test_y, weight)


if __name__ == "__main__":
    main()
