import numpy as np
import math
import glob
from PIL import Image
import os
import random


class data_class:
    def __init__(self, input, target):
        self.input = input
        self.target = target


def create_target(value):
    list = []
    for i in range(0, 11):
        if value == i:
            list.append(1)
        else:
            list.append(0)

    array = np.asarray(list)
    array = array.reshape(11, 1)

    return array


def LoadData(maxImages, startPos):
    inputs = []
    for i in range(startPos, maxImages):
        path = "D:\\DigitRecognition-BestV\\DigitRecognition-Software\\database\\training\\*\\{count}.png".format(count=i)
        filename = glob.glob(path)
        im = Image.open(filename[0], 'r').convert('1')
        data = im.getdata()
        directory = os.path.dirname(filename[0])
        digit = int(directory[len(directory) - 1])

        new_data = []

        new_data += [(data[x] / 255) for x in range(0, 784)]

        value = data_class(new_data, digit)
        value.list = new_data
        value.target = digit

        inputs.append(value)



    for i in range(1, 4079):
        path = "D:\\DigitRecognition-BestV\\DigitRecognition-Software\\database\\training\\plus_db\\plus ({index}).png".format(index=i)
        filename = glob.glob(path)
        im = Image.open(filename[0], 'r').convert('1')
        data = im.getdata()
        digit = 10

        new_data = []

        new_data += [(data[x] / 255) for x in range(0, 784)]

        value = data_class(new_data, digit)
        value.list = new_data
        value.target = digit

        inputs.append(value)



    INPUTS = np.asarray(inputs, dtype=None, order=None)

    for i in range(0, len(INPUTS)):
        INPUTS[i].list = np.asarray(INPUTS[i].list)
        INPUTS[i].list = INPUTS[i].list.reshape(784, 1)

    return inputs


def create_hidden(weight, input, bias, activation, dropout):
    hidden = np.dot(weight, input)
    hidden += bias

    if activation == 'sigmoid':
        for i in range(0, len(hidden)):
            hidden[i] = sigmoid(hidden[i])
            if dropout == True:
                drop_out = random.randint(1, 10)

                if drop_out == 1 or drop_out == 2 or drop_out == 3:
                    hidden[i] = 0

    return hidden


def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1 / (1 + math.exp(gamma))
    return 1 / (1 + math.exp(-gamma))


def d_sigmoid(y):
    return y * (1 - y)


def Derivative(array, activation):
    array2 = []

    if activation == 'sigmoid':
        for i in range(0, len(array)):
            value = d_sigmoid(array[i])
            array2.append(value)

    array2 = np.asarray(array2)

    return array2


def relu(x):
    if x > 0:
        return x
    else:
        return 0


def d_relu(x):
    if x > 0:
        return 1
    else:
        return 0
