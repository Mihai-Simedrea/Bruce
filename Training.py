import numpy as np
import random
import NeuralNetwork as nn

if __name__ == '__main__':
    epochs = 20
    lr = 0.1

    class data_class:
        def __init__(self, input, target):
            self.input = input
            self.target = target

    def initialize_parameters_he(layers_dims):
        # np.random.seed(3)
        parameters = {}
        L = len(layers_dims) - 1
        for l in range(1, L + 1):
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(
                2. / layers_dims[l - 1])
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        return parameters

    weights = initialize_parameters_he([784, 500, 11])

    weights['b2'] = weights['b2'].reshape(11, 1)
    weights['b1'] = weights['b1'].reshape(500, 1)


    print('loading data...')
    inputs = nn.LoadData(60000, 0)
    random.shuffle(inputs)

    for i in range(0, 60000+4078):
        inputs[i].input = np.asarray(inputs[i].input)
        inputs[i].input = inputs[i].input.reshape(784, 1)

    print('')
    print('')

    print('training...')
    print('')
    for z in range(0, epochs):
        for i in range(0, 60000+4078):
            hidden1 = nn.create_hidden(weights['W1'], inputs[i].input, weights['b1'], activation='sigmoid',
                                       dropout=True)
            output = nn.create_hidden(weights['W2'], hidden1, weights['b2'], activation='sigmoid', dropout=False)
            target = nn.create_target(inputs[i].target)
            error = np.subtract(target, output)

            hidden1_error = np.dot(weights['W2'].T, error)

            # Gradient for the OUTPUT #
            derivative_output = nn.Derivative(output, activation='sigmoid')
            gradient = error * derivative_output
            gradient *= lr

            # delta bias2 #
            weights['b2'] += gradient

            # Delta weights based on gradient and transposed matrices #
            delta_weight2 = np.dot(gradient, hidden1.T)

            # Gradient for the HIDDEN #
            derivative_output = nn.Derivative(hidden1, activation='sigmoid')
            gradient = hidden1_error * derivative_output
            gradient *= lr

            # delta bias1 #
            weights['b1'] += gradient

            # Delta weights based on gradient and transposed matrices #
            delta_weight1 = np.dot(gradient, inputs[i].input.T)

            weights['W1'] += delta_weight1
            weights['W2'] += delta_weight2

        np.savetxt('W2.txt', weights['W2'])
        np.savetxt('W1.txt', weights['W1'])

        np.savetxt('b2.txt', weights['b2'])
        np.savetxt('b1.txt', weights['b1'])
        print(z, '/', epochs)
        random.shuffle(inputs)



    np.savetxt('W2.txt', weights['W2'])
    np.savetxt('W1.txt', weights['W1'])

    np.savetxt('b2.txt', weights['b2'])
    np.savetxt('b1.txt', weights['b1'])

