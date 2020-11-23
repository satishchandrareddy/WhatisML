#unittest_forwardbackprop.py

import NeuralNetwork
import numpy as np
import unittest

class Test_functions(unittest.TestCase):
    
    def test_LinearRegression(self):
        # (1) create input/output training data X and Y (random)
        nfeature = 8
        m = 1000
        X = np.random.rand(nfeature,m)
        Y = np.random.rand(1,m)
        # (2) define object
        model = NeuralNetwork.NeuralNetwork(nfeature)
        model.add_layer(1,"linear")
        # (3) compile
        optimizer = None
        model.compile("meansquarederror",optimizer)
        # (4) perform derivative test
        eps = 1e-5
        error = model.test_derivative(X,Y,eps)
        print("forwardbackprop: LinearRegression Error: {}".format(error))
        # (5) assert statement
        self.assertLessEqual(error,1e-7)
  
    def test_LogisticRegression(self):
        # (1) create input/output training data X and Y (random)
        nfeature = 2
        m = 1000
        X = np.random.rand(nfeature,m)
        Y = np.round(np.random.rand(1,m))
        # (2) define object
        model = NeuralNetwork.NeuralNetwork(nfeature)
        model.add_layer(1,"sigmoid")
        # (3) compile
        optimizer = None
        model.compile("binarycrossentropy",optimizer)
        # (4) perform derivative test
        eps = 1e-5
        error = model.test_derivative(X,Y,eps)
        print("forwardbackprop: LogisticRegression Error: {}".format(error))
        self.assertLessEqual(error,1e-7)

    def test_NeuralNetwork_binary(self):
        # (1) create input/output training data X and Y
        nfeature = 2
        m = 1000
        X = np.random.rand(nfeature,m)
        Y = np.round(np.random.rand(1,m))
        # (2) define neural network
        model = NeuralNetwork.NeuralNetwork(nfeature)
        model.add_layer(5,"softplus")
        model.add_layer(3,"tanh")
        model.add_layer(1,"sigmoid")
        # (3) compile
        optimizer = None
        model.compile("binarycrossentropy",optimizer)
        # (4) perform derivative test
        eps = 1e-5
        error = model.test_derivative(X,Y,eps)
        print("forwardbackprop: 3 layer NeuralNetwork Error: {}".format(error))
        self.assertLessEqual(error,1e-7)

    def test_NeuralNetwork_multi(self):
       # (1) create input/output training data X and Y
        nfeature = 2
        m = 1000
        nclass = 4
        X = np.random.rand(nfeature,m)
        Y = np.round((nclass-1)*np.random.rand(1,m))
        # (2) define neural network
        model = NeuralNetwork.NeuralNetwork(nfeature)
        model.add_layer(5,"softplus")
        model.add_layer(3,"tanh")
        model.add_layer(nclass,"softmax")
        # (3) compile
        optimizer = None
        model.compile("crossentropy",optimizer)
        # (4) perform derivative test
        eps = 1e-5
        error = model.test_derivative(X,Y,eps)
        print("forwardbackprop: 3 layer NeuralNetwork softmax Error: {}".format(error))
        self.assertLessEqual(error,1e-7)

if __name__ == "__main__":
    unittest.main()
