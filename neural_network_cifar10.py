import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from torchvision.datasets import CIFAR10

train = CIFAR10(root="data", train=True, download=True)
test = CIFAR10(root="data", train=False, download=True)

X_train = train.data
X_train = X_train.reshape(50000, 3072).T / 255
Y_train = np.array(train.targets)

X_dev = test.data
X_dev = X_dev.reshape(-1, 3072).T / 255
Y_dev = np.array(test.targets)

#Initialise the weights (W) and biases (b) for all three layers with random values between -0.5 and 0.5
#Each W is shaped (neurons in this layer, inputs from the previous layer); each b is (neurons in this layer, 1)
#Layers: 784 inputs -> hidden layer 1 (size) -> hidden layer 2 (size) -> output layer (10, one per digit)
def initParams(size, X):
    W1 = np.random.randn(size, X.shape[0]) * np.sqrt(2 / X.shape[0])
    b1 = np.zeros((size, 1))
    W2 = np.random.randn(size, size)* np.sqrt(2 / size)
    b2 = np.zeros((size, 1))
    W3 = np.random.randn(10, size)* np.sqrt(2 / size)
    b3 = np.zeros((10, 1))
    return W1, b1, W2, b2, W3, b3

#ReLU activation: keeps positive values and sets negative values to 0
def ReLU(Z):
    return np.maximum(0, Z)

#Softmax activation: converts each image's output scores into probabilities that sum to 1
def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)

#One-hot encode the labels, e.g. 3 -> [0,0,0,1,0,0,0,0,0,0], so they can be compared with the output probabilities
def oneHot(Y):
    oneHotY = np.zeros((Y.size, 10)) #one row per image, one column per digit (always 10, even if a batch is missing some digits)
    oneHotY[np.arange(Y.size), Y] = 1 #set a 1 in the column of each image's label
    oneHotY = oneHotY.T #transpose so each image is a column, matching A3
    return oneHotY

#Derivative of ReLU: 1 where the neuron was active (Z > 0), 0 where it wasn't
def deriv_ReLU(Z):
    return Z > 0

#Forward propagation: pass the images through the network to get output probabilities
#Z = weighted sum of the inputs plus bias, A = Z after the activation function
def forwardProp(W1, b1, W2, b2, W3, b3, x):
    Z1 = W1.dot(x) + b1 #hidden layer 1: weighted sum of the pixels
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2 #hidden layer 2: weighted sum of hidden layer 1's outputs
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3 #output layer: one score per digit
    A3 = softmax(Z3) #output probabilities - the network's prediction
    return Z1, A1, Z2, A2, Z3, A3

#Backpropagation: work backwards from the output error to find the gradient of every weight and bias
#Variables starting with d are gradients - how much each value should change to reduce the error
def backwardsProp(Z1, A1, Z2, A2, W2, Z3, A3, W3, X, Y):
    m = Y.size #number of images in the batch, used to average the gradients
    oneHotY = oneHot(Y)

    #Output layer
    dZ3 = A3 - oneHotY #error: predicted probabilities minus the correct one-hot labels
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

    #Hidden layer 2
    dZ2 = W3.T.dot(dZ3) * deriv_ReLU(Z2) #pass the error back through W3; inactive neurons get no gradient
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    #Hidden layer 1
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1) #pass the error back through W2; inactive neurons get no gradient
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3

#Gradient descent step: move every weight and bias a small step against its gradient
#alpha = learning rate (the size of each step)
def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3,alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

#The predicted digit for each image is the one with the highest probability
def get_predictions(A3):
    return np.argmax(A3, 0)

#Fraction of images where the prediction matches the label
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

#Training with mini-batch gradient descent
#Each epoch shuffles the training images, splits them into batches of 64, and updates the weights after every batch
def gradient_descent(X, Y, iterations, alpha, size):
    W1, b1, W2, b2, W3, b3 = initParams(size, X_train)
    accuracies = [] #training accuracy after each epoch, used by visualise.py
    for i in range(iterations):
        #Shuffle images and labels with the same order so each image keeps its label
        shuffled_index = np.random.permutation(X.shape[1])
        shuffledX = X[:,shuffled_index]
        shuffledY = Y[shuffled_index]
        for c in range(0, X.shape[1], 64):
            chunkX = shuffledX[:,c:c+64] #the next 64 images (columns)
            chunkY = shuffledY[c:c+64] #their labels
            Z1,A1,Z2,A2,Z3,A3 = forwardProp(W1, b1, W2, b2, W3, b3, chunkX)
            dW1, db1, dW2, db2, dW3, db3 = backwardsProp(Z1, A1, Z2, A2, W2, Z3, A3, W3, chunkX, chunkY)
            W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        #After each epoch, measure accuracy on the full training set
        _,_,_,_,_,A3 = forwardProp(W1, b1, W2, b2, W3, b3, X)
        print("iteration:", i)
        accuracies.append(get_accuracy(get_predictions(A3), Y))
        print("Accuracy:", accuracies[-1])
    return W1, b1, W2, b2, W3, b3, accuracies

#Only train when this file is run directly, not when visualise.py imports it
if __name__ == "__main__":
    W1,b1,W2,b2,W3,b3,_ = gradient_descent(X_train, Y_train, 20, 0.01, 256)
    #Measure accuracy on the dev set, which the network never trained on
    _, _, _,_,_,A3_dev = forwardProp(W1, b1, W2, b2, W3, b3, X_dev)
    dev_predictions = get_predictions(A3_dev)
    print("Dev accuracy:", np.sum(dev_predictions == Y_dev) / Y_dev.size)
