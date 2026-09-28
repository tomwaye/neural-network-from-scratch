import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Load the pictures. Each row is one picture: the answer (0-9) then 784 pixel values (0-255)
data = pd.read_csv("train.csv")
data = np.array(data)
m,n = data.shape #m = number of pictures, n = numbers per row (answer + 784 pixels)
print(data.shape)
np.random.shuffle(data) #mix up the order so the split below is random

#Surprise test: 1000 pictures the network never trains on, to check it really learned
#.T flips the grid so each picture is a column instead of a row
data_dev = data[0:1000].T
Y_dev = data_dev[0] #the correct answers
X_dev = data_dev[1:n] / 255. #the pixels, shrunk from 0-255 to 0-1


#Practice pictures: everything else, used for training
data_train = data[1000: ].T
Y_train = data_train[0] #the correct answers
X_train = data_train[1:n] / 255. #the pixels, shrunk from 0-255 to 0-1


#Make the starting weights (W) and biases (b) for both layers as random numbers between -0.5 and 0.5
#W = how much each neuron cares about each of its inputs, b = each neuron's starting mood
#Each W is shaped (neurons in this layer, inputs coming in)
def initParams():
    W1 = np.random.rand(64, 784) - 0.5
    b1 = np.random.rand(64, 1) - 0.5
    W2 = np.random.rand(10, 64) - 0.5 # hidden layer
    b2 = np.random.rand(10, 1) - 0.5 #hidden layer
    return W1, b1, W2, b2

#If Z is greater than 0 return Z if less than 0 return 0
def ReLU(Z):
    return np.maximum(0, Z)

#Turn each picture's scores into percentages that add up to 100%
def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)

#Turn each answer into the "perfect" percentages, e.g. 3 -> [0,0,0,1,0,0,0,0,0,0]
def oneHot(Y):
    oneHotY = np.zeros((Y.size, Y.max() + 1)) #a grid of zeros: one row per picture, one column per digit
    oneHotY[np.arange(Y.size), Y] = 1 #put a 1 in each picture's correct digit spot
    oneHotY = oneHotY.T #flip it so each picture is a column, matching A2
    return oneHotY

#The slope of ReLU: True (1) if the neuron was switched on, False (0) if it was off
def deriv_ReLU(Z):
    return Z > 0

#Make a guess: send the pixels through both layers
def forwardProp(W1, b1, W2, b2, x):
    Z1 = W1.dot(x) + b1 #layer 1: each hidden neuron scores how well the picture matches its stencil
    A1 = ReLU(Z1) #layer 1: negative scores become 0 (the neuron stays quiet)
    Z2 = W2.dot(A1) + b2 #layer 2: each digit's voter scores the hidden neurons' results
    A2 = softmax(Z2) #layer 2: turn the voters' scores into percentages - the final guess
    return Z1, A1, Z2, A2

#Work out the blame: starting from the mistake, go backwards to find how much each weight should change
#Anything starting with d means "how much should this change?"
def backwardsProp(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size #number of pictures, used to average the changes
    oneHotY = oneHot(Y)

    #Layer 2 (the voters)
    dZ2 = A2 - oneHotY #the mistake: guess minus the correct answer
    dW2 = 1 / m * dZ2.dot(A1.T) #how to change the voters' weights
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True) #how to change the voters' moods

    #Layer 1 (the hidden neurons)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1) #pass the blame back; neurons that were off get none
    dW1 = 1 / m * dZ1.dot(X.T) #how to change the hidden neurons' weights
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True) #how to change the hidden neurons' moods
    return dW1, db1, dW2, db2

#Fix: nudge every weight and bias a little in the direction that reduces the mistake
#alpha = how big each nudge is
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

#For each picture, pick the digit with the highest percentage
def get_predictions(A2):
    return np.argmax(A2, 0)

#The fraction of pictures where the guess matched the correct answer
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

#Training: start with random weights, then repeat guess -> blame -> fix over and over
def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = initParams()
    for i in range(iterations):
        Z1,A1,Z2,A2 = forwardProp(W1, b1, W2, b2, X) #guess
        dW1, db1, dW2, db2 = backwardsProp(Z1, A1, Z2, A2, W2, X, Y) #blame
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha) #fix
        if i%100==0: #every 100 rounds, show how it's doing
            print("iteration:", i)
            print("Accuracy:", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2

#Only train when this file is run directly, not when visualise.py imports it
if __name__ == "__main__":
    W1,b1,W2,b2 = gradient_descent(X_train, Y_train, 500, 0.1)
