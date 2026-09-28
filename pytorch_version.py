import torch
from torch import nn

import numpy as np
import pandas as pd

#Load the dataset. Each row is one image: the label (0-9) followed by 784 pixel values (0-255)
data = pd.read_csv("train.csv")
data = np.array(data)
m,n = data.shape #m = number of images, n = values per row (label + 784 pixels)
np.random.shuffle(data) #shuffle the rows so the dev/train split is random

#Dev set: 1000 images held out from training, used to check the network generalises
#Unlike main.py there is no .T - PyTorch expects one image per row, shape (images, 784)
data_dev = data[0:1000]
Y_dev = data_dev[:, 0] #labels: the first column
X_dev = data_dev[:, 1:n] / 255. #pixels: every other column, scaled from 0-255 to 0-1

#Training set: the remaining images
data_train = data[1000:]
Y_train = data_train[:, 0]
X_train = data_train[:, 1:n] / 255.

#Convert the NumPy arrays to PyTorch tensors
#Pixels must be float32 (the type the layers use); labels must be long (whole numbers) for the loss function
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)
X_dev = torch.tensor(X_dev, dtype=torch.float32)
Y_dev = torch.tensor(Y_dev, dtype=torch.long)


#The network: replaces initParams and forwardProp in main.py
#nn.Linear(inputs, outputs) is one layer - it creates and initialises its own weights and bias
#nn.Sequential runs the layers in order, so model(X) does the whole forward pass
def build_model(size):
    return nn.Sequential(
        nn.Linear(784, size), #hidden layer 1 (W1, b1)
        nn.ReLU(),
        nn.Linear(size, size), #hidden layer 2 (W2, b2)
        nn.ReLU(),
        nn.Linear(size, size), #output layer (W3, b3) - outputs raw scores (Z3), no softmax
        nn.ReLU(),
        nn.Linear(size, 10),
    )

#Fraction of images where the predicted digit matches the label
def get_accuracy(model, X, Y):
    with torch.no_grad(): #we're only measuring, so skip tracking gradients
        predictions = model(X).argmax(dim=1) #highest score in each row = predicted digit
    return (predictions == Y).float().mean().item() #.item() converts a one-value tensor to a Python number

#Training with mini-batch gradient descent: replaces gradient_descent in main.py
def train(X, Y, epochs, alpha, size, batch_size=64):
    model = build_model(size)

    #CrossEntropyLoss applies softmax and compares with the labels itself, so no oneHot is needed
    #It replaces oneHot and dZ3 = A3 - oneHotY in main.py
    loss_fn = nn.CrossEntropyLoss()

    #SGD (stochastic gradient descent) updates every weight: replaces update_params
    #model.parameters() gives it all the weights and biases in the model
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    for epoch in range(epochs):
        #Shuffle images and labels with the same order, same as main.py
        shuffled_index = torch.randperm(X.shape[0])
        shuffledX = X[shuffled_index] #rows now, not columns
        shuffledY = Y[shuffled_index]

        for c in range(0, X.shape[0], batch_size):
            chunkX = shuffledX[c:c+batch_size]
            chunkY = shuffledY[c:c+batch_size]

            outputs = model(chunkX) #forward propagation
            loss = loss_fn(outputs, chunkY) #how wrong the predictions were

            optimizer.zero_grad() #clear the previous batch's gradients (PyTorch adds to them otherwise)
            loss.backward() #backpropagation: computes the gradient of every weight and bias automatically
            optimizer.step() #gradient descent: moves every weight and bias against its gradient

        print("epoch:", epoch, "Accuracy:", get_accuracy(model, X, Y))
    return model

#Same settings as main.py: 50 epochs, learning rate 0.1, 64 neurons per hidden layer
if __name__ == "__main__":
    model = train(X_train, Y_train, 10, 0.001, 64)
    print("Dev accuracy:", get_accuracy(model, X_dev, Y_dev))
