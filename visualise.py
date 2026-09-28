import numpy as np
from matplotlib import pyplot as plt

from main import (X_train, Y_train, X_dev, Y_dev, initParams, forwardProp,
                  backwardsProp, update_params, get_predictions)

#Same as gradient_descent in main.py, but also records accuracy every iteration
def train_with_history(X, Y, iterations, alpha, size):
    W1, b1, W2, b2 = initParams(size)
    accuracies = []
    for i in range(iterations):
        Z1,A1,Z2,A2 = forwardProp(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backwardsProp(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        accuracies.append(np.sum(get_predictions(A2) == Y) / Y.size)
        if i%10==0:
            print("iteration:", i, "Accuracy:", accuracies[-1])
    return W1, b1, W2, b2, accuracies

W1,b1,W2,b2,accuracies = train_with_history(X_train, Y_train, 500, 0.1, 64)

_, _, _, A2_dev = forwardProp(W1, b1, W2, b2, X_dev)
dev_predictions = get_predictions(A2_dev)
print("Dev accuracy:", np.sum(dev_predictions == Y_dev) / Y_dev.size)

#1. Accuracy over training
plt.plot(accuracies, linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Training accuracy")
plt.title("Training accuracy")
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.show()

#2. The network's guesses on dev images it never trained on
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_dev[:, i].reshape(28, 28), cmap="gray")
    mark = "✓" if dev_predictions[i] == Y_dev[i] else "✗"
    ax.set_title(f"{mark} guess {dev_predictions[i]}, true {Y_dev[i]}")
    ax.axis("off")
plt.tight_layout()
plt.show()

#3. What each hidden neuron looks for: blue pixels excite it, red pixels suppress it
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
limit = np.abs(W1).max()
for i, ax in enumerate(axes.flat):
    ax.imshow(W1[i].reshape(28, 28), cmap="RdBu", vmin=-limit, vmax=limit)
    ax.set_title(f"Neuron {i}")
    ax.axis("off")
plt.tight_layout()
plt.show()
