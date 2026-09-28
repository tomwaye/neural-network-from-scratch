import numpy as np
from matplotlib import pyplot as plt

from main import X_train, Y_train, X_dev, Y_dev, gradient_descent, forwardProp, get_predictions

#Train the network using main.py, with the same settings as main.py
W1,b1,W2,b2,W3,b3,accuracies = gradient_descent(X_train, Y_train, 50, 0.1, 64)

#Predictions on the dev set, which the network never trained on
_,_,_,_,_,A3_dev = forwardProp(W1, b1, W2, b2, W3, b3, X_dev)
dev_predictions = get_predictions(A3_dev)
print("Dev accuracy:", np.sum(dev_predictions == Y_dev) / Y_dev.size)

#1. Training accuracy after each epoch
plt.plot(range(1, len(accuracies) + 1), accuracies, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Training accuracy")
plt.title("Training accuracy")
plt.grid(alpha=0.3)
plt.show()

#2. Dev images the network got wrong, with its prediction and the correct label
mistakes = np.arange(15)
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
for ax, i in zip(axes.flat, mistakes):
    ax.imshow(X_dev[:, i].reshape(28, 28), cmap="gray")
    ax.set_title(f"predicted {dev_predictions[i]}, true {Y_dev[i]}")
for ax in axes.flat:
    ax.axis("off")
fig.suptitle("Dev set mistakes")
plt.tight_layout()
plt.show()

#3. First hidden layer weights: each neuron's 784 weights reshaped to 28x28
#Blue pixels increase the neuron's activation, red pixels decrease it
fig, axes = plt.subplots(8, 8, figsize=(10, 10))
limit = np.abs(W1).max()
for i, ax in enumerate(axes.flat):
    ax.imshow(W1[i].reshape(28, 28), cmap="RdBu", vmin=-limit, vmax=limit)
    ax.axis("off")
fig.suptitle("Hidden layer 1 weights (64 neurons)")
plt.tight_layout()
plt.show()
