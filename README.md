# Neural Network From Scratch

A neural network with two hidden layers that recognises handwritten digits (0–9), written using only NumPy. Every step (forward propagation, backpropagation and mini-batch gradient descent) is implemented by hand.

After 50 epochs it reaches **97.7% accuracy** on 1,000 images it never trained on.

## Results

| Version | Training accuracy | Dev accuracy |
|---|---|---|
| Tutorial version: 1 hidden layer of 10 neurons, full-batch, 500 iterations | 84.3% | 84.8% |
| Mini-batches with shuffling: 1 hidden layer of 10 neurons, 50 epochs | 94.9% | 93.1% |
| Mini-batches with shuffling: 1 hidden layer of 64 neurons, 50 epochs | 99.7% | 96.6% |
| Mini-batches with shuffling: 2 hidden layers of 64 neurons, 50 epochs | 99.99% | 97.7% |

Dev accuracy is measured on 1,000 images held out from training, so it shows how well the network handles digits it hasn't seen. The weights start from random values, so results vary by about a percentage point between runs.

## Credit

Built by following Samson Zhang's video *"Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math)"*.

Beyond the tutorial, I:
- added a **second hidden layer**, extending forward propagation and backpropagation to three layers
- switched to **mini-batch gradient descent**, updating the weights after every 64 images instead of once per pass over the whole dataset
- **shuffled** the training images at the start of every epoch so each mini-batch is different
- made the **hidden layer size configurable** and increased it from 10 to 64 neurons
- measured accuracy on a **held-out dev set** to check the network generalises rather than memorises
- added `visualise.py` to plot training accuracy per epoch, the network's predictions on dev images, and the weights each first-layer neuron has learned
- fixed bugs in the tutorial code: bias gradients are now per-neuron, softmax normalises each image separately, and one-hot encoding always has 10 classes (it previously broke on mini-batches that happened to contain no 9s)

## How it works

The network has 784 inputs (one per pixel of a 28 × 28 image), two hidden layers of 64 neurons each, and an output layer of 10 neurons (one per digit).

- **Forward propagation:** each neuron in the first hidden layer multiplies every pixel by a weight and adds them up. Its weights act like a stencil, so a high score means the picture matches the pattern that neuron is looking for. ReLU switches off neurons with negative scores. The second hidden layer does the same with the first layer's outputs, combining simple patterns into more complex ones. The output layer then scores each digit, and softmax turns those scores into probabilities.
- **Backpropagation:** the network compares its probabilities with the correct answer (one-hot encoded) and works backwards through the layers to find how much each weight contributed to the error. Every hidden layer uses the same steps: pass the error back through the next layer's weights, then zero it for neurons that were inactive.
- **Mini-batch gradient descent:** the training images are shuffled and split into batches of 64. After each batch, every weight is nudged slightly in the direction that reduces the error. One pass through all the batches is an epoch.

The weights start as random noise. Through repeated training they become stencils that match the shapes of digits.

## Running it

1. Download `train.csv` from the [Kaggle Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer/data) and put it in this folder.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the network and print its training and dev accuracy:
   ```bash
   python main.py
   ```
4. Or train it and see the plots:
   ```bash
   python visualise.py
   ```
