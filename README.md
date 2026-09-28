# Neural Network From Scratch

A neural network with two hidden layers, written using only NumPy. Every step (forward propagation, backpropagation and mini-batch gradient descent) is implemented by hand.

I trained it on two datasets:
- **MNIST** (handwritten digits): **97.7%** accuracy on images it never trained on
- **CIFAR-10** (colour photos of objects): **49.9%** accuracy on images it never trained on

The same code does far worse on photos than on digits. The CIFAR-10 section below explains why.

## MNIST: handwritten digits

28 × 28 greyscale images of the digits 0–9, from the [Kaggle Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer/data).

| Version | Training accuracy | Dev accuracy |
|---|---|---|
| Tutorial version: 1 hidden layer of 10 neurons, full-batch, 500 iterations | 84.3% | 84.8% |
| Mini-batches with shuffling: 1 hidden layer of 10 neurons, 50 epochs | 94.9% | 93.1% |
| Mini-batches with shuffling: 1 hidden layer of 64 neurons, 50 epochs | 99.7% | 96.6% |
| Mini-batches with shuffling: 2 hidden layers of 64 neurons, 50 epochs | 99.99% | 97.7% |

Dev accuracy is measured on 1,000 images held out from training, so it shows how well the network handles digits it hasn't seen. The weights start from random values, so results vary by about a percentage point between runs.

## CIFAR-10: colour photos

32 × 32 colour photos in 10 classes: plane, car, bird, cat, deer, dog, frog, horse, ship and truck. Each image has 3,072 inputs (32 × 32 pixels × 3 colour channels). Dev accuracy is measured on CIFAR-10's separate test set of 10,000 images.

All runs use two hidden layers and a learning rate of 0.01.

| Version | Training accuracy | Dev accuracy |
|---|---|---|
| 64 neurons per hidden layer, original weight initialisation | 40% | 39% |
| 64 neurons per hidden layer, He initialisation | — | 45% |
| 256 neurons per hidden layer, He initialisation | 55% | 49.9% |

**Weight initialisation mattered.** The MNIST version starts its weights uniformly between -0.5 and 0.5. That worked on digits, where most pixels are 0, but CIFAR-10 photos have almost no zero pixels and four times as many inputs. The weighted sums grew so large that many neurons never activated, and with 256 neurons softmax overflowed and the network predicted one class for every image (10% accuracy). He initialisation, which scales each layer's starting weights by `sqrt(2 / inputs)`, fixed this.

**Why accuracy is so much lower than on MNIST:** each neuron in the first hidden layer learns one template covering the whole image. Digits are always centred black strokes on a plain background, so templates work well. Objects in photos appear at different positions, sizes and colours against varied backgrounds, and a whole-image template can't match them. Even the training accuracy only reaches 55%. Convolutional neural networks solve this by learning small filters that are applied across the whole image.

## Credit

Built by following Samson Zhang's video *"Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math)"*.

Beyond the tutorial, I:
- added a **second hidden layer**, extending forward propagation and backpropagation to three layers
- switched to **mini-batch gradient descent**, updating the weights after every 64 images instead of once per pass over the whole dataset
- **shuffled** the training images at the start of every epoch so each mini-batch is different
- made the **hidden layer size configurable** and increased it from 10 to 64 neurons
- measured accuracy on a **held-out dev set** to check the network generalises rather than memorises
- trained the network on **CIFAR-10** and added **He initialisation** so it trains on larger, denser inputs
- rebuilt the MNIST network in **PyTorch** (`pytorch_version.py`) to compare it with the hand-written version; with the same settings it reached the same 97.7% dev accuracy
- added `visualise.py` to plot training accuracy per epoch, the network's predictions on dev images, and the weights each first-layer neuron has learned
- fixed bugs in the tutorial code: bias gradients are now per-neuron, softmax normalises each image separately, and one-hot encoding always has 10 classes (it previously broke on mini-batches that happened to contain no 9s)

## How it works

For MNIST, the network has 784 inputs (one per pixel of a 28 × 28 image), two hidden layers of 64 neurons each, and an output layer of 10 neurons (one per digit).

- **Forward propagation:** each neuron in the first hidden layer multiplies every pixel by a weight and adds them up. Its weights act like a stencil, so a high score means the picture matches the pattern that neuron is looking for. ReLU switches off neurons with negative scores. The second hidden layer does the same with the first layer's outputs, combining simple patterns into more complex ones. The output layer then scores each class, and softmax turns those scores into probabilities.
- **Backpropagation:** the network compares its probabilities with the correct answer (one-hot encoded) and works backwards through the layers to find how much each weight contributed to the error. Every hidden layer uses the same steps: pass the error back through the next layer's weights, then zero it for neurons that were inactive.
- **Mini-batch gradient descent:** the training images are shuffled and split into batches of 64. After each batch, every weight is nudged slightly in the direction that reduces the error. One pass through all the batches is an epoch.

The weights start as random noise. Through repeated training they become stencils that match the shapes in the images.

## Files

| File | What it does |
|---|---|
| `neural_network_mnist.py` | The NumPy network, trained on MNIST |
| `neural_network_cifar10.py` | The same network with He initialisation, trained on CIFAR-10 |
| `pytorch_version.py` | The MNIST network rebuilt in PyTorch |
| `visualise.py` | Trains the MNIST network and plots its accuracy, predictions and first-layer weights |

## Running it

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. For MNIST, download `train.csv` from the [Kaggle Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer/data) and put it in this folder. Then run:
   ```bash
   python neural_network_mnist.py
   ```
   Or see the plots:
   ```bash
   python visualise.py
   ```
3. For CIFAR-10, run the script below. It downloads the dataset (about 170 MB) into `data/` the first time:
   ```bash
   python neural_network_cifar10.py
   ```
4. For the PyTorch version (uses `train.csv`):
   ```bash
   python pytorch_version.py
   ```
