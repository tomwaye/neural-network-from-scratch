# Neural Network From Scratch

A two-layer neural network that recognises handwritten digits (0–9), written using only NumPy. Every step (forward propagation, backpropagation and mini-batch gradient descent) is implemented by hand.

After 50 epochs it reaches **96.6% accuracy** on 1,000 images it never trained on.

## Results

| Version | Training accuracy | Dev accuracy |
|---|---|---|
| Tutorial version: 10 hidden neurons, full-batch, 500 iterations | 84.3% | 84.8% |
| Mini-batches with shuffling: 10 hidden neurons, 50 epochs | 94.9% | 93.1% |
| Mini-batches with shuffling: 64 hidden neurons, 50 epochs | 99.7% | 96.6% |

Dev accuracy is measured on 1,000 images held out from training, so it shows how well the network handles digits it hasn't seen.

## Credit

Built by following Samson Zhang's video *"Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math)"*.

Beyond the tutorial, I:
- switched to **mini-batch gradient descent**, updating the weights after every 64 images instead of once per pass over the whole dataset
- **shuffled** the training images at the start of every epoch so each mini-batch is different
- made the **hidden layer size configurable** and increased it from 10 to 64 neurons
- measured accuracy on a **held-out dev set** to check the network generalises rather than memorises
- added `visualise.py` to plot training accuracy, the network's guesses, and what each hidden neuron has learned
- fixed bugs in the tutorial code: bias gradients are now per-neuron, softmax normalises each image separately, and one-hot encoding always has 10 classes (it previously broke on mini-batches that happened to contain no 9s)

## How it works

The network has 784 inputs (one per pixel of a 28 × 28 image), a hidden layer of 64 neurons, and an output layer of 10 neurons (one per digit).

- **Forward propagation:** each hidden neuron multiplies every pixel by a weight and adds them up. Its weights act like a stencil, so a high score means the picture matches the pattern that neuron is looking for. ReLU switches off neurons with negative scores. The output layer does the same with the hidden neurons' scores, and softmax turns the results into a probability for each digit.
- **Backpropagation:** the network compares its probabilities with the correct answer (one-hot encoded) and works backwards through the layers to find how much each weight contributed to the error.
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
4. Or see the plots:
   ```bash
   python visualise.py
   ```
   `visualise.py` still trains with the original full-batch method (500 iterations), so its accuracy is lower than `main.py`'s.
