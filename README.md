# Neural Network From Scratch

A two-layer neural network that recognises handwritten digits (0–9), written using only NumPy. Every step (forward propagation, backpropagation and gradient descent) is implemented by hand.

After 500 training iterations it reaches about **84% accuracy** on 1,000 images it never trained on.

## Credit

Built by following Samson Zhang's video *"Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math)"*.

Beyond the tutorial, I:
- added `visualise.py` to plot training accuracy, the network's guesses, and what each hidden neuron has learned
- measured accuracy on a held-out dev set to check the network generalises
- fixed the bias gradients to be per-neuron and the softmax to normalise each example separately

## How it works

The network has 784 inputs (one per pixel of a 28 × 28 image), a hidden layer of 10 neurons, and an output layer of 10 neurons (one per digit).

- **Forward propagation:** each hidden neuron multiplies every pixel by a weight and adds them up. Its weights act like a stencil, so a high score means the picture matches the pattern that neuron is looking for. ReLU switches off neurons with negative scores. The output layer does the same with the hidden neurons' scores, and softmax turns the results into a probability for each digit.
- **Backpropagation:** the network compares its probabilities with the correct answer (one-hot encoded) and works backwards through the layers to find how much each weight contributed to the error.
- **Gradient descent:** every weight is nudged slightly in the direction that reduces the error, and the process repeats.

The weights start as random noise. Through repeated training they become stencils that match the shapes of digits.

## Running it

1. Download `train.csv` from the [Kaggle Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer/data) and put it in this folder.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the network:
   ```bash
   python main.py
   ```
4. Or train it and see the plots:
   ```bash
   python visualise.py
   ```
