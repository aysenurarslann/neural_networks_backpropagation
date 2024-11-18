# neural_networks_backpropagation
# Neural Network Implementation with Visualization
This project implements a simple multi-layer perceptron (MLP) neural network from scratch using Python. The network is capable of training on a small dataset and visualizes its structure both before and after training.
## Features
- Implements a feedforward neural network from scratch.
- Provides customizable input, hidden, and output layer sizes.
- Includes a detailed visualization of the network architecture.
- Trains on simple datasets using backpropagation and gradient descent.
- Outputs weights, biases, and loss metrics during training.
## Usage
Requirements
Ensure you have Python installed along with the following libraries:

numpy
matplotlib
networkx

### Install the required libraries using:
```python
pip install numpy matplotlib networkx ```

## Run the Code
Specify the Number of Neurons in the Hidden Layer:
Upon running, you will be prompted to input the number of neurons for the hidden layer:

Enter the number of hidden layer neurons:

Enter a positive integer based on your requirements.

## Training and Visualization
The code will:

Train the neural network on a predefined dataset for 10,000 epochs.
Display the structure of the network before training (with initial weights and biases).
Display the structure of the network after training (with updated weights and biases).

# Code Details

## 1. NeuralNetwork Class
This class encapsulates the entire functionality of the neural network, including:

### Initialization (__init__)
Sets up the network with random weights, biases, and a learning rate.
### Forward Pass (forward)
Computes the network's output for given inputs using the sigmoid activation function.
### Backward Pass (backward)
Updates the network's weights and biases using gradient descent based on the error.
### Training (train)
Iteratively trains the network using the provided dataset and visualizes the results.
### Visualization (visualize_network)
Draws the neural network's architecture with weights and biases displayed on the edges and nodes.

## 2. Visualization
The network's structure is visualized using networkx and matplotlib. Key features include:

- Edges labeled with weights.
- Nodes annotated with their bias values.
- Separate graphs for the network before and after training.

# Dataset
The hardcoded dataset (X and y) represents a simple mapping:

- Inputs (X): 4 examples with 4 features each.
- Outputs (y): 4 examples with 2 output values each.

Example:
```python
X = [[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 1], [0, 0, 1, 1]]
y = [[0, 0], [0, 1], [1, 0], [1, 1]]

# Customization

- ### Hidden Layer Size:
Modify the hidden layer size dynamically via user input.

- ### Learning Rate:
Adjust the learning rate in the NeuralNetwork constructor:
```python
nn = NeuralNetwork(input_size=4, hidden_size=hidden_size, output_size=2, learning_rate=0.1)

- ### Training Epochs:
Change the number of training iterations:
```python
nn.train(X, y, epochs=10000)

# Example Output
### 1.Prompt for Hidden Layer Size:
```python
Enter the number of hidden layer neurons: 5

### Training Progress
Displays the final loss after training:
```python
End of Training - Total Error (Loss): 0.001234

### Visualizations

#### Before Training
The network with random weights and biases.

#### After Training
The network with updated weights and biases reflecting learning.

# Future Improvements
- Support for additional activation functions (e.g., ReLU, tanh).
- Integration with larger and more complex datasets.
- Add support for multiple hidden layers.
- Export network parameters (weights and biases) to a file for reuse.

# Acknowledgments
This project is designed to demonstrate the basics of neural network implementation and visualization in Python. It can be extended for more advanced applications.

Happy coding! 🚀
