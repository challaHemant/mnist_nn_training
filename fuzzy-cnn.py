import numpy as np  # Importing NumPy for numerical operations like matrix operations and array manipulations
from keras.datasets import mnist  # Importing the MNIST dataset from Keras

# Function to load MNIST data and normalize it
def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # Load the MNIST dataset (training and testing sets)
    # Normalize pixel values to be between 0 and 1
    x_train = x_train / 255.0  # Divide pixel values by 255 to scale them between 0 and 1 for training data
    x_test = x_test / 255.0  # Same normalization for test data
    return x_train, y_train, x_test, y_test  # Return the normalized training and test data

# One-hot encoding function to convert labels to binary vectors
def one_hot_encode(labels, num_classes=10):
    encoded = np.zeros((len(labels), num_classes))  # Initialize a zero matrix with rows = number of labels and columns = number of classes
    for i, label in enumerate(labels):  # Loop through each label in the dataset
        encoded[i, label] = 1  # Set the corresponding position for each label to 1 (binary representation)
    return encoded  # Return the encoded labels as one-hot vectors

# Sigmoid activation function (used in hidden layers)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Apply the sigmoid function to the input, returning a value between 0 and 1

# Sigmoid derivative for backpropagation (to calculate the gradient)
def sigmoid_derivative(x):
    return x * (1 - x)  # Derivative of the sigmoid function (used during backpropagation)

# Softmax function for multi-class classification (used for output layer)
def softmax(logits):
    exp_vals = np.exp(logits - np.max(logits))  # Compute the exponentials for each logit (for numerical stability)
    return exp_vals / np.sum(exp_vals)  # Normalize the exponentials to sum to 1, giving us the probabilities

# Initialize the network (weights and biases)
def initialize_network(input_size, hidden_size, output_size):
    np.random.seed(42)  # Ensure reproducibility by setting a fixed random seed
    weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01  # Initialize input-to-hidden layer weights with small random values
    biases_input_hidden = np.zeros((1, hidden_size))  # Initialize input-to-hidden biases to zero

    weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01  # Initialize hidden-to-output layer weights
    biases_hidden_output = np.zeros((1, output_size))  # Initialize hidden-to-output biases to zero

    return (weights_input_hidden, biases_input_hidden, weights_hidden_output, biases_hidden_output)  # Return the initialized parameters

# Training function for the neural network
def train(x_train, y_train, input_size, hidden_size, output_size, epochs=10, learning_rate=0.1):
    # Initialize the network parameters (weights and biases)
    weights_input_hidden, biases_input_hidden, weights_hidden_output, biases_hidden_output = initialize_network(input_size, hidden_size, output_size)

    for epoch in range(epochs):  # Loop through the number of training epochs
        total_loss = 0  # Initialize total loss for the current epoch
        correct_predictions = 0  # Initialize correct predictions for accuracy calculation

        for i in range(len(x_train)):  # Loop through each training example
            # Forward pass
            input_layer = x_train[i].flatten()  # Flatten the 28x28 image into a 1D vector
            hidden_layer_input = np.dot(input_layer, weights_input_hidden) + biases_input_hidden  # Linear transformation to hidden layer
            hidden_layer_output = sigmoid(hidden_layer_input)  # Apply sigmoid activation to the hidden layer

            output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + biases_hidden_output  # Linear transformation to output layer
            output_layer_output = softmax(output_layer_input)  # Apply softmax to get probabilities for each class

            # Compute the loss (Cross-Entropy)
            true_label = y_train[i]  # Get the true label (one-hot encoded)
            loss = -np.sum(true_label * np.log(output_layer_output + 1e-9))  # Cross-entropy loss with a small epsilon for numerical stability
            total_loss += loss  # Add the loss to the total loss for this epoch

            # Check if the prediction is correct
            predicted_label = np.argmax(output_layer_output)  # Predicted class
            actual_label = np.argmax(true_label)  # True class
            if predicted_label == actual_label:
                correct_predictions += 1  # Increment correct predictions if the prediction matches the true label

            # Backpropagation
            output_error = output_layer_output - true_label  # Compute the error for the output layer
            hidden_error = np.dot(output_error, weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)  # Compute the error for the hidden layer

            # Update weights and biases using gradient descent
            weights_hidden_output -= learning_rate * np.outer(hidden_layer_output, output_error)  # Update output weights
            biases_hidden_output -= learning_rate * output_error  # Update output biases

            weights_input_hidden -= learning_rate * np.outer(input_layer, hidden_error)  # Update hidden weights
            biases_input_hidden -= learning_rate * hidden_error  # Update hidden biases

        # Calculate average loss and accuracy
        avg_loss = total_loss / len(x_train)
        accuracy = (correct_predictions / len(x_train)) * 100  # Accuracy in percentage

        # Print loss and accuracy for this epoch
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return (weights_input_hidden, biases_input_hidden, weights_hidden_output, biases_hidden_output)  # Return the trained parameters

# Main function to run everything
def main():
    # Load MNIST data
    x_train, y_train, x_test, y_test = load_mnist()  # Load and normalize MNIST data

    # One-hot encode the labels
    y_train_encoded = one_hot_encode(y_train)  # One-hot encode the training labels
    y_test_encoded = one_hot_encode(y_test)  # One-hot encode the test labels

    # Initialize network sizes
    input_size = 28 * 28  # 28x28 images flattened into a 1D vector (784 features)
    hidden_size = 128  # Number of neurons in the hidden layer
    output_size = 10  # 10 classes for digits 0-9

    # Train the network
    train(x_train, y_train_encoded, input_size, hidden_size, output_size, epochs=10, learning_rate=0.1)  # Train the network for 10 epochs with a learning rate of 0.1

if __name__ == "__main__":
    main()  # Call the main function to start the program
