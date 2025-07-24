#  MNIST Neural Network from Scratch using NumPy

This repository contains a clean, from-scratch implementation of a simple neural network for digit classification using the [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset). The entire training pipeline — from data preprocessing to forward/backward propagation and evaluation — is built using **pure NumPy**.

---

## 📌 Highlights

-  Fully Connected Neural Network with 1 hidden layer
-  Achieves ~97% accuracy on MNIST test data
-  Implemented **Sigmoid** and **Softmax** activation functions
-  Uses **Cross-Entropy Loss** for multi-class classification
-  Custom **Forward Propagation** and **Backpropagation** algorithms
-  Visualizes **Loss** and **Accuracy** during training

---
##  Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mnist-nn.git
cd mnist-nn

2. Install Dependencies
pip install -r requirements.txt
Note: Keras is only used to load the MNIST dataset, not for modeling.

3. Train the Model
python train.py

4. Evaluate the Model
python evaluate.py

# Results
Loss Curve	Accuracy Curve

📈 Final Test Accuracy: ~97.8%

 Key Concepts
 #Forward Propagation
Computes the output probabilities using:

Sigmoid (hidden layer)

Softmax (output layer)

# Backward Propagation
Computes gradients using the chain rule:

Updates weights using gradient descent

# Cross-Entropy Loss
Used for comparing predicted probabilities to true labels (one-hot encoded).

🔍 Sample Output
Epoch 10/10
Loss: 0.0672 | Accuracy: 97.8%

📚 Learnings
Gained deep understanding of neural network internals

Implemented activation functions, weight updates, and gradient flow from scratch

Identified how learning rate and initialization affect convergence

🔮 Future Improvements
Implement ReLU activation

Add Mini-batch Gradient Descent

Support deeper networks and more datasets (e.g., Fashion-MNIST, CIFAR-10)

Add command-line arguments for hyperparameters

