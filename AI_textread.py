from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import time


X, Y = fetch_openml("mnist_784", version=1, return_X_y=True)

traning_set = 70000
x_set = 784
# Convert DataFrames and Series to numpy arrays
X_np = X.values
Y_np = Y.astype(int).values  # Convert Series to numpy array of strings

shuffle_index = np.random.permutation(len(X_np))
X_shuffled = X_np[shuffle_index]
Y_shuffled = Y_np[shuffle_index]

X_preserve = X_shuffled[0:1000].T
Y_preserve = Y_shuffled[0:1000]
X_preserve = X_preserve / 255.0 

X_train = X_shuffled[1000:70000].T
Y_train = Y_shuffled[1000:70000]
X_train = X_train / 255.0  
print(X_train.shape, Y_train.shape)
time.sleep(6)

W1 = np.random.rand(10, x_set) - 0.5
B1 = np.random.rand(10, 1) - 0.5
W2 = np.random.rand(10, 10) - 0.5
B2 = np.random.rand(10, 1) - 0.5

def ReLU(x):
    return np.maximum(x, 0)

def softmax(x):
    A = np.exp(x) / sum(np.exp(x))
    return A

def forward_propagation(X_train, W1, B1, W2, B2):
    Z1 = W1.dot(X_train) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax(Z2)
    return Z1 , A1, Z2, A2

def one_hot_encode(Y):
    one_hot = np.zeros((Y.size, Y.max() + 1))
    one_hot[np.arange(Y.size), Y] = 1
    one_hot = one_hot.T

    return one_hot

def derivative_ReLU(Z):
    return Z > 0

def backward_propagation(X_train, Y_train, W1, B1, W2, B2, Z1, A1, Z2, A2):
    Y_one_hot = one_hot_encode(Y_train)
    dZ2 = A2 - Y_one_hot
    dW2 = 1/traning_set * dZ2.dot(A1.T)
    dB2 = 1/traning_set * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * derivative_ReLU(Z1)
    dW1 = 1 / traning_set * dZ1.dot(X_train.T)
    dB1 = 1 / traning_set * np.sum(dZ1)


    return dW1, dB1, dW2, dB2

def update_weights(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate=0.1):
    update_W1 = W1 - learning_rate * dW1
    update_B1 = B1 - learning_rate * dB1
    update_W2 = W2 - learning_rate * dW2
    update_B2 = B2 - learning_rate * dB2
    return update_W1, update_B1, update_W2, update_B2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def make_predictions(X, W1, B1, W2, B2):
    _, _, _, A2 = forward_propagation(X, W1, B1, W2, B2)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

epochs = 500

for epoch in range(epochs):

    Z1, A1, Z2, A2 = forward_propagation(X_train, W1, B1, W2, B2)
    dW1, dB1, dW2, dB2 = backward_propagation(X_train, Y_train, W1, B1, W2, B2, Z1, A1, Z2, A2)
    W1, B1, W2, B2 = update_weights(W1, B1, W2, B2, dW1, dB1, dW2, dB2)
    if epoch % 50 == 0:
        predictions = get_predictions(A2)
        accuracy = get_accuracy(predictions, Y_train)
        print(f"Epoch {epoch}, Accuracy: {accuracy:.4f}")
