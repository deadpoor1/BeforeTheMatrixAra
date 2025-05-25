import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

X = np.array([-2, -1, 0, 1, 2])
Y = np.array([-2, -1, 0, 1, 2])
alpha = 0.01  # learning rate


# one node
w1 = np.random.randn(1)  # weights for input layer to hidden layer
b1 = np.random.randn(1)      # biases for input layer 

def forward_propagation(X, w,b):
    return w * X + b  # Linear transformation (y = wx + b)

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)  # Mean Squared Error

def gradient_wights(X, y_true, y_pred):
    return np.dot(2*X, y_pred - y_true).mean()

def gradient_bias(X, y_true, y_pred):
    return np.dot(2, y_pred - y_true).mean()


epochs = 500
fig = plt.figure()
camera = Camera(fig)
print(f'Initial weights: {w1[0]:.4f}, Initial bias: {b1[0]:.4f}')
print(f'Initial loss: {MSE(Y, forward_propagation(X, w1, b1)):.4f}')
for epoch in range(epochs):

    y_pred = forward_propagation(X, w1, b1)
    loss = MSE(Y, y_pred)
    dw = gradient_wights(X, Y, y_pred)  # compute gradient
    db = gradient_bias(X, Y, y_pred)
    w1 -= alpha * dw  # update weights
    b1 -= alpha * db  # update biases
    if (epoch + 1) % 20 == 0:
        #first epochs
        if epoch+1 == 20:
            plt.title('Neural Network single Node Gradient Descent')
            plt.scatter(X, y_pred, label='Predicted', color='red')
            plt.plot(X, Y, label='True', color='blue')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            camera.snap()
    # Plot for the animation
        plt.scatter(X, y_pred,color='red')
        plt.plot(X, Y, color='blue')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        camera.snap()
print(f'Final weights: {w1[0]:.4f}, Final bias: {b1[0]:.4f}')
print(f'Final loss: {MSE(Y, forward_propagation(X, w1, b1)):.4f}')

animation = camera.animate()
animation.save('gradient_descent.gif', writer='pillow')
plt.close()