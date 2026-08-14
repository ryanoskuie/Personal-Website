import numpy as np

# Training data from the slide
X = np.array([
    [1.0, 1.0],
    [9.4, 6.4],
    [2.5, 2.1],
    [8.0, 7.7],
    [0.5, 2.2],
    [7.9, 8.4],
    [7.0, 7.0],
    [2.8, 0.8],
    [1.2, 3.0],
    [7.8, 6.1]
])

y = np.array([1, -1, 1, -1, 1, -1, -1, 1, 1, -1])  # labels

# Add bias term (x0 = 1)
X = np.insert(X, 0, 1, axis=1)

# Parameters
alpha = .2  # learning rate
w = np.array([-0.6, 0.75, 0.5])  # initial weights (including bias)
epochs = 500  # fixed to 500 iterations

# Activation function
def activation(z):
    return 1 if z >= 0 else -1

# Training loop
for _ in range(epochs):
    for i in range(len(X)):
        z = np.dot(w, X[i])
        y_pred = activation(z)
        if y_pred != y[i]:  # update if misclassified
            w = w - alpha * (y_pred - y[i]) * X[i]
            

print("Final weights after 500 iterations:", w)