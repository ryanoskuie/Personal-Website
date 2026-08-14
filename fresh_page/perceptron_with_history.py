from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
# x: features (sepal/petal measurements), y: class labels (0,1,2)
x, y = iris.data, iris.target

# Split into training and test sets. Keep random_state fixed for reproducibility.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Standardize features (zero mean, unit variance) - important for neural networks
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Configure a small MLP (one hidden layer of 10 units)
# - max_iter: maximum training iterations (epochs)
# - activation: ReLU nonlinearity in hidden layer
# - solver: 'adam' optimizer (adaptive SGD)
# - verbose=True prints training loss each iteration (can be noisy)
mlp = MLPClassifier(hidden_layer_sizes=(10,),
                    max_iter=1000,
                    activation='relu',
                    solver='adam',
                    verbose=True,
                    random_state=42)

# Train the neural network on the training data
mlp.fit(x_train, y_train)

# Make predictions on the test set and evaluate accuracy
y_pred = mlp.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
