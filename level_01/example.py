import numpy as np


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# Input data
X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# Target
y = np.array([[0, 0, 1, 1]]).T

# Set random seed
np.random.seed(1)

# Initialize hidden layer with random weights
syn0 = 2 * np.random.random((3, 1)) - 1

print("Input:\n{}".format(X))
print("Target:\n{}".format(y))

# Input data
l0 = X

for i, j in enumerate(range(60000)):

    # Feed-forward
    l1 = l0.dot(syn0)

    # Transform into [0, 1] probabilistic decision
    l1 = sigmoid(l1)

    # Compute error
    l1_error = y - l1

    # Compute gradient as the sigmoid slope
    loss = l1_error * sigmoid(l1, deriv=True)

    # Update weights
    syn0 += l0.T.dot(loss)

    if i % 10000 == 0:
        print("Training iteration {} - loss {}".format(i, sum(loss)))

print("Output:\n{}".format(l1))
