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

# randomly initialize our weights with mean 0
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

print("Input:\n{}".format(X))
print("Target:\n{}".format(y))

# Input data
l0 = X

for i, j in enumerate(range(60000)):

    # Feed-forward through two layers network
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))

    # Compute output error
    l2_error = y - l2

    # Compute gradient as the sigmoid slope
    l2_delta = l2_error * sigmoid(l2, deriv=True)

    # How much did each l1 value contribute to the l2 error
    l1_error = l2_delta.dot(syn1.T)

    # In what direction
    l1_delta = l1_error * sigmoid(l1, deriv=True)

    # Update weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

    if i % 10000 == 0:
        print("Training iteration {} - loss {}".format(i, sum(l2_error)))

print("Output:\n{}".format(l2))
