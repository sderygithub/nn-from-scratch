import numpy as np


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

y = np.array([[0, 0, 1, 1]]).T

np.random.seed(1)

syn0 = 2 * np.random.random((3, 1)) - 1

print("Input:\n{}".format(X))
print("Output:\n{}".format(y))

# Input data
l0 = X

for i, j in enumerate(range(60000)):

    # Feed-forward
    l1 = np.dot(l0, syn0)

    # Transform into [0, 1] probabilistic decision
    l1 = sigmoid(l1)

    # Compute error
    l1_error = y - l1

    # Compute gradient as the sigmoid slope
    loss = l1_error * sigmoid(l1, deriv=True)

    # Update weights
    syn0 += np.dot(l0.T, loss)

    if i % 10000 == 0:
        print("Training iteration {} - loss {}".format(i, sum(loss)))

print(l1)
