from module import Module

import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # https://stackoverflow.com/questions/32030343/subtracting-the-mean-of-each-row-in-numpy-with-broadcasting
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def sigmoid(x):
    """"""
    return 1 / (1 + np.exp(-x))


class Threshold(Module):

    def __init__(self, threshold, value):
        """ Thresholds each element of the input Tensor

        Threshold is defined as:

         y =  x        if x >  threshold
              value    if x <= threshold

        """
        super().__init__()
        self.threshold = threshold
        self.value = value

    def forward(self, input):
        return input * (input > self.threshold)


class ReLU(Threshold):

    def __init__(self):
        super().__init__(0, 0)


class Sigmoid(Module):
    """Applies the element-wise function,
    :math:`f(x) = 1 / ( 1 + exp(-x))`
    """

    def forward(self, input):
        return sigmoid(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


class Tanh(Module):
    """Applies element-wise,
    :math:`f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
    """

    def forward(self, input):
        return np.tanh(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


class Softmax(Module):
    """Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and sum to 1

    Softmax is defined as
    :math:`f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)`
    where `shift = max_i x_i`
    """

    def forward(self, input):
        return softmax(input)
