#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sigmoid(x):
    """"""
    return 1 / (1 + np.exp(-x))


class Threshold:

    def __init__(self, threshold, value):
        """ Thresholds each element of the input Tensor

        Threshold is defined as::

         y =  x        if x >  threshold
              value    if x <= threshold

        """
        self.threshold = threshold
        self.value = value

    def forward(self, input):
        return input * (input > self.threshold)


class ReLU(Threshold):

    def __init__(self):
        super(ReLU, self).__init__(0, 0)


class Sigmoid():
    """Applies the element-wise function,
    :math:`f(x) = 1 / ( 1 + exp(-x))`
    """

    def forward(self, input):
        return sigmoid(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


class Tanh():
    """Applies element-wise,
    :math:`f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
    """

    def forward(self, input):
        return np.tanh(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


class Softmax():
    """Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and sum to 1

    Softmax is defined as
    :math:`f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)`
    where `shift = max_i x_i`
    """

    def forward(self, input):
        return softmax(input)
