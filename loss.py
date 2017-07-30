
from module import Module

import numpy as np


class Loss(Module):

    def __init__(self, size_average=True):
        super(Loss, self).__init__()
        self.size_average = size_average


class MSELoss(Loss):
    r""" Creates a loss that measures the mean squared error between
    `n` elements in the input `x` and target `y`:

    :math:`{loss}(x, y)  = 1/n \sum |x_i - y_i|^2`
    """

    def forward(self, model, input, target):
        prediction = model.predict(input)
        return np.sum(np.power(prediction - target, 2)) / len(input)
