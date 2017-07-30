#!/usr/bin/env python
# -*- coding: utf-8 -*-

from autograd import grad

import activation as F
from module import Module

import numpy as np

# Numerical stability Epsilon
epsilon = 0.0000000001

"""
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
"""


def linear(input, weight, bias=None):
    """Compute linear transformation with bias."""
    output = input.dot(weight)
    if bias is not None:
        output += bias
    return output


class Linear(Module):

    def __init__(self, weights, bias):
        """"""
        super().__init__()
        self.weights = weights
        self.bias = bias
        self.derivable = True

    @classmethod
    def random(cls, n_input, n_hidden):
        """"""
        return cls(weights=np.random.randn(n_input, n_hidden),
                   bias=np.zeros((1, n_hidden)))

    def forward(self, input):
        return linear(input, self.weights, self.bias)

    def backward(self, gradient, regularization=None):
        delta = (self.weights.T).dot(gradient)
        delta_bias = np.sum(delta, axis=0, keepdims=True)

        # Apply regularization on gradient
        if regularization:
            delta = regularization(delta)

        self.weights += -epsilon * delta
        self.bias += delta_bias


class Sequential(Module):

    def __init__(self, *args):
        """"""
        super(Sequential, self).__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def __call__(self, x):
        return self.forward(x)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def forward(self, input):
        for module in self._modules.values():
            input = module.forward(input)
        return input

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs, axis=1)
