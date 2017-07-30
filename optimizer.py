import numpy as np


class Optimizer:

    def __init__(self):
        """"""
        self.reg_lambda = 0.01
        self.learning_rate = 0.01

    def train(self, model, x, y):
        """
        @TODO Regularization needs to be abstracted away
        @TODO Learning rate should be part of a strategy (e.g. annealing)
        @TODO Batching of data
        @TODO Data epochs
        """
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Variable of input data to the Module and it produces
        # a Variable of output data.
        z1 = model[0].forward(x)
        a1 = model[1].forward(z1)
        z2 = model[2].forward(a1)
        probs = model[3].forward(z2)

        # Compute and print loss. We pass Variables containing the predicted and true
        # values of y, and the loss function returns a Variable containing the loss.
        # loss = loss_fn(y_pred, y)
        delta3 = probs
        delta3[range(len(x)), y] -= 1

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Variables with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(model[2].weights.T) * (1 - np.power(a1, 2))
        dW1 = (x.T).dot(delta2)
        db1 = np.sum(delta2, axis=0)

        # Update the weights using gradient descent. Each parameter is a Variable, so
        # we can access its data and gradients like we did before.
        dW1 += self.reg_lambda * model[0].weights
        dW2 += self.reg_lambda * model[2].weights

        # Gradient descent parameter update
        model[0].weights -= self.learning_rate * dW1
        model[0].bias -= self.learning_rate * db1
        model[2].weights -= self.learning_rate * dW2
        model[2].bias -= self.learning_rate * db2
