

class Variable:
    """A kind of Variable that is to be considered a module parameter.

    Args:
        data (Tensor):
            Parameter tensor.

        requires_grad (bool, optional):
            If the parameter requires gradient.
    """

    def __init__(self, data, requires_grad=True):
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
