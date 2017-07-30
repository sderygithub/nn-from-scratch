
from variable import Variable


class Parameter(Variable):
    """A kind of Variable that is to be considered a module parameter.

    Args:
        data (Tensor):
            Parameter tensor.

        requires_grad (bool, optional):
            If the parameter requires gradient.
    """
    def __new__(cls, data=None, requires_grad=True):
        return super(Parameter, cls).__init__(cls,
                                              data=data,
                                              requires_grad=requires_grad)

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()
