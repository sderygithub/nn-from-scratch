
from parameter import Parameter

import numpy as np


class Tensor(Parameter):

    def __init__(self, data):
        super(Tensor, self).__init__(self,
                                     data=data,
                                     requires_grad=True)

    @classmethod
    def random(cls, shape):
        data = np.random.randn(shape[0], shape[1])

    @property
    def T(self):
        pass
