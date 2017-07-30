from collections import OrderedDict
from variable import Variable


class Module:

    def __init__(self):
        """ Base class for all neural network modules.

        Your models should also subclass this class.
        """
        self._modules = OrderedDict()
        self._parameters = OrderedDict()

    def add_module(self, name, module):
        """Adds a child module to the current module.
        The module can be accessed as an attribute using the given name."""
        if hasattr(self, name):
            raise KeyError("attribute already exists '{}'".format(name))
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                module))
        self._modules[name] = module

    def children(self):
        """Returns an iterator over immediate children modules."""
        for name, module in self.named_children():
            yield module

    def named_children(self):
        """Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself."""
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def parameters(self):
        """Returns an iterator over module parameters.
        This is typically passed to an optimizer."""
        for name, param in self.named_parameters():
            yield param

    def named_parameters(self, memo=None, prefix=''):
        """ Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself."""
        if memo is None:
            memo = set()
        for name, p in self._parameters.items():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p
        for mname, module in self.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in module.named_parameters(memo, submodule_prefix):
                yield name, p

    def zero_grad(self):
        """Sets gradients of all model parameters to zero."""
        for p in self.parameters():
            if p.grad is not None:
                if p.grad.volatile:
                    p.grad.data.zero_()
                else:
                    data = p.grad.data
                    p.grad = Variable(data.new().resize_as_(data).zero_())
