import torch.nn as nn

class Registry(dict):

    def __setitem__(self, key, value):
        assert issubclass(value, nn.Module)
        assert key not in self
        if key is None: key = value.__name__
        super().__setitem__(key, value)
    
    def register(self, key=None):
        def _register(value):
            self[key] = value
            return value
        return _register

registry = Registry()