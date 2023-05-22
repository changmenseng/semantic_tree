import importlib
from .registry import registry

modules = ('lstm', 'transformer')
for module in modules:
    importlib.import_module(f'.{module}', __name__)

def get_module(config, **kwargs):
    module_class = registry[config['type']]
    return module_class(**config['hparams'], **kwargs)