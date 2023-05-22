import math
import torch.nn as nn
import torch.optim as optim
import transformers
from typing import List

from .utils import get_parameters_by_prefix

type2optimizer = {
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'adam': optim.Adam,
    'adamax': optim.Adamax,
    'adamw': optim.AdamW,
    'rmsprop': optim.RMSprop,
    'sgd': optim.SGD
}

def get_warmup_cosine_optimizer(
    module: nn.Module,
    optim_type: str,
    lr: float,
    warmup_steps: int,
    max_steps: int,
    n_cycles: int,
    **optim_kwargs
): 
    optimizer = type2optimizer[optim_type](module.parameters(), lr=lr, **optim_kwargs)
    scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
        num_cycles=n_cycles)  

    return optimizer, scheduler