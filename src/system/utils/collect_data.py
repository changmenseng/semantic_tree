import torch
import torch.distributed as dist

def collect(outputs):
    n_batchs = len(outputs)
    n_items = len(outputs[0])
    items = []
    for i in range(n_items):
        item = torch.cat([output[i] for output in outputs], 0)
        items.append(item)
    return items

def collect_from_other_rank(tensor, device, world_size=None):
    if world_size is None:
        world_size = dist.get_world_size()
    size = torch.tensor(tensor.shape[0], device=device)
    gather_sizes = [size.clone() for _ in range(world_size)]
    dist.all_gather(gather_sizes, size)

    max_size = max(gather_sizes)
    padding_size = (max_size - size).item()
    if padding_size > 0:
        # padding
        padding_tensor = torch.zeros(padding_size, *tensor.shape[1:], device=device)
        tensor = torch.cat([tensor, padding_tensor], 0)
    
    gather_tensors = [torch.zeros(max_size.item(), device=device, dtype=tensor.dtype) for _ in range(world_size)]
    dist.all_gather(gather_tensors, tensor)

    res = torch.cat([tensor[:size] for size, tensor in zip(gather_sizes, gather_tensors)], dim=0)

    return res