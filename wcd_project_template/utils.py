from typing import Callable
from pydoc import locate

import torch
import torch.distributed as dist

def load_from_import_str(name: str) -> Callable:
    """This function is used to import any class from a string.
    Ex: load_from_import_str('torch.nn.CrossEntropyLoss') will return the CrossEntropyLoss

    Args:
        name (str): the import string

    Returns:
        Callable: the class to be imported
    """
    # components = name.split('.')
    # mod = __import__(components[0])
    # for comp in components[1:]:
    #     mod = getattr(mod, comp)
    # return mod
    if name is None:
        return None
    if isinstance(name, str):
        mod = locate(name)
    else:
        mod = name
        return mod
    return mod

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict