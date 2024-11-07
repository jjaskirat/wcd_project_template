from typing import Callable

def load_from_import_str(name: str) -> Callable:
    """This function is used to import any class from a string.
    Ex: load_from_import_str('torch.nn.CrossEntropyLoss') will return the CrossEntropyLoss

    Args:
        name (str): the import string

    Returns:
        Callable: the class to be imported
    """
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod