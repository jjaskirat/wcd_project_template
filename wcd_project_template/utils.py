import importlib

def load_from_import_str(import_str):
    model = importlib.import_module(import_str)
    return model