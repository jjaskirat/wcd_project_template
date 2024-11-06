import importlib

def load_model_from_import_str(self, import_str):
    model = importlib.import_module(import_str)
    return model