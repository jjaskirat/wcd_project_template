import torch

from wcd_project_template.utils import load_from_import_str

class Model():
    """The Model class is a Wrapper over the torch Module created.
    """
    def __init__(self, model_cls, config_model):
        self.config_model = config_model
        if isinstance(model_cls, torch.nn.Module):
            self.model = model_cls
        else:
            self.model = model_cls(**self.config_model['model_hyperparameters'])
        
        if self.config_model['weights_path'] is not None:
            self.model = self.load_model_weights(self.model)

        if len(self.config_model['freeze_layers']):
            self.model = self.freeze_model(self.model)
        print('done loading model')

    def load_model_weights(self, model):
        state_dict_path = self.config_model['weights_path']
        model.load_state_dict(torch.load(state_dict_path))
        return model
    
    # def freeze_model(self, model):
    #     for name, param in model.named_paramaters():
    #         if ()