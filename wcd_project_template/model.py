import torch

from wcd_project_template.utils import load_from_import_str

class Model(torch.nn.Module):
    """The Model class is a Wrapper over the torch Module created.
    """
    def __init__(self, model_or_model_cls, config_model):
        super(Model, self).__init__()
        self.config_model = config_model
        if isinstance(model_or_model_cls, torch.nn.Module):
            self.model = model_or_model_cls
        else:
            self.model = model_or_model_cls(**self.config_model['model_hyperparameters'])
        
        self.load_model_weights(self.config_model['weights_path'])
        self.freeze_layers(self.config_model['freeze_layers'])
        
        print('done loading model')
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def load_model_weights(self, weights_path):
        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))
    
    def freeze_layers(self, freeze_layers):
        for name, param in self.named_parameters():
            if name in freeze_layers:
                param.requires_grad = False