import matplotlib.pyplot as plt
import time
import torch
from tqdm import tqdm
from typing import Any, Callable, Tuple, Union, List, Dict

from wcd_project_template.data import Data
from wcd_project_template.model import Model
from wcd_project_template.utils import load_from_import_str

class ModelEngine:
    def __init__(
        self,
        device
    ):
        self.device = device

    def train(
            self,
            model: Model,
            data_train: Data,
            data_valid: Data,
            config_training: dict,
            config_evaluation: dict,
        ) -> Model:
        """_summary_

        Args:
            model (Model): the model to be trained
            data_train (Data): training data
            data_valid (Data): validation data
            config_training (dict): training configuration
            config_evaluation (dict): evaluation configuration

        Returns:
            Model: the trained model
        """
        training_meta_data = {
            'loss_train': [],
            'loss_valid': [],
            'metric_train': [],
            'metric_valid': [],
        }

        model.to(self.device)
        optimizer = self._get_optimizer(model, config_training)
        criterion = self._get_criterion(config_training)
        metric = self._get_metric(config_training)
        metric = metric.to(self.device) if metric else metric
        # df_train, df_val = self._get_df_splits()
        dataloader_train = data_train.get_dataloader()
        num_epochs = config_training['num_epochs']
        print("Total Training Samples: ", len(data_train))
        for epoch in tqdm(range(num_epochs)):
            start_time_train = time.monotonic()
            # Train one epoch
            loss_train, metric_train = self._train_one_epoch(model, dataloader_train, optimizer, criterion, metric)
            end_time_train = time.monotonic()
            start_time_valid = time.monotonic()
            # Validate
            loss_valid, metric_valid = self.evaluate(model, data_valid, config_evaluation)
            end_time_valid = time.monotonic()
            # Append loss and metric to training_meta_data
            training_meta_data['loss_train'].append(loss_train)
            training_meta_data['loss_valid'].append(loss_valid)
            training_meta_data['metric_train'].append(metric_train)
            training_meta_data['metric_valid'].append(metric_valid)
            # Prints
            print(f"Epoch-{epoch + 1}: ===========================================")
            print(f"Training: ===========\
            \nLoss: {loss_train},\
            \nMetrics: {metric_train},\
            \nTime: {end_time_train - start_time_train} seconds")
            print(f"Validation: =========\
            \nLoss = {loss_valid},\
            \nMetrics: {metric_valid},\
            \nTime: {end_time_valid - start_time_valid} seconds")
            print("")
        # self._save_model(model)
        # print(f'model saved at: {self.save_dir}/{self.name}.pth')
        print("="*50)
        return model, training_meta_data

    def evaluate(
        self,
        model: Model,
        data_valid: Data,
        config_evaluation: dict
        ) -> Tuple[float]:
        """_summary_

        Args:
            model (nn.Module): _description_
            loader (DataLoader): _description_
            criterion (Any): _description_

        Returns:
            Tuple[float]: _description_
        """
        model.to(self.device)
        metric = self._get_metric(config_evaluation).to(self.device)
        criterion = self._get_criterion(config_evaluation)
        dataloader = data_valid.get_dataloader()
        output, label = self._make_predictions(model, dataloader)
        
        val_metric = metric(output, label) if metric is not None else torch.Tensor([0])
        val_loss = criterion(output, label) if criterion is not None else torch.Tensor([0])
        val_metric = self.to_numpy_obj(val_metric)
        val_loss = self.to_numpy_obj(val_loss)

        return val_loss, val_metric
    
    def _train_one_epoch(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: Callable,
        criterion: Callable,
        metric: Callable,
        ):
        """Trains the model for one epoch
        You can customize this method according to your liking

        Args:
            model (nn.Module): model to train
            dataloader (DataLoader): train dataloader
            optimizer (Any)
            criterion (Any)

        Returns:
            epoch loss and epoch accuracy
        """
        epoch_loss = 0
        epoch_metric = {}
        # Train the model
        # model = model.model
        model.train()
        for (input, label) in dataloader:
            input, label = self.preprocess_data_for_training(input, label)
            # Training pass
            optimizer.zero_grad()
            try:
                output = model(input)
            except:
                output = model(*input)
            output = self.postprocess_output_after_training(output)
            loss = criterion(output, label) if criterion is not None else torch.Tensor([0])
            # Backpropagation
            loss.backward()
            # Calculate metric
            metric_value = metric(output, label) if metric is not None else torch.Tensor([0])
            if not isinstance(metric_value, dict):
                metric_value = {
                    type(metric): metric_value
                }
            # Optimizing weights
            optimizer.step()
            epoch_loss += loss.item()
            metric_value = self.to_numpy_obj(metric_value)
            epoch_metric = self.update_metric(epoch_metric, metric_value)
        return (
            epoch_loss / len(dataloader),
            {k: v / len(dataloader) for k, v in epoch_metric.items()}
        )
    
    def _make_predictions(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        return_input: bool = False,
        ) -> Tuple[torch.Tensor]:
        """Makes the prediction of model on dataloader

        Args:
            model (nn.Module): model to make predictions
            dataloader (DataLoader): dataloader to make predictions on

        Returns:
            Tuple[torch.Tensor]: all_outputs and all_labels
        """
        all_labels = None
        all_outputs = None
        all_inputs = None
        # Evaluate the model
        # model = model.model
        model.eval()
        with torch.no_grad():
            for (input, label) in dataloader:
                input, label = self.preprocess_data_for_prediction(input, label)
                # Run predictions
                output = model(input)
                output = self.postprocess_output_after_prediction(output)
                all_outputs = self.append_new_value(all_outputs, output)
                all_labels = self.append_new_value(all_labels, label)
                all_inputs = self.append_new_value(all_inputs, input)
        
        if return_input:
            return all_outputs, all_labels, all_inputs
        else:
            return all_outputs, all_labels
    
    def preprocess_data_for_training(self, input, label):
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def postprocess_output_after_training(self, output):
        return output
    
    def preprocess_data_for_prediction(self, input, label):
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def postprocess_output_after_prediction(self, output):
        return output
    
    def append_new_value(self, entire_list, new_value):
        if entire_list is None:
            entire_list = new_value
        elif isinstance(entire_list, torch.Tensor):
            entire_list = torch.cat([entire_list, new_value])
        elif isinstance(entire_list, list):
            entire_list += new_value
        else:
            raise NotImplementedError
        
        return entire_list
    
    def update_metric(self, epoch_metric, metric_value):
        if epoch_metric == {}:
            for k, v in metric_value.items():
                epoch_metric[k] = v
        else:
            for k, v in metric_value.items():
                epoch_metric[k] += v
        return epoch_metric
    
    @staticmethod
    def _get_optimizer(model: Model, config: dict) -> torch.optim.Optimizer:
        """returns the PyTorch Optimizer from config
        Parameters loaded are the model parameters

        Args:
            model (nn.Module): model to optimize

        Returns:
            optim: Optimizer
            Only from the library: PyTorch
        """
        optimizer_name = config['optimizer']['name']
        optimizer_config = config['optimizer']['config']
        optimizer_cls = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_cls(
                        params = model.parameters(),
                        **optimizer_config
                    )
        return optimizer
    
    @staticmethod
    def _get_criterion(config) -> Callable:
        """returns the criterion loaded from training config

        Returns:
            Any: Criterion
            Generally from the library: PyTorch
            Implementing a Custom Criterion: https://discuss.pytorch.org/t/custom-loss-functions/29387/2
        """
        criterion_name = config['criterion']['name']
        criterion_config = config['criterion']['config']
        criterion_cls = load_from_import_str(criterion_name)
        criterion = criterion_cls(**criterion_config) if criterion_cls is not None else None
        return criterion
    
    @staticmethod
    def _get_metric(config) -> Callable:
        """returns the metric loaded from training config

        Returns:
            Any: Metric.
            Generally from the library: TorchMetrics
            Implementing a Custom Metric: https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
        """
        metric_name = config['metric']['name']
        metric_config = config['metric']['config']
        metric_cls = load_from_import_str(metric_name)
        metric = metric_cls(**metric_config) if metric_cls is not None else None
        return metric
    
    def plot(self, meta_data: dict, to_plot: Union[str, List]):
        if isinstance(to_plot, str):
            plt.plot(meta_data[to_plot], label=to_plot)
        else:
            for to_plot_sub in to_plot:
                plt.plot(meta_data[to_plot_sub], label=to_plot_sub)
        plt.title('Meta Data Plot')
        plt.legend()
        plt.show()
        
    def to_numpy_obj(self, torch_obj: Union[torch.Tensor, Dict, List]):
        if isinstance(torch_obj, dict):
            numpy_obj = {k: v.detach().cpu().numpy() for k,v in torch_obj.items()}
        elif isinstance(torch_obj, list):
            numpy_obj = [i.detach().cpu().numpy() for i in torch_obj]
        else:
            numpy_obj = torch_obj.detach().cpu().numpy()
        return numpy_obj