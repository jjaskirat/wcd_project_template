import matplotlib.pyplot as plt
import time
import torch
from tqdm import tqdm
from typing import Any, Callable, Tuple, Union, List

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

        model.model.to(self.device)
        optimizer = self._get_optimizer(model.model, config_training)
        criterion = self._get_criterion(config_training)
        metric = self._get_metric(config_training).to(self.device)
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
            print("Epoch-%d: " % (epoch+1))
            print("Training: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds"\
                % (loss_train, metric_train, end_time_train - start_time_train))
            print("Validation: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds"\
                % (loss_valid, metric_valid, end_time_valid - start_time_valid))
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
        model.model.to(self.device)
        metric = self._get_metric(config_evaluation).to(self.device)
        criterion = self._get_criterion(config_evaluation)
        dataloader = data_valid.get_dataloader()
        output, label = self._make_predictions(model, dataloader)
        val_metric = metric(output, label).item()
        val_loss = criterion(output, label).item()
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
        epoch_metric = 0
        # Train the model
        model = model.model
        model.train()
        for (input, label) in dataloader:
            input = input.to(self.device)
            label = label.to(self.device)
            # Training pass
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            # Backpropagation
            loss.backward()
            # Calculate metric
            metric_value = metric(output, label)
            # Optimizing weights
            optimizer.step()
            epoch_loss += loss.item()
            epoch_metric += metric_value.item()
        return epoch_loss / len(dataloader), epoch_metric / len(dataloader)
    
    def _make_predictions(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader
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
        # Evaluate the model
        model = model.model
        model.eval()
        with torch.no_grad():
            for (input, label) in dataloader:
                input = input.to(self.device)
                label = label.to(self.device)
                # Run predictions
                output = model(input)
                if all_outputs is not None:
                    all_outputs = torch.cat([all_outputs, output])
                else:
                    all_outputs = output
                if all_labels is not None:
                    all_labels = torch.cat([all_labels, label])
                else:
                    all_labels = label
        return all_outputs, all_labels
    
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
        criterion = criterion_cls(**criterion_config)
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
        metric = metric_cls(**metric_config)
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