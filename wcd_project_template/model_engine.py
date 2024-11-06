from abc import ABC, abstractmethod
import time
import torch
from tqdm import tqdm
from typing import Any, Callable, Tuple

from .data import Data
from .model import Model
from .utils import load_from_import_str

class ModelEngine:
    def __init__(
        self,
    ):
        pass

    def train(self, model, data_train, data_valid, config_training, config_evaluation):
        """
        Training Loop
        """
        # model = self.model
        optimizer = self._get_optimizer(model.model)
        criterion = self._get_criterion()
        metric = self._get_metric()
        # df_train, df_val = self._get_df_splits()
        dataloader_train = data_train.get_dataloader()
        num_epochs = config_training['num_epochs']
        print("Total Training Samples: ", len(self.data_train))
        for epoch in tqdm(range(num_epochs)):
            train_start_time = time.monotonic()
            # Train one epoch
            train_loss, train_metric = self._train_one_epoch(model, dataloader_train, optimizer, criterion, metric)
            train_end_time = time.monotonic()
            val_start_time = time.monotonic()
            # Validate
            val_loss, val_metric = self.evaluate(model, data_valid, config_evaluation)
            val_end_time = time.monotonic()
            # Append loss and accuracy to class lists
            self.train_loss_list.append(train_loss)
            self.train_metric_list.append(train_metric)
            self.val_loss_list.append(val_loss)
            self.val_metric_list.append(val_metric)
            # Prints
            print("Epoch-%d: " % (epoch+1))
            print("Training: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds"\
                % (train_loss, train_metric, train_end_time - train_start_time))
            print("Validation: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds"\
                % (val_loss, val_metric, val_end_time - val_start_time))
            print("")
        self._save_model(model)
        print(f'model saved at: {self.save_dir}/{self.name}.pth')
        print("="*50)
        return model

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
            metric = metric(output, label)
            # Optimizing weights
            optimizer.step()
            epoch_loss += loss.item()
            epoch_metric += metric.item()
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
        metric = self._get_metric(config_evaluation)
        criterion = self._get_criterion(config_evaluation)
        dataloader = data_valid.get_dataloader()
        output, label = self._make_predictions(model, dataloader)
        val_metric = metric(output, label).item()
        val_loss = criterion(output, label).item()
        return val_loss, val_metric
    
    def _get_optimizer(self, model: Model) -> torch.optim.Optimizer:
        """returns the PyTorch Optimizer from config
        Parameters loaded are the model parameters

        Args:
            model (nn.Module): model to optimize

        Returns:
            optim: Optimizer
            Only from the library: PyTorch
        """
        optimizer_name = self.config_training['optimizer']['name']
        optimizer_config = self.config_training['optimizer']['config']
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
        metric_name = config['criterion']['name']
        metric_config = config['criterion']['config']
        metric_cls = load_from_import_str(metric_name)
        metric = metric_cls(**metric_config)
        return metric