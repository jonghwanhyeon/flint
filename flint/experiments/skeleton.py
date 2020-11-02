from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.metrics import Accuracy, Precision, Recall

from . import Experiment


class Skeleton(Experiment):
    hyperparameters = {
        'optimizer': partial(optim.SGD, lr=0.001, momentum=0.9),
        'lr_scheduler': partial(optim.lr_scheduler.StepLR, step_size=16, gamma=0.9),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = nn.Module()
        self.criterion = F.cross_entropy
        self.metrics = nn.ModuleDict({
            'accuracy': Accuracy(),
            'precision': Precision(),
            'recall': Recall(),
        })

    def training_step(self, batch, batch_index):
        inputs, actual = batch
        predicted = self.model(inputs)

        loss = self.criterion(predicted, actual)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_index):
        inputs, actual = batch
        predicted = self.model(inputs)
        
        metrics = {}
        for key, metric in self.metrics.items():
            metric(predicted, actual)
            metrics[f'validation_{key}'] = metric

        self.log_dict({
            'validation_loss': self.criterion(predicted, actual),
            **metrics},  on_epoch=True)
            
    def test_step(self, batch, batch_index):
        inputs, actual = batch
        predicted = self.model(inputs)

        metrics = {}
        for key, metric in self.metrics.items():
            metric(predicted, actual)
            metrics[f'test_{key}'] = metric

        self.log_dict({
            'test_loss': self.criterion(predicted, actual),
            **metrics},  on_epoch=True)

    def configure_optimizers(self):
        optimizer = self.hyperparameters['optimizer'](self.parameters())
        lr_scheduler = self.hyperparameters['lr_scheduler'](optimizer)
        
        return [optimizer], [lr_scheduler]
