import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from flint.utils import merge_dict


class DataModule(LightningDataModule):
    hyperparameters = {}

    def __init__(self, **kwargs):
        super().__init__()
        self.hyperparameters = merge_dict(self.hyperparameters, kwargs)


class SkeletonDataModule(DataModule):
    hyperparameters = {
        'batch_size': {
            'train': 256,
            'validation': 128,
            'test': 128}
    }
    
    def __init__(self, **kwargs):
        super().__init__()
        self.hyperparameters = merge_dict(self.hyperparameters, kwargs)

    def prepare_data(self):
        # Use this method to do things that might write to disk 
        # or that need to be done only from a single GPU in distributed settings
        # - download
        # - tokenize
        pass

    def setup(self, stage=None):
        # There are also data operations you might want to perform on every GPU. Use setup to do things like:
        # - count number of classes
        # - build vocabulary
        # - perform train/val/test splits
        pass

    def train_dataloader(self):
        self.train_dataset = Dataset()

        return DataLoader(self.train_dataset,
                          batch_size=self.hyperparameters['batch_size']['train'],
                          num_workers=torch.get_num_threads())

    def val_dataloader(self):
        self.validation_dataset = Dataset()

        return DataLoader(self.validation_dataset,
                          batch_size=self.hyperparameters['batch_size']['validation'],
                          num_workers=torch.get_num_threads())

    def test_dataloader(self):
        self.test_dataset = Dataset()

        return DataLoader(self.test_dataset,
                          batch_size=self.hyperparameters['batch_size']['test'],
                          num_workers=torch.get_num_threads())
