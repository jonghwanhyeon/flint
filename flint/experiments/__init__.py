from pytorch_lightning import LightningModule

from flint.utils import merge_dict


class Experiment(LightningModule):
    hyperparameters = {}

    def __init__(self, **kwargs):
        super().__init__()
        self.hyperparameters = merge_dict(self.hyperparameters, kwargs)
