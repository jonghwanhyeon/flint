import os
from functools import partial

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from flint.datamodule import SkeletonDataModule
from flint.experiments.skeleton import Skeleton

if __name__ == '__main__':
    datamodule = SkeletonDataModule(
        batch_size={
            'train': 48,
            'validation': 24,
            'test': 24,
        },
    )

    experiment = Skeleton(
        optimizer=partial(torch.optim.SGD, lr=0.001, momentum=0.9),
        lr_scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=128, gamma=0.98),
    )

    logger = TensorBoardLogger(save_dir=os.getcwd(), name='logs/skeleton')
    trainer = Trainer(
        gpus=[0], 
        num_sanity_val_steps=0,
        logger=logger,
    )
    trainer.fit(experiment, datamodule)
