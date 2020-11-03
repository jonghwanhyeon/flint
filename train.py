import os
from functools import partial

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from flint.datamodule import SkeletonDataModule
from flint.experiments.skeleton import Skeleton

identifier = 'skeleton'

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

    trainer = Trainer(
        gpus=[0], 
        num_sanity_val_steps=0,
        logger=TensorBoardLogger(save_dir=os.getcwd(), name=f'logs/{identifier}'),
        callbacks=[
            ModelCheckpoint(
                monitor='validation_loss',
                dirpath=f'checkpoints/{identifier}',
                filename='{epoch}-{step}-{validation_loss:.4f}',
            ),
        ]
    )
    trainer.fit(experiment, datamodule)
