import os
import shutil
import sys
from functools import partial

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from flint.datamodule import SkeletonDataModule
from flint.experiments.skeleton import Skeleton
from flint.utils import enter


def prepare_logger(name, version):
    logger = TensorBoardLogger(
        save_dir=os.getcwd(), name=f'logs/{name}', version=version,
        default_hp_metric=False,
    )

    if os.path.exists(logger.log_dir):
        print(f'Directory "{logger.log_dir}" exists')
        choice = enter('Do you want to remove the directory?', ('Y', 'n'))
        if choice == 'y':
            shutil.rmtree(logger.log_dir)
        else:
            sys.exit(1)

    return logger

if __name__ == '__main__':
    experiment = {
        'name': 'skeleton',
        'version': None,
    }
    
    datamodule = SkeletonDataModule(
        batch_size={
            'train': 48,
            'validation': 24,
            'test': 24,
        },
    )
    skeleton = Skeleton(
        optimizer=partial(torch.optim.SGD, lr=0.001, momentum=0.9),
        lr_scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=128, gamma=0.98),
    )

    logger = prepare_logger(**experiment)
    trainer = Trainer(
        gpus=[0], 
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                monitor='validation_loss',
                save_last=True,
                save_top_k=-1, # all models are saved
                dirpath=f'{logger.log_dir}/checkpoints',
                filename='{epoch}-{step}-{validation_loss:.4f}',
            ),
        ]
    )
    trainer.fit(skeleton, datamodule)