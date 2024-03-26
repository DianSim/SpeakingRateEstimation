"""
Contains various utility functions for PyTorch model training and saving.
"""
import matplotlib.pyplot as plt
import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os

def build_trainer(config):
    model_dir = os.path.join(config['model_dir'], config['model_name'])
    os.makedirs(model_dir, exist_ok=True)
    ckpt_dir = os.path.join(model_dir, 'checkpoints')
    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best-{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}-{val_top3_accuracy:.2f}',
        save_top_k=5,
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_last=True,
        every_n_epochs = 1
    )

    tensorboard_dir = os.path.join(model_dir, 'tensorboard_logs')
    tensorboard_logger = TensorBoardLogger(save_dir=tensorboard_dir)

    # Create a PyTorch Lightning trainer with the generation of logs disabled
    trainer = Trainer(max_epochs=config['train_params']['max_epochs'], 
                      gpus=[2],
                      logger=tensorboard_logger,
                      callbacks=[checkpoint_callback]
                      )
    return trainer

def save_as_img(x, dir, name):
    """
    Saves given matrice (MFCC) as image in the given folder
    """
    plt.imshow(x, cmap='viridis')
    plt.xlabel('time')
    plt.ylabel('frequ')
    plt.savefig(os.path.join(dir, name))