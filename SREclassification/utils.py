# utils.py
"""
Contains various utility functions for PyTorch model training and saving.
"""
import matplotlib.pyplot as plt
import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import torch


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
                      devices=1,
                      accelerator="gpu",
                    #   gpus=[3], # [0,1,2,3]
                      logger=tensorboard_logger,
                      callbacks=[checkpoint_callback]
                      )
    return trainer

# {'loss':loss, 'accuracy':self.accuracy(y_hat, y), 'top3_accuracy':self.top3_accuracy(y_hat, y)}

def avg_loss_metric_from_batch_list(predictions):
    """
    Calculates the average loss and pcc from a list of batch predictions
    """
    total_loss = 0
    total_acc = 0
    total_top3_acc = 0
    total_mse = 0
    for pred in predictions:
        total_loss += pred['loss']
        total_acc += pred['accuracy']
        total_top3_acc += pred['top3_accuracy']
        total_mse += pred['mse']
    avg_loss = total_loss/len(predictions)
    avg_acc = total_acc/len(predictions)
    avg_top3_acc = total_top3_acc/len(predictions)
    avg_mse = total_mse/len(predictions)
    return avg_loss, avg_mse, avg_acc, avg_top3_acc


def save_as_img(x, dir, name):
    """
    Saves given matrice (MFCC) as image in the given folder
    """
    plt.imshow(x, cmap='viridis')
    plt.xlabel('time')
    plt.ylabel('frequ')
    plt.savefig(os.path.join(dir, name))
