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
        filename='best-{epoch:02d}-{val_loss:.2f}-{val_pcc:.2f}',
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
                      gpus=[1], # [0,1,2,3]
                      logger=tensorboard_logger,
                      callbacks=[checkpoint_callback]
                      )
    return trainer


def avg_loss_metric_from_batch_list(predictions):
    """
    Calculates the average loss and pcc from a list of batch predictions
    """
    total_loss = 0
    total_pcc = 0
    for pred in predictions:
        total_loss += pred['loss']
        total_pcc += pred['pcc']
    avg_loss = total_loss/len(predictions)
    avg_pcc = total_pcc/len(predictions)
    return avg_loss, avg_pcc


def abs_error_list(predictions):
    """calculates the list of absolute errors from a list of batch predictions"""
    list_of_abs_errors = []
    for batch_pred in predictions:
        abs_error = torch.abs(batch_pred['y_hat']-batch_pred['y'])
        list_of_abs_errors += abs_error.squeeze().tolist()

    return list_of_abs_errors

def save_as_img(x, dir, name):
    """
    Saves given matrice (MFCC) as image in the given folder
    """
    plt.imshow(x, cmap='viridis')
    plt.xlabel('time')
    plt.ylabel('frequ')
    plt.savefig(os.path.join(dir, name))


def save_histogram_stat(errors, dir, name):
    """
    Saves histogram of errors in the given folder
    and writes the statistics in a text file
    """
    os.makedirs(dir, exist_ok=True)
    plt.hist(errors, bins=int(max(errors)-min(errors)))
    plt.xlabel('error')
    plt.ylabel('frequency')
    plt.savefig(os.path.join(dir, name))

    with open(os.path.join(dir,f'stat.txt'), 'w') as file:
        file.write(f"max: {max(errors)}\n")
        file.write(f"min: {min(errors)}\n")
        file.write(f"mean: {sum(errors)/len(errors)}\n")
        file.write(f"median: {sorted(errors)[len(errors)//2]}\n")
        file.write(f"std: {(sum([(x - sum(errors)/len(errors))**2 for x in errors])/(len(errors)-1))**0.5}\n")

