"""
Contains functionality for creating PyTorch DataLoaders for
audio data speaking rate estimation task.
"""
import os
from torch.utils.data import DataLoader
from config import config
# from LSTM.dataset import AudioDataset, collate_fn
from MatchBoxNet.dataset import AudioDataset

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        val_dir: str,
        batch_size: int,
        num_workers: int,
        collate_fn: object=None,
        transform: object=None
    ):
    """Creates training, testing and validation DataLoaders.

    Takes in a training, testing and validation directory pathes and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    val_dir: Path to validation directory.
    batch_size: Number of samples per batch in each of the DataLoaders.
    collate_fn: function to collate lists of samples into variable length batches
    num_workers: An integer for number of workers per DataLoader.

    Returns:
    A tuple of (train_dataloader, val_dataloader, test_dataloader).
    Example usage:
        train_dataloader, val_dataloader, test_dataloader = \
        = create_dataloaders(train_dir=path/to/train_dir,
                                test_dir=path/to/test_dir,
                                collate_fn=collate_fn,
                                batch_size=32,
                                num_workers=4)
    """
    train_dataset = AudioDataset(data_dir=train_dir, transform=transform)
    val_dataset = AudioDataset(data_dir=val_dir)
    test_dataset = AudioDataset(data_dir=test_dir)

    print('train data size: ', len(train_dataset))
    print('val data size: ', len(val_dataset))
    print('test data size: ', len(test_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, test_dataloader