"""
Contains functionality for creating PyTorch DataLoaders for 
audio data speaking rate estimation task.
"""
import os
from torch.utils.data import DataLoader
from config import config
# from LSTM_regression.dataset import AudioDataset, collate_fn
# from LSTM_classification.dataset import AudioDataset, collate_fn
from MatchBoxNet_classification.dataset import AudioDataset

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
        train_dir: str, 
        test_dir: str, 
        val_dir: str,
        # transform: transforms.Compose,
        batch_size: int, 
        collate_fn: object=None,
        num_workers: int=NUM_WORKERS
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
    train_dataset = AudioDataset(data_dir=train_dir)
    val_dataset = AudioDataset(data_dir=val_dir)
    test_dataset = AudioDataset(data_dir=test_dir)

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
        num_workers=num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    train_dir = '/Users/dianasimonyan/Desktop/Thesis/torch_implementation/datasets/LibriSpeechChuncked_v2/train-clean-100'
    test_dir = '/Users/dianasimonyan/Desktop/Thesis/torch_implementation/datasets/LibriSpeechChuncked_v2/test-clean'
    val_dir = '/Users/dianasimonyan/Desktop/Thesis/torch_implementation/datasets/LibriSpeechChuncked_v2/dev-clean'

    # train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_dir=train_dir, 
    #                                                                        test_dir=test_dir, 
    #                                                                        val_dir=val_dir, 
    #                                                                        batch_size=16, 
    #                                                                        collate_fn=collate_fn)
    # print('train data count: ', len(train_dataloader))
    # print('val data count: ', len(val_dataloader))
    # print('test data count: ', len(test_dataloader))

    # for batch_seq, labels in train_dataloader:
    #     print(batch_seq.shape)
    #     print(labels)
    #     print()

    train_dir = '/Users/dianasimonyan/Desktop/Thesis/torch_implementation/datasets/LibriSpeechChuncked_v2/train-clean-100'
    train_dataset = AudioDataset(data_dir=train_dir)
    s = set()
    for x in train_dataset:
        print(x[0].shape)

    print(s)
