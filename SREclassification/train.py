import os
import torch
import data_setup, engine, model, utils
from config import config

# from LSTM_regression.dataset import collate_fn
from LSTM.dataset import collate_fn


train_dir = '/Users/dianasimonyan/Desktop/Thesis/torch_implementation/datasets/LibriSpeechChuncked_v2/train-clean-100'
val_dir = '/Users/dianasimonyan/Desktop/Thesis/torch_implementation/datasets/LibriSpeechChuncked_v2/dev-clean'
test_dir = '/Users/dianasimonyan/Desktop/Thesis/torch_implementation/datasets/LibriSpeechChuncked_v2/test-clean'

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, val_dataloader, test_dataloader = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    val_dir = val_dir,
    batch_size=config['train_params']['batch_size'],
    collate_fn = collate_fn
)

model = model.LSTMClassification().to(device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.001)

engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=val_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=config['train_params']['epochs'],
             device=device)