# train.py

import pytorch_lightning as pl
import data_setup, model, utils
from utils import build_trainer
from config import config
import torch
# from LSTM.dataset import collate_fn
from augmentation import NoiseAugmentation

# for MatchBoxNet change the model to MatchBoxNetreg, config['model_name'] to 'MatchBoxNetreg'
# and import correct datset and model


train_dir = config['train_dir']
val_dir = config['val_dir']
test_dir = config['test_dir']


train_dataloader, val_dataloader, test_dataloader = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    val_dir = val_dir,
    num_workers=12,
    batch_size=config['train_params']['batch_size'],
    # collate_fn = collate_fn,
    transform=NoiseAugmentation(noise_dir=config['noise_dir'])
)

model = model.MatchBoxNetreg(B=3, R=2, C=112)

# --------------------------------------------------------------SWA experiment----------------------------------------------------------------------
path = '/home/diana/Desktop/MyWorkspace/Project/SpeakingRateEstimation/SREregression/models_2sec/rMatchBoxNet-3x2x112/checkpoints/best-epoch=198-val_loss=1.50-val_pcc=0.93.ckpt'
state_dict = torch.load(path)
model.load_state_dict(state_dict['state_dict'])

#load model and 
trainer = build_trainer(config)
trainer.fit(model, train_dataloader, val_dataloader)