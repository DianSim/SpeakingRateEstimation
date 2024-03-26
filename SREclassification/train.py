import pytorch_lightning as pl
import data_setup, model, utils
from utils import build_trainer
from config import config
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
    num_workers=50,
    batch_size=config['train_params']['batch_size'],
    # collate_fn = collate_fn,
    transform=NoiseAugmentation(noise_dir=config['noise_dir'])
)

model = model.MatchBoxNetclass(B=6, R=2, C=64)

# for x in train_dataloader:
#     print(model(x[0]).shape)
#     break

# model = model.LSTMClassification()

# for x in train_dataloader:
#     print(x[1].dtype)
#     y = model(x[0])
#     print(y.dtype)
#     print(x[1].shape)
#     print(y.shape)
#     break

trainer = build_trainer(config)
trainer.fit(model, train_dataloader, val_dataloader)