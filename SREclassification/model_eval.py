# model_eval.py
import pytorch_lightning as pl
import data_setup, model
from utils import build_trainer, avg_loss_metric_from_batch_list
from config import config
# from LSTM.dataset import collate_fn
import torch
from torch import nn



train_dir = config['train_dir']
val_dir = config['val_dir']
test_dir = config['test_dir']


train_dataloader, val_dataloader, test_dataloader = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    val_dir = val_dir,
    num_workers=50,
    batch_size=config['train_params']['batch_size'],
    # collate_fn = collate_fn
    )

model = model.MatchBoxNetclass(B=3, R=2, C=112)
# model = model.LSTMClassification()

#LOAD THE MODEL FROM CKPT

path = '/data/saten/diana/SpeakingRateEstimation/SREclassification/Models/cMatchBoxNet-3x2x112/checkpoints/best-epoch=101-val_loss=1.55-val_accuracy=0.34-val_top3_accuracy=0.80.ckpt'
state_dict = torch.load(path)
model.load_state_dict(state_dict['state_dict'])
model.eval()

trainer = build_trainer(config)
test_predictions = trainer.predict(model, test_dataloader)
val_predictions = trainer.predict(model, val_dataloader)
train_predictions = trainer.predict(model, train_dataloader)

test_avg_loss, test_avg_mse, test_avg_acc, test_avg_top3_acc = avg_loss_metric_from_batch_list(test_predictions)
val_avg_loss, val_avg_mse,  val_avg_acc, val_avg_top3_acc = avg_loss_metric_from_batch_list(val_predictions)
train_avg_loss, train_avg_mse, train_avg_acc, train_avg_top3_acc = avg_loss_metric_from_batch_list(train_predictions)

print('avg_train_loss: ', train_avg_loss)
print('avg_train_mse: ', train_avg_mse)
print('avg_train_acc: ', train_avg_acc)
print('avg_train_top3_acc: ', train_avg_top3_acc)
print()

print('avg_val_loss: ', val_avg_loss)
print('avg_val_mse: ', val_avg_mse)
print('avg_val_acc: ', val_avg_acc)
print('avg_val_top3_acc: ', val_avg_top3_acc)
print()

print('avg_test_loss: ', test_avg_loss)
print('avg_test_mse: ', test_avg_mse)
print('avg_test_acc: ', test_avg_acc)
print('avg_test_top3_acc: ', test_avg_top3_acc)

# print()

# 5 write inference part