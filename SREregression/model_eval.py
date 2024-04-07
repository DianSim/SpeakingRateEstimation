#model_eval.py

import pytorch_lightning as pl
import data_setup, model
from utils import build_trainer, avg_loss_metric_from_batch_list, abs_error_list,save_histogram_stat
from config import config
from LSTM.dataset import collate_fn
import torch
import os


train_dir = config['train_dir']
val_dir = config['val_dir']
test_dir = config['test_dir']


train_dataloader, val_dataloader, test_dataloader = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    val_dir = val_dir,
    num_workers=12,
    batch_size=config['train_params']['batch_size'],
    # collate_fn = collate_fn
    )

model = model.MatchBoxNetreg(B=3, R=2, C=112)
# model = model.LSTMRegression()

#LOAD THE MODEL FROM CKPT

path = '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/SREregression/models/rMatchBoxNet-3x2x112/checkpoints/best-epoch=198-val_loss=1.50-val_pcc=0.93.ckpt'
state_dict = torch.load(path)
model.load_state_dict(state_dict['state_dict'])
model.eval()

trainer = build_trainer(config)
test_predictions = trainer.predict(model, test_dataloader)
val_predictions = trainer.predict(model, val_dataloader)
train_predictions = trainer.predict(model, train_dataloader)

test_avg_loss, test_avg_pcc = avg_loss_metric_from_batch_list(test_predictions)
val_avg_loss, val_avg_pcc = avg_loss_metric_from_batch_list(val_predictions)
train_avg_loss, train_avg_pcc = avg_loss_metric_from_batch_list(train_predictions)

print('avg_train_loss: ', train_avg_loss)
print('avg_train_pcc: ', train_avg_pcc)
print()

print('avg_val_loss: ', val_avg_loss)
print('avg_val_pcc: ', val_avg_pcc)
print()

print('avg_test_loss: ', test_avg_loss)
print('avg_test_pcc: ', test_avg_pcc)
print()

list_of_abs_errors = abs_error_list(test_predictions)
model_name = path.split(os.sep)[-3]
save_histogram_stat(list_of_abs_errors, os.path.join(config['model_error_dir'], model_name), 'hist_error.png')

# histogram sarqel
# 2 aragcrats data sarqel
# 3 classification u lstm porcel mnacac modelnery ,
# 4 exceli sheetum ardyunqnery pahel
# 5 write inference part