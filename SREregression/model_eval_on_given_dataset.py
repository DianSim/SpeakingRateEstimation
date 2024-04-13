import os
from inference import inference
from torchmetrics.regression import PearsonCorrCoef
from torch import nn
import torch
import model
import time
import csv
import tqdm
import librosa
from config import config


model = model.MatchBoxNetreg(B=3, R=2, C=112)
path = '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/SREregression/models/rMatchBoxNet-3x2x112/checkpoints/best-epoch=198-val_loss=1.50-val_pcc=0.93.ckpt'
state_dict = torch.load(path)
model.load_state_dict(state_dict['state_dict'])

model_name = path.split(os.sep)[-3]

data_dir = '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/CommonVoice/ru/clips_wav_16khz_labeled'

labels_csyl = []
labels_sp_rate = []
preds_csyl = []
preds_sp_rate = []

header = True
corpuse_size = 0
for root, dirs, files in os.walk(data_dir):
    for file in tqdm.tqdm(files):
        if file[-4:] == '.wav':
            corpuse_size += 1
            file_path = os.path.join(root, file)

            print('file path: ', file_path)

            start_time = time.time()
            pred = inference(model, audio_path=file_path)
            end_time = time.time()

            inference_time = end_time - start_time

            preds_csyl.append(pred['syl_count'])
            preds_sp_rate.append(pred['speaking_rate'])

            csyl = int(file.split('.')[0].split('_')[-1])

            labels_csyl.append(csyl)

            audio, sr = librosa.load(file_path, sr=config['sample_rate'])
            audio_len = audio.shape[0]/sr

            labels_sp_rate.append(csyl/audio_len)
            
            print('label csyl: ', csyl)
            print('prediction csyl: ', pred['syl_count'])
            print('label sp_rate: ', csyl/audio_len)
            print('prediction sp_rate: ', pred['speaking_rate'])
            print('inference time: ', inference_time) 
            print()
            
            

MSE = nn.MSELoss()
PCC = PearsonCorrCoef()

labels_csyl = torch.tensor(labels_csyl, dtype=torch.float32)
labels_sp_rate = torch.tensor(labels_sp_rate, dtype=torch.float32)
preds_csyl = torch.stack(preds_csyl)
preds_sp_rate = torch.stack(preds_sp_rate)

mse_csyl = MSE(preds_csyl, labels_csyl)
mse_sp_rate = MSE(preds_sp_rate, labels_sp_rate)

pcc_csyl = PCC(preds_csyl, labels_csyl)
pcc_sp_rate = PCC(preds_sp_rate, labels_sp_rate)

print(f'mse csyl: {mse_csyl.item():.4f}')
print(f'pcc csyl: {pcc_csyl.item():.4f}')
print()

print(f'mse sp_rate: {mse_sp_rate.item():.4f}')
print(f'pcc sp_rate: {pcc_sp_rate.item():.4f}')


# save computed loss and metric in given file

language = 'Russian'
with open("model_eval_res_common_voices.csv", "a", newline="") as f:
    writer = csv.writer(f)
    # writer.writerow(["Model", "Corpus", "#audios", "Language", "MSE_csyl", "PCC_csyl", "MSE_sp_rate", "PCC_sp_rate"])
    writer.writerow([model_name, "Common Voice", corpuse_size, language, f'{mse_csyl.item():.4f}', f'{pcc_csyl.item():.4f}',
                                                                        f'{mse_sp_rate.item():.4f}', f'{pcc_sp_rate.item():.4f}'])



