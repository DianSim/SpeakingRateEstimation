import os
from inference import inference
from torchmetrics.regression import PearsonCorrCoef, MeanAbsoluteError
from torch import nn
import torch
import model
import time
import csv
import tqdm
import librosa
from config import config
import pickle


model = model.MatchBoxNetreg(B=3, R=2, C=112)
path = '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/SREregression/models_4sec/rMatchBoxNet-3x2x112/checkpoints/best-epoch=194-val_loss=4.60-val_pcc=0.95.ckpt'
state_dict = torch.load(path)
model.load_state_dict(state_dict['state_dict'])

model_name = path.split(os.sep)[-3]

data_dir = '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/CommonVoice/hy-AM/clips_wav_16khz_labeled'

labels_csyl = []
labels_sp_rate = []
preds_csyl = []
preds_sp_rate = [] 

header = True
corpuse_size = 0
for root, dirs, files in os.walk(data_dir):
    for file in tqdm.tqdm(files):
        # if corpuse_size == 100:
        #     break
        if file[-4:] == '.wav':
            corpuse_size += 1
            file_path = os.path.join(root, file)

            print('file path: ', file_path)

            start_time = time.time()
            pred = inference(model, audio_path=file_path)
            if pred is None:
                continue
            end_time = time.time()

            inference_time = end_time - start_time

            if pred['syl_count'].isnan() or pred['speaking_rate'].isnan():
                continue

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
            
# # save lists as pickle
# with open('labels_csyl.pkl', 'wb') as f:
#     pickle.dump(labels_csyl, f)

# with open('labels_sp_rate.pkl', 'wb') as f:
#     pickle.dump(labels_sp_rate, f)

# with open('preds_csyl.pkl', 'wb') as f:
#     pickle.dump(preds_csyl, f)

# with open('preds_sp_rate.pkl', 'wb') as f:
#     pickle.dump(preds_sp_rate, f)
             

MSE = nn.MSELoss()
MAE = MeanAbsoluteError()
PCC = PearsonCorrCoef()

labels_csyl = torch.tensor(labels_csyl, dtype=torch.float32)
labels_sp_rate = torch.tensor(labels_sp_rate, dtype=torch.float32)
preds_csyl = torch.stack(preds_csyl)
preds_sp_rate = torch.stack(preds_sp_rate)
   

mse_csyl = MSE(preds_csyl, labels_csyl)
mae_csyl = MAE(preds_csyl, labels_csyl)
mae_sp_rate = MAE(preds_sp_rate, labels_sp_rate)

pcc_csyl = PCC(preds_csyl, labels_csyl)
pcc_sp_rate = PCC(preds_sp_rate, labels_sp_rate)

print(f'mae csyl: {mae_csyl.item():.4f}')
print(f'mse csyl: {mse_csyl.item():.4f}')
print(f'pcc csyl: {pcc_csyl.item():.4f}')
print()

print(f'mae sp_rate: {mae_sp_rate.item():.4f}')
print(f'pcc sp_rate: {pcc_sp_rate.item():.4f}')



# save computed loss and metric in given file

language = 'Armenian'
with open("model_eval_res_common_voices2.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Corpus",'language', '#audios', "MAE_csyl", "MSE_csyl","PCC_csyl", "MAE_sp_rate", "PCC_sp_rate"])
    writer.writerow([model_name, "Common Voice", language, corpuse_size, f'{mae_csyl.item():.4f}',f'{mse_csyl.item():.4f}', f'{pcc_csyl.item():.4f}',
                                                                        f'{mae_sp_rate.item():.4f}', f'{pcc_sp_rate.item():.4f}'])



