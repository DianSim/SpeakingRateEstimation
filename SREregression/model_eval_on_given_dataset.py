import os
from inference import inference
from torchmetrics.regression import PearsonCorrCoef
from torch import nn
import torch
import model
import time
import csv
import tqdm


model = model.MatchBoxNetreg(B=3, R=2, C=112)
path = '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/SREregression/models/rMatchBoxNet-3x2x112/checkpoints/best-epoch=198-val_loss=1.50-val_pcc=0.93.ckpt'
model_name = path.split(os.sep)[-3]

data_dir = '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/CommonVoice/hy-AM/clips_wav_16khz_labeled'

labels = []
preds = []

for root, dirs, files in os.walk(data_dir):
    for file in tqdm.tqdm(files):
        file_path = os.path.join(root, file)

        start_time = time.time()
        pred = inference(model, model_chckp=path, audio_path=file_path)
        end_time = time.time()

        inference_time = end_time - start_time
        preds.append(pred['syl_count'])

        label = int(file.split('.')[0].split('_')[-1])
        labels.append(label)

        print('label: ', file.split('.')[0].split('_')[-1])
        print('prediction: ', pred)
        print('inference time: ', inference_time) 
        print()

mse = nn.MSELoss()
pcc = PearsonCorrCoef()

labels = torch.tensor(labels, dtype=torch.float32)
preds = torch.stack(preds)
print(mse(preds, labels))
print(pcc(preds, labels))

# save computed loss and metric in given file

# language = 'Armenian'
# with open("model_eval_res_common_voices.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Model", "Corpus", "Language", "MSE", "PCC"])
