import nemo.collections.asr as nemo_asr
import glob
import os
import soundfile as sf
import sounddevice as sd
import tqdm
from torch import nn
from torchmetrics.regression import PearsonCorrCoef, MeanAbsoluteError
import torch


def Eng_syl_computing(sentence):
    vowels = ['a', 'e', 'i', 'o', 'u', 'y']
    # unicode_eng_vowels = [1377, 1381, 1383, 1384, 1387, 1400, 1413]
    sentence = sentence.lower()
    syl_count = 0
    for s in sentence:
        if s in vowels:
            syl_count += 1
    return syl_count


def Ru_syl_computing(sentence):
    """The function takes Russion sentence as input and returns number of syllables in the sentence"""
    ru_vowels = ['а', 'я', 'у', 'ю', 'о', 'е', 'ё', 'э', 'и', 'ы']
    unicode_ru_vowels = [1072, 1103, 1091, 1102, 1086, 1077, 1105, 1101, 1080, 1099]
    
    sentence = sentence.lower()
    syl_count = 0
    for s in sentence:
        if ord(s) in unicode_ru_vowels:
            syl_count += 1
    return syl_count


def It_syl_computing(sentence):
    """The function takes Italian sentence as input and returns number of syllables in the sentence"""
    reg_vowels_codes = [ 0x0061, 0x0065, 0x0069, 0x006F , 0x0075]
    stressed_vowels_codes = [0x00E0, 0x00E8, 0x00E9, 0x00EC,0x00F2, 0x00F3, 0x00F9]
    reg_vowels = [chr(int(s)) for s in reg_vowels_codes]
    stressed_vowels = [chr(int(s)) for s in stressed_vowels_codes]
    
    sentence = sentence.lower()
    syl_count = 0
    for s in sentence:
        if (ord(s) in reg_vowels_codes) or (ord(s) in stressed_vowels_codes):
            syl_count += 1
    return syl_count


def Sp_syl_computing(sentence):
    """The function takes Spanish sentence as input and returns number of syllables in the sentence"""
    reg_vowels_codes= [int(0x0061), int(0x0065), int(0x0069), int(0x006F), int(0x0075)]
    stressed_vowles_codes = [int(0x00E1), int(0x00E9), int(0x00ED), int(0x00F3) , int(0x00FA), int(0x00FC)] 
    sp_reg_vowels = [chr(c) for c in reg_vowels_codes]
    sp_stressed_vowels = [chr(c) for c in stressed_vowles_codes]
    
    sentence = sentence.lower()
    syl_count = 0
    for s in sentence:
        if (ord(s) in reg_vowels_codes) or (ord(s) in stressed_vowles_codes):
            syl_count += 1
    return syl_count


def eval(asr_model, file_pathes, lngu):
    """
    lngu: language of model (['eng'|'arm'|'ru'|'es'|'it'])
    """
    labels_csyl = []
    labels_sp_rate = []
    preds_csyl = []
    preds_sp_rate = []
    
    for file in tqdm.tqdm(file_pathes):
        # print('audio path: ', file)
        
        csyl = int(file.split(os.sep)[-1].split('.')[0].split('_')[-1])
        labels_csyl.append(csyl)
        # print('csyl: ', csyl)
    
        audio, sr = sf.read(file, dtype='float32')
        audio_len = audio.shape[0]/sr
        sp_rate = csyl/audio_len
        labels_sp_rate.append(sp_rate)
        # print('sp_rate: ', sp_rate)
        transcript = asr_model.transcribe(file)

        if lngu == 'eng':
            pred_csyl = Eng_syl_computing(transcript[0][0])
        elif lngu == 'arm':
            pred_csyl = AM_syl_computing(transcript[0][0])
        elif lngu == 'ru':
            pred_csyl = Ru_syl_computing(transcript[0][0])
        elif lngu == 'es':
            pred_csyl = Sp_syl_computing(transcript[0][0])
        elif lngu == 'it':
            pred_csyl = It_syl_computing(transcript[0][0])
            
        preds_csyl.append(pred_csyl)
        pred_sp_rate = pred_csyl/audio_len
        preds_sp_rate.append(pred_sp_rate)
        
    return labels_csyl, labels_sp_rate, preds_csyl, preds_sp_rate
        


# datasets
libri_data_dir = '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/LibriSpeech/test-clean-labeled'
it_data_dir = '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/CommonVoice/it/clips_wav_16khz_labeled'

# models
eng_asr_model = nemo_asr.models.ASRModel.from_pretrained("stt_en_conformer_transducer_small")
it_asr_model = nemo_asr.models.ASRModel.from_pretrained("stt_it_quartznet15x5")
ru_asr_model = nemo_asr.models.ASRModel.from_pretrained("stt_ru_quartznet15x5")



file_paths = glob.glob(it_data_dir + "/**/*.wav", recursive=True)
print('len of the corpus: ', len(file_paths))

labels_csyl, labels_sp_rate, preds_csyl, preds_sp_rate = eval(it_asr_model, file_pathes, 'it')

# save lists as pickle
with open('it_labels_csyl.pkl', 'wb') as f:
    pickle.dump(labels_csyl, f)

with open('it_labels_sp_rate.pkl', 'wb') as f:
    pickle.dump(labels_sp_rate, f)

with open('it_preds_csyl.pkl', 'wb') as f:
    pickle.dump(preds_csyl, f)

with open('it_preds_sp_rate.pkl', 'wb') as f:
    pickle.dump(preds_sp_rate, f)


MSE = nn.MSELoss()
MAE = MeanAbsoluteError()
PCC = PearsonCorrCoef()

labels_csyl = torch.tensor(labels_csyl, dtype=torch.float32)
labels_sp_rate = torch.tensor(labels_sp_rate, dtype=torch.float32)
preds_csyl = torch.tensor(preds_csyl, dtype=torch.float32)
preds_sp_rate = torch.tensor(preds_sp_rate, dtype=torch.float32)

mae_csyl = MAE(preds_csyl, labels_csyl)
mse_csyl = MSE(preds_csyl, labels_csyl)
mae_sp_rate = MAE(preds_sp_rate, labels_sp_rate)

pcc_csyl = PCC(preds_csyl, labels_csyl)
pcc_sp_rate = PCC(preds_sp_rate, labels_sp_rate)

print(f'mae csyl: {mae_csyl.item():.4f}')
print(f'mse csyl: {mse_csyl.item():.4f}')
print(f'pcc csyl: {pcc_csyl.item():.4f}')
print()

print(f'mae sp_rate: {mae_sp_rate.item():.4f}')
print(f'pcc sp_rate: {pcc_sp_rate.item():.4f}')

language = 'Spanish'
with open("nemo_eval_res_common_voices.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Corpus",'language', '#audios', "MAE_csyl", "MSE_csyl","PCC_csyl", "MAE_sp_rate", "PCC_sp_rate"])
    writer.writerow([model_name, "Common Voice", language, corpuse_size, f'{mae_csyl.item():.4f}',f'{mse_csyl.item():.4f}', f'{pcc_csyl.item():.4f}',
                                                                        f'{mae_sp_rate.item():.4f}', f'{pcc_sp_rate.item():.4f}'])
