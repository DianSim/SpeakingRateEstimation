import time
import librosa
from config import config
import torch
import torchaudio.transforms as T
import model
import nemo.collections.asr as nemo_asr


config_feature = config['feature']


def Eng_syl_computing(sentence):
    vowels = ['a', 'e', 'i', 'o', 'u', 'y']
    # unicode_eng_vowels = [1377, 1381, 1383, 1384, 1387, 1400, 1413]
    sentence = sentence.lower()
    syl_count = 0
    for s in sentence:
        if s in vowels:
            syl_count += 1
    return syl_count


def our_model_inf_time(audio_file):
    Model = model.MatchBoxNetreg(B=3, R=2, C=112)
    path2 = '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/SREregression/models_2sec/rMatchBoxNet-3x2x112/checkpoints/best-epoch=198-val_loss=1.50-val_pcc=0.93.ckpt'

    state_dict = torch.load(path2)
    Model.load_state_dict(state_dict['state_dict'])

    mfcc_transform =  T.MFCC(
        sample_rate=config['sample_rate'],
        n_mfcc=config_feature['mfcc_num_mel_bins'], # ??self.fft_length//2 + 1
        melkwargs={
            "n_fft": config_feature['window_size_ms']*config['sample_rate']//1000, # length of the FFT window
            "n_mels": config_feature['mfcc_num_mel_bins'],
            "hop_length": config_feature['window_stride']*config['sample_rate']//1000,
            "mel_scale": "htk",
        }
    )

    # -----------------------data preprocessing-----------------------
    start_time = time.time()

    audio, sr = librosa.load(audio_file, sr=config['sample_rate'])

    audio = torch.from_numpy(audio)
    mean = torch.mean(audio)
    std = torch.std(audio)

    audio = audio - mean
    audio = audio / std

    chunks = list(torch.split(audio, config['input_len']))

    # pad last chunck with 0s if necessary
    if chunks[-1].shape[0] < config['input_len']:
        pad_size = config['input_len'] - chunks[-1].shape[0]
        last_chunk = torch.nn.functional.pad(chunks[-1], pad=(0, pad_size))
        chunks[-1] = last_chunk

    chunck_mfccs = [mfcc_transform(chunk) for chunk in chunks]
    batch_of_chuncks = torch.stack(chunck_mfccs, dim=0)
    
    Model.eval()
    with torch.no_grad():
        pred = Model(batch_of_chuncks)
        pred = pred.sum()

    end_time = time.time()

    inference_time = end_time - start_time

    return inference_time

    # print('inference time rMatchBoxNet-3x2x112: ', our_inference_time)


def nemo_inference_time(audio_file):

    eng_asr_model = nemo_asr.models.ASRModel.from_pretrained("stt_en_conformer_transducer_small")

    start_time = time.time()
    transcript = eng_asr_model.transcribe(audio_file)
    pred_csyl = Eng_syl_computing(transcript[0][0])
    end_time = time.time()

    inference_time = end_time-start_time
    return inference_time


audio_path = '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/LibriSpeech/test-clean-labeled/1221/135766/11_91.wav'
audio, sr = librosa.load(audio_path, sr=config['sample_rate'])

matchBox_inf_time = our_model_inf_time(audio_path)
nemo_inf_time = nemo_inference_time(audio_path)
print('audio length (sec): ', audio.shape[0]/ sr)
print(f'rMatchBoxNet-3x2x112: {matchBox_inf_time}sec' )
print(f"NeMo ASR-based methods {nemo_inf_time}sec")

# rMatchBoxNet-3x2x112: 0.017902135848999023sec
# NeMo ASR-based methods 0.38150715827941895sec

#  21.345
# rMatchBoxNet-3x2x112: 0.015993118286132812sec
# NeMo ASR-based methods 0.3836069107055664sec