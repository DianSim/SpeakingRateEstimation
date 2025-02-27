# inference.py
import model
from config import config
import torch

import argparse
import librosa
from config import config
import torchaudio.transforms as T


config_feature = config['feature']


def inference(model, audio_path):
    """The function implements inference part of the model 
    and returns number of syllables and speaking rate of the given audio"""

    # load audio
    try:
        audio, sr = librosa.load(audio_path, sr=config['sample_rate'])

        # -----------------------data preprocessing-----------------------
        audio = torch.from_numpy(audio)
        mean = torch.mean(audio)
        std = torch.std(audio)

        audio = audio - mean
        audio = audio / std

        chunks = list(torch.split(audio, config['input_len']))

        # # testing correctness
        # if (chunks[0] != audio[:config['input_len']]).sum() == 0:
        #     print('Correct chunking')

        # pad last chunck with 0s if necessary
        if chunks[-1].shape[0] < config['input_len']:
            pad_size = config['input_len'] - chunks[-1].shape[0]
            last_chunk = torch.nn.functional.pad(chunks[-1], pad=(0, pad_size))
            chunks[-1] = last_chunk

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

        chunck_mfccs = [mfcc_transform(chunk) for chunk in chunks]
        batch_of_chuncks = torch.stack(chunck_mfccs, dim=0)


        model.eval()
        with torch.no_grad():
            pred = model(batch_of_chuncks)

        return {'syl_count': pred.sum(), 'speaking_rate': pred.sum()/(audio.shape[0]/sr)}

    except Exception as e:
        print("An error occurred:", e)
        return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inference script for the Speaking Rate Estimation model.")
    parser.add_argument('--audio', dest="audio_path", type=str, help="The path to the audio file to compute the speaking rate for.")
    args = parser.parse_args()

    # -----------------------inference-----------------------
    model = model.MatchBoxNetreg(B=3, R=2, C=112)
    path = '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/SREregression/models/rMatchBoxNet-3x2x112/checkpoints/best-epoch=198-val_loss=1.50-val_pcc=0.93.ckpt'

    state_dict = torch.load(path)
    model.load_state_dict(state_dict['state_dict'])
   

    pred = inference(model, args.audio_path)
    print('Syl count: ' , pred['syl_count'])
    print('Speaking rate: ', pred['speaking_rate'])
