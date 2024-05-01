"""Dataset class for MatchBoxNet regression model"""
import os
from torch.utils.data import Dataset
import torch
import torchaudio
import glob
from config import config
import torchaudio.transforms as T

config_feature = config['feature']

class AudioDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.input_len = config['input_len']
        self.file_list = glob.glob(os.path.join(data_dir, '**/*.wav'), recursive=True)
        self.transform = transform
        self.mfcc_transform =  T.MFCC(
            sample_rate=config['sample_rate'],
            n_mfcc=config_feature['mfcc_num_mel_bins'], 
            melkwargs={
                "n_fft": config_feature['window_size_ms']*config['sample_rate']//1000, # length of the FFT window
                "n_mels": config_feature['mfcc_num_mel_bins'],
                "hop_length": config_feature['window_stride']*config['sample_rate']//1000,
                "mel_scale": "htk",
            }
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        waveform = torch.squeeze(waveform)
        waveform = self.right_pad(waveform)

        if self.transform is not None:
            waveform = self.transform(waveform)

        # normalization
        mean = torch.mean(waveform)
        std = torch.std(waveform)

        waveform = waveform - mean
        waveform = waveform / std

        mfcc = self.mfcc_transform(waveform)

        file_name = file_path.split(os.path.sep)[-1]
        label = file_name.split('.')[0].split('_')[0]
        label = torch.tensor(int(label), dtype=torch.float32)
        return mfcc, label.view(1)

    def right_pad(self, wave):
        signal_length = wave.shape[0]
        # Can not be bigger as we took dataset max length as input_len
        if signal_length < self.input_len:
            pad_size = self.input_len - signal_length
            wave = torch.nn.functional.pad(wave, pad=(0, pad_size))
        return wave
